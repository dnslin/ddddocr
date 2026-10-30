import base64
import contextlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import unittest

from PIL import Image
import uvicorn

from ddddocr.api.app import app


class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).resolve().parents[1] / "samples" / "yzm1.png"
        cls.image = fixture.read_bytes()
        cls.ocr_image = (fixture.parent / "ocr_digits.png").read_bytes()
        cls.color_image = (fixture.parent / "ocr_colors.png").read_bytes()
        with Image.open(fixture) as image:
            cls.image_width, cls.image_height = image.size

        cls.listener = socket.socket()
        cls.listener.bind(("127.0.0.1", 0))
        cls.port = cls.listener.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        cls.thread = threading.Thread(
            target=cls.server.run,
            kwargs={"sockets": [cls.listener]},
            daemon=True,
        )
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started:
            if not cls.thread.is_alive() or time.monotonic() >= deadline:
                cls.server.should_exit = True
                cls.listener.close()
                raise RuntimeError("Detection API server did not start")
            time.sleep(0.01)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.listener.close()
        if cls.thread.is_alive():
            raise RuntimeError("Detection API server did not stop")

    def post(self, path, content, content_type):
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)) as connection:
            connection.request("POST", path, body=content, headers={"Content-Type": content_type})
            response = connection.getresponse()
            return response.status, json.loads(response.read())

    def assert_detection(self, status, body):
        self.assertEqual(status, 200, body)
        boxes = body["result"]
        self.assertIsInstance(boxes, list)
        for box in boxes:
            self.assertEqual(len(box), 4)
            self.assertTrue(all(isinstance(value, int) for value in box))
            x1, y1, x2, y2 = box
            self.assertTrue(0 <= x1 < x2 <= self.image_width)
            self.assertTrue(0 <= y1 < y2 <= self.image_height)
        # The fixture's first green digit covers this point. Empty or misplaced
        # detections must not make a successful HTTP response look correct.
        self.assertTrue(any(x1 <= 90 <= x2 and y1 <= 110 <= y2 for x1, y1, x2, y2 in boxes))

    def test_base64_image_returns_valid_detection_boxes(self):
        content = json.dumps({"image": base64.b64encode(self.image).decode("ascii")}).encode()
        self.assert_detection(*self.post("/det?show_ad=false", content, "application/json"))

    def test_uploaded_image_returns_valid_detection_boxes(self):
        boundary = "ddddocr-detection-regression"
        content = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="yzm1.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + self.image + f"\r\n--{boundary}--\r\n".encode()
        self.assert_detection(*self.post(
            "/det/file?show_ad=false", content, f"multipart/form-data; boundary={boundary}"
        ))

    def post_file(self, path, image, fields=None):
        boundary = "ddddocr-api-regression"
        content = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + image + b"\r\n"
        for name, value in (fields or {}).items():
            content += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        content += f"--{boundary}--\r\n".encode()
        return self.post(path, content, f"multipart/form-data; boundary={boundary}")

    def post_ocr(self, upload=False, image=None, **options):
        image = self.ocr_image if image is None else image
        if upload:
            fields = {name: json.dumps(value) for name, value in options.items()}
            return self.post_file("/ocr/file?show_ad=false", image, fields)
        content = {"image": base64.b64encode(image).decode("ascii"), **options}
        return self.post("/ocr?show_ad=false", json.dumps(content).encode(), "application/json")

    def assert_ocr_text(self, status, body):
        self.assertEqual(status, 200, body)
        self.assertEqual(body["result"], "1234")
        self.assertGreaterEqual(body["processing_time"], 0)

    def test_base64_ocr_returns_expected_text(self):
        self.assert_ocr_text(*self.post_ocr())

    def test_uploaded_ocr_returns_expected_text(self):
        self.assert_ocr_text(*self.post_ocr(upload=True))

    def test_ocr_probability_returns_text_and_scores(self):
        for upload in (False, True):
            with self.subTest(upload=upload):
                status, body = self.post_ocr(upload=upload, probability=True)
                self.assertEqual(status, 200, body)
                result = body["result"]
                self.assertEqual(result["text"], "1234")
                self.assertTrue(result["probabilities"])
                self.assertTrue(result["charset"])
                self.assertTrue(0 <= result["confidence"] <= 1)

    def test_ocr_accepts_preset_color_filter(self):
        for upload in (False, True):
            with self.subTest(upload=upload):
                self.assert_ocr_text(*self.post_ocr(upload=upload, image=self.color_image, colors=["green"]))

    def test_ocr_accepts_custom_hsv_ranges(self):
        ranges = {"green": [[40, 50, 50], [80, 255, 255]]}
        for upload in (False, True):
            with self.subTest(upload=upload):
                self.assert_ocr_text(*self.post_ocr(upload=upload, image=self.color_image, custom_color_ranges=ranges))

    def test_json_apis_reject_invalid_base64(self):
        for path in ("/ocr", "/det"):
            with self.subTest(path=path):
                status, body = self.post(path, b'{"image":"not-base64"}', "application/json")
                self.assertEqual(status, 422, body)

    def test_file_apis_reject_empty_images(self):
        for path in ("/ocr/file", "/det/file"):
            with self.subTest(path=path):
                status, body = self.post_file(path, b"")
                self.assertEqual(status, 400, body)

    def test_ocr_rejects_invalid_image_contents(self):
        payload = json.dumps({"image": base64.b64encode(b"not an image").decode()}).encode()
        status, body = self.post("/ocr", payload, "application/json")
        self.assertEqual(status, 400, body)
        status, body = self.post_file("/ocr/file", b"not an image")
        self.assertEqual(status, 400, body)

    def test_uploaded_ocr_rejects_invalid_form_options(self):
        for fields in ({"colors": "not-json"}, {"custom_color_ranges": "not-json"}, {"probability": "maybe"}):
            with self.subTest(fields=fields):
                status, body = self.post_file("/ocr/file", self.ocr_image, fields)
                self.assertEqual(status, 400, body)


if __name__ == "__main__":
    unittest.main()
