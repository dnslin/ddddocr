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


class DetectionAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).resolve().parents[1] / "samples" / "yzm1.png"
        cls.image = fixture.read_bytes()
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


if __name__ == "__main__":
    unittest.main()
