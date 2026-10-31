"""Integration tests use disposable toy graphs, never a production classifier."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np

from .app import BoundedVideoUpload, MAX_UPLOAD, create_app
from .inference import MotionClassifier, file_hash, sample_video
from .prepare_model import install_bundle


def toy_package(root):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    for name, input_name, input_shape, axis, in_features, out_features in (
        ("encoder.onnx", "pixel_values", [1, 3, 224, 224], [2, 3], 3, 384),
        ("head.onnx", "frame_features", [1, 18, 384], [1], 384, 3),
    ):
        weights = np.zeros((in_features, out_features), dtype=np.float32)
        bias = np.zeros(out_features, dtype=np.float32)
        if out_features == 3:
            bias[:] = [2, 0, -1]
        nodes = [helper.make_node("ReduceMean", [input_name], ["mean"], axes=axis, keepdims=0),
                 helper.make_node("MatMul", ["mean", "w"], ["product"]),
                 helper.make_node("Add", ["product", "b"], ["output"])]
        graph = helper.make_graph(nodes, "disposable-integration-fixture",
                                  [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)],
                                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_features])],
                                  [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        onnx.save(model, root / name)
    meta = {"format_version": 1, "architecture": "dinov3-catcc", "purpose": "trained_classifier",
            "model_version": "DISPOSABLE-TEST-FIXTURE", "class_names": ["Squat", "Push-up", "Lunge"],
            "num_frames": 18, "frame_size": 224, "feature_dim": 384,
            "image_mean": [.485, .456, .406], "image_std": [.229, .224, .225],
            "sha256": {name: file_hash(root / name) for name in ("encoder.onnx", "head.onnx")}}
    (root / "model.json").write_text(json.dumps(meta), encoding="utf-8")


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def make_video(self):
        import cv2
        path = self.root / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 64))
        self.assertTrue(writer.isOpened())
        for index in range(50):
            writer.write(np.full((64, 64, 3), index * 4, np.uint8))
        writer.release()
        return path

    def test_preview_and_static_ui(self):
        engine = MotionClassifier(self.root)
        with TestClient(create_app(engine)) as client:
            self.assertFalse(client.get("/api/status").json()["ready"])
            response = client.get("/")
            self.assertIn("Understand every move", response.text)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            self.assertEqual(client.get("/static/app.js").status_code, 200)
            self.assertEqual(client.post("/api/analyze", files={"file": ("clip.mp4", b"video")}).status_code, 503)
            self.assertEqual(client.get("/healthz").status_code, 200)

    def test_onnx_video_api_and_cleanup(self):
        toy_package(self.root)
        path = self.make_video()
        engine = MotionClassifier(self.root)
        with TestClient(create_app(engine)) as client:
            self.assertTrue(client.get("/api/status").json()["ready"])
            seen = []
            original = engine.predict
            def track(video):
                seen.append(Path(video))
                return original(video)
            with patch.object(engine, "predict", side_effect=track):
                response = client.post("/api/analyze", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["predicted_class"], "Squat")
            self.assertAlmostEqual(sum(item["probability"] for item in result["probabilities"]), 1, places=6)
            self.assertEqual(result["video"]["sampled_frames"], 18)
            self.assertEqual(result["video"]["total_frames"], 50)
            self.assertTrue(seen)
            self.assertTrue(all(not file.exists() for file in seen))
            self.assertEqual(client.post("/api/analyze", files={"file": ("bad.mp4", b"not video")}).status_code, 422)
            self.assertEqual(client.post("/api/analyze", files={"file": ("empty.mp4", b"")}).status_code, 400)
            self.assertEqual(client.post("/api/analyze", files={"file": ("bad.txt", b"text")}).status_code, 415)
            self.assertEqual(client.post("/api/analyze", content=b"x", headers={"Content-Length": str(MAX_UPLOAD * 2)}).status_code, 413)

    def test_sampler_matches_training_loader(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Sampling parity needs the training environment's PyTorch installation")
        path = self.make_video()
        # Isolate PyTorch from ORT's native libraries, which can conflict on Windows.
        script = """
import sys
import torch
import numpy as np
from dataloader.video import load_video
from webapp.inference import sample_video
torch.set_num_threads(2)
actual, _ = sample_video(sys.argv[1])
expected = load_video(sys.argv[1], 18, 224).numpy().transpose(0, 2, 3, 1)
np.testing.assert_allclose(actual / 255.0, expected, atol=1e-7)
"""
        result = subprocess.run([sys.executable, "-c", script, str(path)],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_checksum_rejected(self):
        toy_package(self.root)
        with (self.root / "head.onnx").open("ab") as handle:
            handle.write(b"modified")
        engine = MotionClassifier(self.root)
        with self.assertLogs("webapp.inference", level="ERROR"):
            engine.load()
        self.assertFalse(engine.ready)

    def test_chunked_request_is_bounded_before_parser(self):
        async def check():
            async def inner(scope, receive, send):
                self.fail("Oversized request reached the parser")
            async def receive():
                return {"type": "http.request", "body": b"x" * (1024 * 1024), "more_body": True}
            sent = []
            async def send(message):
                sent.append(message)
            await BoundedVideoUpload(inner)({"type": "http", "method": "POST", "path": "/api/analyze", "headers": []}, receive, send)
            self.assertEqual(sent[0]["status"], 413)
        asyncio.run(check())

    def test_busy_server_returns_429(self):
        toy_package(self.root)
        engine = MotionClassifier(self.root)
        started, release = threading.Event(), threading.Event()
        def slow_predict(path):
            started.set()
            release.wait(5)
            return {"ok": True}
        with TestClient(create_app(engine)) as client, patch.object(engine, "predict", side_effect=slow_predict):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(client.post, "/api/analyze", files={"file": ("clip.mp4", b"video")})
                try:
                    self.assertTrue(started.wait(5))
                    second = client.post("/api/analyze", files={"file": ("clip.mp4", b"video")})
                    self.assertEqual(second.status_code, 429)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5).status_code, 200)

    def test_bundle_blocks_path_traversal_and_installs_exact_files(self):
        archive_path = self.root / "model.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("../outside.txt", "bad")
        with self.assertRaises(ValueError):
            install_bundle(archive_path, self.root / "installed")
        toy_package(self.root)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name in ("encoder.onnx", "head.onnx", "model.json"):
                archive.write(self.root / name, name)
        install_bundle(archive_path, self.root / "installed")
        self.assertEqual(file_hash(self.root / "head.onnx"), file_hash(self.root / "installed/head.onnx"))


if __name__ == "__main__":
    unittest.main()
