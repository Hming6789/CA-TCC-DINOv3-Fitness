"""CPU-only serving for exported DINOv3 + CA-TCC models; no PyTorch on Render."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


class ModelUnavailable(Exception):
    pass


class InvalidVideo(ValueError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_video(path, num_frames=18, frame_size=224, max_duration=30):
    """Match dataloader.video.load_video: uniform rounded indices, OpenCV RGB resize."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or count < num_frames:
            raise InvalidVideo(f"Choose a readable video with at least {num_frames} frames.")
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise InvalidVideo("The video has invalid frame-rate or image metadata.")
        duration = count / fps
        if duration > max_duration or count > 18000 or width * height > 3840 * 2160:
            raise InvalidVideo(f"Use a clip up to {max_duration} seconds and 4K resolution.")
        indices = np.rint(np.linspace(0, count-1, num_frames)).astype(int).tolist()
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise InvalidVideo("Some sampled frames could not be decoded. Try an MP4 export.")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
        return np.stack(frames), {"duration_seconds": round(duration, 2), "fps": round(fps, 2),
                                  "width": width, "height": height, "total_frames": count,
                                  "sampled_frames": num_frames, "sample_indices": indices}
    finally:
        capture.release()


class MotionClassifier:
    def __init__(self, directory, threads=2):
        self.directory = Path(directory)
        self.threads = threads
        self.encoder = self.head = self.metadata = None
        self.error = None

    @property
    def ready(self):
        return self.encoder is not None and self.head is not None

    def load(self):
        manifest = self.directory / "model.json"
        if not manifest.exists():
            self.error = "Connect an exported model to enable video analysis."
            return
        try:
            import onnxruntime as ort
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if meta.get("format_version") != 1 or meta.get("architecture") != "dinov3-catcc":
                raise ValueError("Unsupported model package")
            labels = meta["class_names"]
            if not isinstance(labels, list) or len(labels) != 3 or len(set(labels)) != 3 or not all(isinstance(x, str) and 0 < len(x) < 80 for x in labels):
                raise ValueError("Expected three unique class labels")
            if meta["num_frames"] != 18 or meta["frame_size"] != 224 or meta["feature_dim"] != 384:
                raise ValueError("Expected the validated 18-frame ViT-S/16 export")
            if meta.get("purpose") != "trained_classifier":
                raise ValueError("Only a trained classifier package can be served")
            for name in ("encoder.onnx", "head.onnx"):
                if file_hash(self.directory / name) != meta["sha256"][name]:
                    raise ValueError(f"Model checksum mismatch: {name}")
            mean, std = np.asarray(meta["image_mean"]), np.asarray(meta["image_std"])
            if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid preprocessing metadata")
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, self.threads))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            encoder = ort.InferenceSession(str(self.directory / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
            head = ort.InferenceSession(str(self.directory / "head.onnx"), options, providers=["CPUExecutionProvider"])
            # Verify the graph contract before exposing a ready status.
            if encoder.get_inputs()[0].name != "pixel_values" or encoder.get_inputs()[0].shape != [1, 3, 224, 224]:
                raise ValueError("Encoder input schema mismatch")
            if head.get_inputs()[0].name != "frame_features" or head.get_inputs()[0].shape != [1, 18, 384]:
                raise ValueError("Temporal head input schema mismatch")
            self.encoder, self.head, self.metadata = encoder, head, meta
            self.error = None
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Could not load the configured model package")
            self.encoder = self.head = None
            self.error = "The configured model could not be loaded. Check the server configuration."

    def status(self):
        return {"ready": self.ready, "mode": "model" if self.ready else "preview",
                "model_name": "DINOv3 + CA-TCC", "runtime": "ONNX Runtime · CPU",
                "class_names": self.metadata["class_names"] if self.ready else ["Squat", "Push-up", "Lunge"],
                "num_frames": 18, "frame_size": 224,
                "model_version": self.metadata.get("model_version", "1") if self.ready else None,
                "message": self.error}

    def predict(self, path):
        if not self.ready:
            raise ModelUnavailable(self.error or "The classifier is not connected.")
        start = time.perf_counter()
        frames, video = sample_video(path)
        mean = np.asarray(self.metadata["image_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(self.metadata["image_std"], dtype=np.float32).reshape(1, 3, 1, 1)
        embeddings = []
        for rgb in frames:
            pixels = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            pixels = (pixels - mean) / std
            embedding = self.encoder.run(None, {"pixel_values": pixels})[0]
            if embedding.shape != (1, 384) or not np.isfinite(embedding).all():
                raise RuntimeError("Encoder returned invalid features")
            embeddings.append(embedding)
        features = np.stack(embeddings, axis=1).astype(np.float32)
        logits = self.head.run(None, {"frame_features": features})[0]
        if logits.shape != (1, 3) or not np.isfinite(logits).all():
            raise RuntimeError("Classifier returned invalid logits")
        scores = np.exp(logits[0] - logits[0].max())
        scores /= scores.sum()
        order = np.argsort(-scores)
        return {"predicted_class": self.metadata["class_names"][int(order[0])],
                "confidence": float(scores[order[0]]),
                "probabilities": [{"label": label, "probability": float(score)} for label, score in zip(self.metadata["class_names"], scores)],
                "inference_seconds": round(time.perf_counter()-start, 3), "video": video,
                "model": {"name": "DINOv3 + CA-TCC", "version": self.metadata.get("model_version", "1"), "runtime": "ONNX CPU"},
                "note": "Class probabilities describe exercise identity, not movement quality."}
