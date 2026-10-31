"""Export a trained FitnessVideo classifier into two CPU-serving ONNX graphs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import torch
from torch import nn

from predict_video import load_classifier
from .inference import file_hash


class FrameEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values):
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]


class TemporalHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.adapter = model.temporal_adapter
        self.classifier = model.logits

    def forward(self, frame_features):
        features = self.adapter(frame_features)
        return self.classifier(features.flatten(1))


def export_graphs(model, directory):
    """Export and numerically check graphs. This does not certify training accuracy."""
    import onnx
    import onnxruntime as ort
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    # Eager attention avoids SDPA exporter/version differences on CPU.
    model.encoder.backbone.set_attn_implementation("eager")
    encoder = FrameEncoder(model.encoder.backbone).eval()
    head = TemporalHead(model).eval()
    generator = torch.Generator().manual_seed(42)
    pixels = torch.randn(1, 3, 224, 224, generator=generator)
    features = torch.randn(1, 18, 384, generator=generator)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    errors = {}
    with torch.no_grad():
        for name, module, sample, input_name, output_name in (
            ("encoder.onnx", encoder, pixels, "pixel_values", "embedding"),
            ("head.onnx", head, features, "frame_features", "logits"),
        ):
            path = directory / name
            torch.onnx.export(module, sample, str(path), input_names=[input_name],
                              output_names=[output_name], opset_version=17, dynamo=False)
            onnx.checker.check_model(str(path))
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {input_name: sample.numpy()})[0]
            expected = module(sample).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=5e-4)
            errors[name] = float(np.abs(actual - expected).max())
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model_artifacts")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    # Require an explicitly recorded supervised training mode, rather than a random head.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("training_mode") not in {"supervised", "ft_1p", "train_linear", "dinov3_finetune"}:
        raise ValueError("Export requires a supervised/fine-tuned classification checkpoint.")
    del saved
    model, configs = load_classifier(checkpoint, torch.device("cpu"))
    if model.backbone_type != "dinov3" or configs.num_frames != 18 or configs.frame_size != 224 or configs.num_classes != 3:
        raise ValueError("This serving profile supports three-class DINOv3 ViT-S/16 with 18 x 224 RGB frames.")
    processor = model.encoder.processor
    if (processor.size != {"height": 224, "width": 224} or
            getattr(processor, "do_center_crop", False) or getattr(processor, "do_pad", False) or
            not processor.do_normalize or not processor.do_rescale or
            not np.isclose(processor.rescale_factor, 1 / 255)):
        raise ValueError("The checkpoint preprocessing differs from the supported web input recipe.")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing model files are never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging)
        errors = export_graphs(model, stage)
        manifest = {"format_version": 1, "architecture": "dinov3-catcc", "purpose": "trained_classifier",
                    "model_version": args.version, "class_names": list(configs.class_names),
                    "num_frames": 18, "frame_size": 224, "feature_dim": 384,
                    "image_mean": list(processor.image_mean), "image_std": list(processor.image_std),
                    "checkpoint_sha256": file_hash(checkpoint), "created_at": datetime.now(timezone.utc).isoformat(),
                    "export_max_abs_error": errors,
                    "sha256": {name: file_hash(stage / name) for name in ("encoder.onnx", "head.onnx")}}
        (stage / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output.mkdir(exist_ok=True)
        for file in stage.iterdir():
            shutil.copy2(file, output / file.name)
    bundle = output.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("encoder.onnx", "head.onnx", "model.json"):
            archive.write(output / name, name)
    print(json.dumps({"directory": str(output), "bundle": str(bundle), "bundle_sha256": file_hash(bundle),
                      "export_max_abs_error": errors}, indent=2))


if __name__ == "__main__":
    main()
