"""NEW: Smoke test bằng dữ liệu giả và checkpoint thật; không đo chất lượng nhận diện."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config_files.FitnessVideo_Configs import Config
from config_files.HAR_Configs import Config as HARConfig
from dataloader.dataloader import Load_Dataset, data_generator
from dataloader.video import VideoDataset
from models.model import base_Model
from models.TC import TC
from trainer.trainer import model_train, model_evaluate


def assert_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Expected invalid input to be rejected")


def check_original(device):
    config = HARConfig()
    config.batch_size = 2
    samples = torch.randn(2, 9, 128)
    dataset = Load_Dataset({"samples": samples, "labels": torch.tensor([0, 1])},
                           config, "self_supervised")
    model = base_Model(config).to(device)
    tc = TC(config, device).to(device)
    logits, features = model(samples.to(device))
    assert logits.shape == (2, 6) and features.shape == (2, 128, 18)
    assert "conv_block1.0.weight" in model.state_dict()
    loss, _ = model_train(model, tc, torch.optim.Adam(model.parameters()),
                          torch.optim.Adam(tc.parameters()), nn.CrossEntropyLoss(),
                          DataLoader(dataset, batch_size=2), config, device, "self_supervised")
    assert torch.isfinite(loss)
    assert_gradient(model.conv_block1)
    assert_gradient(tc.Wk)
    return {"logits": list(logits.shape), "features": list(features.shape), "backward": "passed"}


def run_cli_pipeline(root, config, device):
    # Tất cả dữ liệu/video/checkpoint ở thư mục tạm, không lẫn với dataset thật.
    import cv2

    with tempfile.TemporaryDirectory(prefix="catcc_smoke_") as temp:
        temp = Path(temp)
        data_dir = temp / "FitnessVideo"
        data_dir.mkdir()
        samples = torch.randint(0, 256, (5, config.num_frames, 3, 48, 48), dtype=torch.uint8)
        labels = torch.tensor([0, 1, 2, 0, 1])
        for split in ("train", "val", "test"):
            n = 5 if split == "train" else 3
            torch.save({"samples": samples[:n], "labels": labels[:n]}, data_dir / f"{split}.pt")
        torch.save({"samples": samples[:2], "labels": labels[:2]}, data_dir / "train_1perc.pt")
        video_path = data_dir / "synthetic.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 48))
        assert writer.isOpened(), "Cannot create synthetic video fixture"
        try:
            for frame in samples[0]:
                rgb = frame.permute(1, 2, 0).numpy()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        manifest = data_dir / "fixture.json"
        manifest.write_text(json.dumps([{"path": "synthetic.avi", "label": 0}]), encoding="utf-8")
        decoded = VideoDataset(manifest, config, "supervised")[0][0]
        assert decoded.shape == (18, 3, 224, 224) and torch.isfinite(decoded).all()
        # Kiểm tra chọn manifest trong data_generator, không chỉ gọi decoder riêng lẻ.
        (data_dir / "val.json").write_text(manifest.read_text(), encoding="utf-8")
        _, validation, _ = data_generator(data_dir, config, "supervised")
        assert len(validation.dataset) == 1

        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        def run(command, name):
            result = subprocess.run([sys.executable, *command], cwd=root, env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=300)
            if result.returncode:
                raise RuntimeError(f"{name} failed:\n{result.stdout}\n{result.stderr}")
            print(f"CLI {name}: PASS", flush=True)
            return result.stdout

        common = ["main.py", "--selected_dataset", "FitnessVideo", "--backbone_type", "dinov3",
                  "--data_path", str(temp), "--logs_save_dir", str(temp / "logs"),
                  "--experiment_description", "smoke", "--run_description", "synthetic",
                  "--epochs", "1", "--batch_size", "2", "--device", str(device)]
        modes = ["self_supervised", "ft_1p", "gen_pseudo_labels", "SupCon", "ft_SupCon_1p"]
        for mode in modes:
            run([*common, "--training_mode", mode], mode)
        pseudo = torch.load(data_dir / "pseudo_train_data.pt", weights_only=True)
        assert len(pseudo["labels"]) == 5, "Pseudo labels must include the incomplete final batch"
        checkpoint = temp / "logs/smoke/synthetic/ft_SupCon_1p_seed_0/saved_models/ckp_last.pt"
        prediction = run(["predict_video.py", "--video", str(video_path),
                          "--checkpoint", str(checkpoint), "--device", str(device)], "prediction")
        output = json.loads(prediction[prediction.index("{"):])
        assert set(output["probabilities"]) == set(config.class_names)
        assert abs(sum(output["probabilities"].values()) - 1) < 1e-5
        from predict_video import load_classifier
        ssl_checkpoint = checkpoint.parents[2] / "self_supervised_seed_0/saved_models/ckp_last.pt"
        expect_value_error(lambda: load_classifier(ssl_checkpoint, device))
        return {"stages": modes + ["prediction"], "manifest_decode": "passed",
                "pseudo_labels_count": len(pseudo["labels"]), "synthetic_only": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage", choices=["backbone", "full"], default="full")
    parser.add_argument("--cli", action="store_true", help="Also exercise main.py semi-supervised stages")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(f"Python: {sys.executable}", flush=True)
    device = torch.device(args.device)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(4)
    config = Config()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model = base_Model(config).to(device)
    model.train()
    assert not model.encoder.backbone.training
    image = torch.randint(0, 256, (1, 1, 3, 64, 64), dtype=torch.uint8)
    feature = model.encoder(image)
    assert feature.shape == (1, 1, 384) and not feature.requires_grad
    # Dữ liệu uint8 và float phải chuẩn hóa tương đương, không rescale hai lần.
    float_feature = model.encoder(image.float() / 255)
    torch.testing.assert_close(feature, float_feature, rtol=3e-3, atol=3e-3)
    report = {"synthetic_only": True, "python": sys.version.split()[0],
              "environment": Path(sys.executable).parent.name,
              "torch": torch.__version__, "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "image_features": [1, 384], "preprocessing_uint8_float": "passed"}
    print("Test 1 - real DINOv3 checkpoint / 384-dim image: PASS", flush=True)

    if args.stage == "full":
        source = {"samples": torch.randint(0, 256, (2, 18, 3, 64, 64), dtype=torch.uint8),
                  "labels": torch.tensor([0, 1])}
        dataset = VideoDataset(source, config, "self_supervised")
        loader = DataLoader(dataset, batch_size=2)
        data, _, _, _ = next(iter(loader))
        observed_chunks = []
        hook = model.encoder.backbone.register_forward_pre_hook(
            lambda module, args, kwargs: observed_chunks.append(kwargs["pixel_values"].shape[0]),
            with_kwargs=True)
        frame_features = model.encoder(data)
        hook.remove()
        assert frame_features.shape == (2, 18, 384)
        assert max(observed_chunks) <= config.frame_batch_size and sum(observed_chunks) == 36
        features = model.temporal_adapter(frame_features)
        assert features.shape == (2, config.final_out_channels, config.features_len)
        assert features.device == device and features.dtype == torch.float32
        expect_value_error(lambda: model.temporal_adapter(frame_features[:, :6]))
        expect_value_error(lambda: model.encoder(data[:, :, :1]))
        print("Tests 2/3 - video extraction / temporal adapter: PASS", flush=True)

        tc = TC(config, device).to(device)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.lr)
        tc_optimizer = torch.optim.Adam(tc.parameters(), lr=config.lr)
        modes = []
        for mode in ("self_supervised", "SupCon", "supervised"):
            before = model.temporal_adapter.projection[0].weight.detach().clone()
            loss, _ = model_train(model, tc, optimizer, tc_optimizer, nn.CrossEntropyLoss(),
                                  loader, config, device, mode)
            assert torch.isfinite(loss)
            assert_gradient(model.temporal_adapter)
            assert not torch.equal(before, model.temporal_adapter.projection[0].weight)
            assert all(p.grad is None for p in model.encoder.backbone.parameters())
            assert not model.encoder.backbone.training
            if mode != "supervised":
                assert_gradient(tc.Wk)
                assert_gradient(tc.seq_transformer)
                assert_gradient(tc.projection_head)
            else:
                assert_gradient(model.logits)
            modes.append(mode)
            print(f"Test 4 - trainer {mode}: forward/backward/optimizer PASS", flush=True)
        evaluation = model_evaluate(model, tc, loader, device, "supervised")
        assert torch.isfinite(evaluation[0]) and len(evaluation[2]) == 2
        report.update(video_features=list(frame_features.shape), temporal_features=list(features.shape),
                      logits=[2, 3], trainer_modes=modes, gradients="passed",
                      frozen_backbone="passed", evaluation="passed", max_frame_chunk=max(observed_chunks))
        report["original_backbone"] = check_original(device)
        print("Original CNN regression: PASS", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["gpu_memory_mib"] = {
            "peak_allocated": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            "peak_reserved": round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
            "total": round(torch.cuda.get_device_properties(device).total_memory / 2**20, 2)}
        print("Test 5 - GPU memory:", report["gpu_memory_mib"], flush=True)
    if args.cli:
        # Giải phóng model trước khi tiến trình CLI con sử dụng cùng GPU 4GB.
        del model, feature, float_feature
        if args.stage == "full":
            del tc, optimizer, tc_optimizer, features, frame_features, before
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report["cli_pipeline"] = run_cli_pipeline(Path(__file__).resolve().parent, config, device)
    report["status"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
