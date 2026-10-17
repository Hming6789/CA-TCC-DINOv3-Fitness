import argparse
import json
import sys

import torch

from config_files.FitnessVideo_Configs import Config
from dataloader.video import load_video
from models.model import base_Model


# NEW: Dự đoán dùng đúng encoder + adapter + classification head đã huấn luyện.
def load_classifier(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "video_config" not in checkpoint:
        raise ValueError("Expected a FitnessVideo checkpoint saved by Trainer")
    if checkpoint.get("training_mode") in ("self_supervised", "SupCon"):
        raise ValueError("Classification head is not trained; run supervised or ft_1p first")
    configs = Config()
    for key, value in checkpoint["video_config"].items():
        setattr(configs, key, value)
    model = base_Model(configs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), configs


def main():
    parser = argparse.ArgumentParser(description="DINOv3 + CA-TCC video classification")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"Python: {sys.executable}")
    model, configs = load_classifier(args.checkpoint, torch.device(args.device))
    clip = load_video(args.video, configs.num_frames, configs.frame_size).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(clip)
        probabilities = logits.softmax(dim=1)[0].cpu()
    index = int(probabilities.argmax())
    print(json.dumps({"predicted_class": configs.class_names[index],
                      "probabilities": dict(zip(configs.class_names, probabilities.tolist()))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
