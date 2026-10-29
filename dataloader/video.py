import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# NEW: Lấy frame đều theo thời gian; RGB float [0,1], chưa chuẩn hóa ImageNet.
def load_video(path, num_frames, frame_size):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count < num_frames:
            raise ValueError(f"Video must contain at least {num_frames} readable frames: {path}")
        indices = torch.linspace(0, count - 1, num_frames).round().long().tolist()
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        return torch.stack(frames).float().div_(255)
    finally:
        capture.release()


def video_views(clip):
    # Hai góc nhìn thay đổi ánh sáng đồng nhất toàn clip, không đảo thứ tự frame.
    weak = (clip * torch.empty(()).uniform_(0.9, 1.1)).clamp(0, 1)
    contrast = torch.empty(()).uniform_(0.7, 1.3)
    brightness = torch.empty(()).uniform_(-0.1, 0.1)
    mean = clip.mean(dim=(-2, -1), keepdim=True)
    strong = ((clip - mean) * contrast + mean + brightness).clamp(0, 1)
    return weak, strong


class VideoDataset(Dataset):
    def __init__(self, source, configs, training_mode):
        self.configs = configs
        self.contrastive = training_mode in ("self_supervised", "SupCon", "dinov3_finetune")
        self.samples = None
        if configs.num_frames < configs.features_len:
            raise ValueError("num_frames must be >= features_len")
        if isinstance(source, dict):
            self.samples = torch.as_tensor(source["samples"]).cpu()
            self.labels = torch.as_tensor(source["labels"], dtype=torch.long).cpu()
            if self.samples.ndim != 5 or self.samples.shape[2] != 3:
                raise ValueError("Video samples must have shape (N,T,3,H,W)")
            if self.samples.shape[1] < configs.num_frames:
                raise ValueError("Video clips have fewer frames than num_frames")
            if self.labels.ndim != 1 or len(self.samples) != len(self.labels):
                raise ValueError("Expected one label per video clip")
        else:
            manifest = Path(source)
            self.records = json.loads(manifest.read_text(encoding="utf-8"))
            self.paths = [manifest.parent / item["path"] for item in self.records]
            self.labels = torch.tensor([item["label"] for item in self.records], dtype=torch.long)
        if not len(self.labels) or (self.labels < 0).any() or (self.labels >= configs.num_classes).any():
            raise ValueError("Dataset must be nonempty, with labels in [0, num_classes)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.samples is None:
            clip = load_video(self.paths[index], self.configs.num_frames, self.configs.frame_size)
        else:
            clip = self.samples[index]
            indices = torch.linspace(0, clip.shape[0] - 1, self.configs.num_frames).round().long()
            clip = clip[indices]
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255
            elif (not clip.is_floating_point() or not torch.isfinite(clip).all()
                  or clip.min() < 0 or clip.max() > 1):
                raise ValueError("Video samples must be uint8 [0,255] or float [0,1]")
            clip = F.interpolate(clip.float(), size=(self.configs.frame_size,) * 2,
                                 mode="bilinear", align_corners=False)
        aug1, aug2 = video_views(clip) if self.contrastive else (clip, clip)
        return clip, self.labels[index], aug1, aug2
