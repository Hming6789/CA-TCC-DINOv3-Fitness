from contextlib import nullcontext

import torch
from torch import nn


# NEW: Backbone ảnh đóng băng; chỉ adapter và các module CA-TCC được học mặc định.
class DINOv3Backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.frozen = configs.freeze_dinov3
        self.frame_batch_size = configs.frame_batch_size
        if self.frame_batch_size < 1:
            raise ValueError("frame_batch_size must be positive")
        self.processor = AutoImageProcessor.from_pretrained(
            configs.dinov3_model_id, local_files_only=configs.local_files_only)
        self.backbone = AutoModel.from_pretrained(
            configs.dinov3_model_id, local_files_only=configs.local_files_only)
        self.feature_dim = self.backbone.config.hidden_size
        if self.feature_dim != 384:
            raise ValueError("This integration requires DINOv3 ViT-S/16 (384 features)")
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, video):
        if video.ndim != 5 or video.shape[2] != 3 or min(video.shape) < 1:
            raise ValueError("Expected RGB video (B, T, 3, H, W)")
        is_uint8 = video.dtype == torch.uint8
        if not is_uint8 and (not video.is_floating_point() or
                             not torch.isfinite(video).all() or
                             video.min() < 0 or video.max() > 1):
            raise ValueError("Video must be uint8 [0,255] or float [0,1], before normalization")
        device = next(self.backbone.parameters()).device
        batch, frames = video.shape[:2]
        flat_frames = video.reshape(-1, *video.shape[2:])
        features = []
        # (B,T,3,H,W) -> các nhóm frame nhỏ -> (B,T,384); giữ nguyên thứ tự thời gian.
        for chunk in flat_frames.split(self.frame_batch_size):
            inputs = self.processor(
                images=chunk, do_rescale=is_uint8, return_tensors="pt",
                input_data_format="channels_first", device=device)
            pixels = inputs["pixel_values"].to(device)
            grad_context = torch.no_grad() if self.frozen else nullcontext()
            amp_context = (torch.autocast("cuda", dtype=torch.float16)
                           if self.frozen and device.type == "cuda" else nullcontext())
            with grad_context, amp_context:
                outputs = self.backbone(pixel_values=pixels)
                # Token CLS biểu diễn toàn ảnh; không lấy register/patch token làm frame.
                features.append(outputs.last_hidden_state[:, 0, :].float())
        return torch.cat(features, dim=0).reshape(batch, frames, self.feature_dim)


class TemporalAdapter(nn.Module):
    def __init__(self, feature_dim, configs):
        super().__init__()
        self.output_length = configs.features_len
        if not 0 < configs.TC.timesteps < self.output_length:
            raise ValueError("Require 0 < TC.timesteps < features_len")
        # NEW: C và L lấy từ cấu hình CA-TCC, không gán cứng theo DINOv3.
        self.projection = nn.Sequential(
            nn.Conv1d(feature_dim, configs.final_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.output_length)

    def forward(self, frame_features):
        if frame_features.ndim != 3 or frame_features.shape[1] < self.output_length:
            raise ValueError("Expected (B,T,D) with T >= configs.features_len")
        # (B,T,384) -> (B,384,T) -> (B,C,L), tương thích TC và logits gốc.
        return self.pool(self.projection(frame_features.transpose(1, 2)))
