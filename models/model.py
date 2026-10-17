from torch import nn


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        # MODIFIED: Giữ nguyên tên tham số CNN để đọc được checkpoint gốc.
        self.backbone_type = getattr(configs, "backbone_type", "original")
        if self.backbone_type == "dinov3":
            from .dinov3 import DINOv3Backbone, TemporalAdapter
            self.encoder = DINOv3Backbone(configs)
            self.temporal_adapter = TemporalAdapter(self.encoder.feature_dim, configs)
            self.logits = nn.Linear(configs.features_len * configs.final_out_channels,
                                    configs.num_classes)
            return
        if self.backbone_type != "original":
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        # MODIFIED: Hai backbone cùng trả về (logits, features) cho trainer hiện tại.
        if self.backbone_type == "dinov3":
            x = self.temporal_adapter(self.encoder(x_in))
        else:
            x = self.conv_block1(x_in)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
