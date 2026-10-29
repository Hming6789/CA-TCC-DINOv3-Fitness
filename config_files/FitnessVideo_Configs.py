from .HAR_Configs import Config as HARConfig


# NEW: Giữ C=128, L=18, TC.hidden_dim=100 và TC.timesteps=6 của HAR.
class Config(HARConfig):
    def __init__(self):
        super().__init__()
        self.backbone_type = "dinov3"
        self.dinov3_model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.freeze_dinov3 = True
        self.local_files_only = True
        self.frame_batch_size = 2
        self.num_frames = self.features_len
        self.frame_size = 224
        self.class_names = ["Squat", "Push-up", "Lunge"]
        self.num_classes = len(self.class_names)
        self.batch_size = 2
        self.num_workers = 0
        # NEW: Fine-tune một block cuối trên GPU 4GB, LR nhỏ hơn adapter/head.
        self.unfreeze_last_blocks = 1
        self.backbone_lr = 1e-5
        self.grad_clip_norm = 1.0
        self.finetune_ce_weight = 1.0
        self.finetune_temporal_weight = 0.1
        self.finetune_context_weight = 0.1
