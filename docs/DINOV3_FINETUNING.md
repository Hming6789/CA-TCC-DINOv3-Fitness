# Cập nhật loss và fine-tune DINOv3 — 30/10/2026

`dinov3_finetune` học đồng thời block cuối của DINOv3 ViT-S/16, Temporal Adapter, classification head và Temporal Contrasting của CA-TCC. Các chế độ cũ vẫn có thể dùng để so sánh.

## Loss được nối trực tiếp vào trainer

Trong `models/loss.py::DINOv3CATCCLoss.forward`:

```text
L_CE       = (CE(logits_view1, labels) + CE(logits_view2, labels)) / 2
L_temporal = (TC(view1, view2) + TC(view2, view1)) / 2
L_context  = SupCon(normalize(context_view1), normalize(context_view2), labels)
L_total    = w_ce * L_CE + w_temporal * L_temporal + w_context * L_context
```

Mặc định `w_ce=1.0`, `w_temporal=0.1`, `w_context=0.1`. Context lấy từ projection head TC. TC nhận đặc trưng `(B,128,18)` đã chuẩn hóa; CE nhận logits chưa softmax `(B,3)`. Hai view giữ cùng nhãn và thứ tự thời gian. Trọng số là cấu hình khởi đầu, có thể điều chỉnh qua CLI.

Đường gradient: loss → head/TC → adapter → block cuối DINOv3. Prefix còn lại đóng băng. Đây là objective fine-tune downstream cho CA-TCC. Quy trình pretraining DINOv3 gốc gồm teacher/student và các loss DINO/iBOT/KoLeo/Gram, xem [Meta ssl_meta_arch.py](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py).

## File và hàm để giảng viên đối chiếu

| File | Thay đổi |
|---|---|
| `models/loss.py::DINOv3CATCCLoss` | Công thức loss, kiểm tra trọng số, trả từng thành phần |
| `models/dinov3.py::DINOv3Backbone` | Chọn số block cuối được học; prefix eval; bật autograd cho fine-tune |
| `trainer/trainer.py::optimizer_groups` | Adam: LR adapter/head `3e-4`, backbone `1e-5` |
| `trainer/trainer.py::model_train` | Hai view → TC hai chiều → joint loss → backward → clip gradient → optimizer |
| `trainer/trainer.py::Trainer`, `training_checkpoint` | Log CE/temporal/context/total theo epoch; lưu best theo validation CE và last |
| `main.py` | Mode/CLI mới; nạp cả model và TC từ checkpoint khởi tạo |
| `config_files/FitnessVideo_Configs.py` | Số block, LR, trọng số loss và gradient clipping |
| `dataloader/video.py`, `dataloader/dataloader.py` | Augmentation hai view và batch≥2 cho mode mới |
| `test_dinov3_finetune.py` | Công thức loss, gradient từng nhánh, thay đổi trọng số, checkpoint và CLI |

## Kiểm thử ngay

Trong môi trường Conda đã cài:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
python test_dinov3_finetune.py --device cuda:0 --cli --output reports/finetune_gpu.json
python test_dinov3_finetune.py --device cpu --output reports/finetune_cpu.json
python smoke_test_dinov3_catcc.py --device cuda:0 --cli --output reports/regression_finetune_gpu.json
```

Các bài kiểm thử dùng video tổng hợp. Báo cáo GPU ghi 2 epoch: total loss `1.239198 → 1.131357`, block cuối thay đổi, prefix giữ nguyên, gradient head/adapter/TC đạt; checkpoint nạp lại cho logits tương đương. GPU peak allocated `544.74 MiB` của test bao gồm cả model nạp lại. CLI kiểm tra cả khởi tạo từ pretrained lẫn từ checkpoint CA-TCC. Đây là số đo kiểm thử kỹ thuật; không dùng để kết luận chất lượng nhận diện.

## Fine-tune khi có các split FitnessVideo

Giữ định dạng `data/FitnessVideo/train.json`, `val.json`, `test.json` hoặc dictionary tensor `.pt` như README.

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode dinov3_finetune --experiment_description FitnessVideo_experiments --run_description joint_loss --device cuda:0 --batch_size 2 --frame_batch_size 1 --unfreeze_last_blocks 1 --backbone_lr 1e-5 --ce_weight 1 --temporal_weight 0.1 --context_weight 0.1 --epochs 20
```

Để khởi tạo từ một checkpoint CA-TCC, thêm `--init_checkpoint DUONG_DAN_CHECKPOINT.pt`. Model và TC được nạp strict; mapping lớp phải khớp `Squat / Push-up / Lunge`. Đây là khởi tạo cho lượt fine-tune mới; optimizer và số epoch bắt đầu lại. Không cần tạo pseudo-label cho mode này: sử dụng nhãn train trực tiếp.

Kết quả trong `experiments_logs/FitnessVideo_experiments/joint_loss/dinov3_finetune_seed_0/`:

- `loss_history.json`: từng thành phần train loss; validation CE/accuracy.
- `saved_models/ckp_best.pt`: checkpoint có validation CE thấp nhất.
- `saved_models/ckp_last.pt`: checkpoint epoch cuối.

Validation dùng CE trên clip gốc, còn train dùng joint loss trên hai view; hai giá trị khác objective. Best checkpoint được chọn bằng validation CE, không bằng test set. Đánh giá cuối trong `main.py` dùng last; muốn dự đoán bằng best, truyền đường dẫn best cho `predict_video.py`.

```powershell
python predict_video.py --video data/FitnessVideo/clips/example.mp4 --checkpoint experiments_logs/FitnessVideo_experiments/joint_loss/dinov3_finetune_seed_0/saved_models/ckp_best.pt --device cuda:0
```

Checkpoint và dữ liệu ở thư mục gitignore; GitHub lưu mã nguồn, hướng dẫn và báo cáo JSON.
