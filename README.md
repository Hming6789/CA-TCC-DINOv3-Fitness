# CA-TCC với DINOv3 cho AI Fitness Coach

Tích hợp **DINOv3 ViT-S/16** vào encoder của CA-TCC để học biểu diễn chuỗi video cho ba lớp **Squat / Push-up / Lunge**. Đây là phần thay backbone.

**Trạng thái:** đã kiểm thử kỹ thuật bằng checkpoint pretrained thật và dữ liệu giả. Chưa có kết quả huấn luyện/đánh giá trên video bài tập thật; không công bố Accuracy hoặc F1-score.

## 1. Encoder trước và sau thay đổi

**Cập nhật 30/10/2026 — loss và fine-tune DINOv3:** chế độ `dinov3_finetune` mở khóa block cuối ViT, dùng LR backbone riêng và loss `CE + 0.1 × Temporal + 0.1 × SupCon`. Xem [công thức, các file thay đổi và lệnh chạy](docs/DINOV3_FINETUNING.md). Kết quả kiểm thử: [GPU](reports/finetune_gpu.json), [CPU](reports/finetune_cpu.json).

Encoder gốc trong `models/model.py::base_Model` gồm ba khối Conv1d–BatchNorm–ReLU–MaxPool, có Dropout ở khối đầu. Với HAR: đầu vào `(B,9,128)`, features `(B,128,18)`, logits `(B,6)`.

Nhánh mới vẫn dùng `base_Model`, giữ giao diện `forward() -> (logits, features)`:

```text
Video RGB (B,T,3,H,W), mặc định T=18
    -> lấy frame đều theo thời gian, resize 224x224
    -> AutoImageProcessor của checkpoint (normalize; rescale đúng một lần)
    -> DINOv3 ViT-S/16 đóng băng, xử lý từng nhóm 2 frame
    -> token CLS từng frame (B,T,384)
    -> TemporalAdapter: transpose -> Conv1d(384,C,1) -> ReLU -> Dropout
                        -> AdaptiveAvgPool1d(L)
    -> features (B,C,L) = (B,128,18)
         |-- hai view: normalize -> TC hai chiều -> contextual loss NT-Xent / SupCon
         |-- phân loại: flatten -> Linear(C*L,3) -> logits (B,3)
```

**TC là nhánh học tương phản trong lúc huấn luyện**, không phải một tầng bắt buộc nằm nối tiếp classification head khi dự đoán. Các chế độ cũ giữ DINOv3 đóng băng. Chế độ mới `dinov3_finetune` huấn luyện block cuối DINOv3, adapter, classification head và TC cùng nhau bằng loss kết hợp.

`FitnessVideo_Configs.Config` kế thừa HAR thực tế: `final_out_channels=128`, `features_len=18`, `TC.hidden_dim=100`, `TC.timesteps=6`. Không giả định rằng DINOv3 tự tạo ra đặc trưng 128 chiều. Adapter đọc C/L từ cấu hình; yêu cầu `T >= L > TC.timesteps`. Mặc định T=L nên pooling không đổi chiều thời gian.

DINOv3 đóng băng dùng `eval()` + `no_grad()`, kể cả sau `model.train()`; GPU dùng autocast FP16 và trả features FP32. Trong `dinov3_finetune`, các block được chọn nhận gradient FP32, prefix đóng băng vẫn eval. Video và hai view ở CPU; backbone xử lý theo nhóm frame. Bộ tiền xử lý và token CLS theo [tài liệu Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3); checkpoint là [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

## 2. Các file đã cập nhật

| File | Class/hàm hoặc phần thay đổi | Mục đích |
|---|---|---|
| `models/loss.py` | `DINOv3CATCCLoss` | Kết hợp CE hai view, Temporal Contrasting hai chiều và SupCon; xuất từng thành phần loss |
| `test_dinov3_finetune.py`, `docs/DINOV3_FINETUNING.md` | Kiểm thử và hướng dẫn fine-tune | Xác nhận trọng số DINOv3 thay đổi, prefix giữ nguyên, checkpoint nạp lại đúng |
| `models/dinov3.py` — mới | `DINOv3Backbone`, `TemporalAdapter` | Load cache, preprocessing, CLS theo frame, freeze/chunk, chuyển sang (B,C,L) |
| `models/model.py` | `base_Model.__init__ / forward` | Chọn encoder original/dinov3, giữ CNN và tên tham số checkpoint cũ |
| `models/TC.py` | `TC.__init__ / forward` | Giữ TC, chỉ rõ chiều LogSoftmax, kiểm tra L, cấp phát theo device/dtype đầu vào |
| `config_files/FitnessVideo_Configs.py` — mới | `Config` | Kế thừa HAR, thêm cấu hình video, ba lớp, batch=2 và chunk=2 |
| `dataloader/video.py` — mới | `VideoDataset`, `load_video`, `video_views` | Đọc video RGB/manifest/tensor, lấy frame, tạo hai view giữ thứ tự thời gian |
| `dataloader/dataloader.py` | `Load_Dataset`, `data_generator` | Chọn loader đúng backbone; không nhầm batch nhỏ thành chiều kênh; giữ đủ val/test |
| `dataloader/augmentations.py` | `permutation` | Tương thích NumPy mới với đoạn không đều và bảo toàn từng kênh tín hiệu |
| `trainer/trainer.py` | `prepare_input`, `model_train`, `model_evaluate`, `Trainer`, `gen_pseudo_labels` | Video ở CPU; NT-Xent theo batch thực; lưu cấu hình checkpoint; pseudo data về CPU |
| `main.py` | Khởi tạo cấu hình, CLI, các nhánh training | Chọn backbone, batch/epoch/chunk; sửa `data_perc`; tạo đầy đủ pseudo-label |
| `utils.py` | `copy_Files` | Lưu các module mới vào thư mục thí nghiệm |
| `predict_video.py` — mới | `load_classifier`, `main` | Nạp checkpoint, xử lý video, xuất nhãn và xác suất |
| `smoke_test_dinov3_catcc.py` — mới | `main`, `check_original`, `run_cli_pipeline` | Test checkpoint thật, gradient, GPU/CPU, các giai đoạn CLI và suy luận |
| `requirements-dinov3.txt`, `.gitignore`, `.gitattributes` — mới | Phụ thuộc, quy tắc loại trừ và CRLF | Không nâng cấp torch; loại dữ liệu/token/checkpoint; giữ kiểu xuống dòng gốc |
| `reports/smoke_gpu.json`, `reports/smoke_cpu.json` — mới | Kết quả máy đọc được | Bằng chứng smoke test, không phải benchmark nhận diện |
| `README.md`, `docs/README_upstream.md` | Tài liệu tiếng Việt và bản gốc được giữ lại | Hướng dẫn nộp bài, bảo toàn nguồn và citation của tác giả |

Các phần thêm/sửa quan trọng có comment tiếng Việt và dấu `# NEW` / `# MODIFIED`. Không thay module attention hoặc công thức NT-Xent/SupCon. Tệp `test_dinov3.py` và thư mục `.vscode/` có sẵn được giữ nguyên.

## 3. Môi trường Windows / Conda

Môi trường dùng để kiểm thử: **catcc-dinov3**, Python 3.10.22, PyTorch 2.9.1+cu126, Transformers 4.57.6, NVIDIA GeForce RTX 3050 Laptop GPU 4GB.

Trong PowerShell:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available())"
python -m pip check
```

`sys.executable` phải trỏ vào `miniconda3\envs\catcc-dinov3\python.exe`. Nếu Conda chưa được khởi tạo trong terminal, dùng trực tiếp:

```powershell
& "$HOME\miniconda3\envs\catcc-dinov3\python.exe" -c "import sys; print(sys.executable)"
```

Chỉ khi thiếu thư viện, kiểm tra trước rồi cài trong môi trường đang active:

```powershell
python -m pip show transformers huggingface-hub einops opencv-python pandas scikit-learn openpyxl
python -m pip install -r requirements-dinov3.txt
```

Môi trường hiện tại đã đủ phụ thuộc; không cần cài lại, đổi môi trường base, tạo venv hay nâng cấp PyTorch. `requirements-dinov3.txt` không cài torch/torchvision; giữ bộ CUDA đang hoạt động. Checkpoint và processor đã có trong Hugging Face cache, mặc định `local_files_only=True`: không tải lại, không cần truyền token vào code. Trên máy khác cần chuẩn bị cache và quyền truy cập model theo quy định của Meta.

## 4. Kiểm thử lại, không cần dataset thật

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
# Test 1: checkpoint thật và vector ảnh 384 chiều
python smoke_test_dinov3_catcc.py --device cuda:0 --stage backbone

# Test 1-5 + main.py, fine-tune, pseudo-label, SupCon, dự đoán video
python smoke_test_dinov3_catcc.py --device cuda:0 --cli --output reports/smoke_gpu.json

# CPU: encoder, adapter, ba chế độ trainer và CNN gốc
python smoke_test_dinov3_catcc.py --device cpu --output reports/smoke_cpu.json
```

Test CLI sinh clip AVI giả, tensor và checkpoint trong thư mục tạm, kiểm tra rồi tự dọn. Không lưu model đã huấn luyện thực tế. Các chỉ số do trainer sinh ra trong test này chỉ thuộc dữ liệu giả; script không báo chúng như chất lượng nhận diện. Chạy lại bằng Python thông thường nếu cơ chế sandbox của công cụ chặn thư mục tạm.

## 5. Định dạng dữ liệu video

Chưa cung cấp dataset thật. Khi có dữ liệu, đặt trong `data/FitnessVideo/` (đã gitignore). Ưu tiên manifest để không nạp toàn bộ video vào RAM:

```text
data/FitnessVideo/
    train.json
    val.json
    test.json
    clips/...
```

Mỗi JSON là danh sách các clip, đường dẫn tương đối tính từ thư mục chứa manifest:

```json
[
  {"path": "clips/squat_001.mp4", "label": 0},
  {"path": "clips/pushup_001.mp4", "label": 1},
  {"path": "clips/lunge_001.mp4", "label": 2}
]
```

Mapping cố định: **0=Squat, 1=Push-up, 2=Lunge**. Mỗi clip chứa ít nhất 18 frame giải mã được. Loader lấy đều 18 frame toàn clip và resize 224×224; đây là baseline, chưa tối ưu crop người hay tốc độ chuyển động.

Ngoài JSON, loader hỗ trợ `train.pt / val.pt / test.pt` gồm dictionary:
- `samples`: torch Tensor `(N,T,3,H,W)`, T≥18, RGB `uint8 [0,255]` hoặc float `[0,1]`.
- `labels`: torch Tensor `(N,)`, kiểu long, giá trị 0–2.

Không chuẩn hóa ImageNet trước khi đưa vào loader. `.pt` được đọc với `weights_only=True`, vì vậy các file pickle chứa NumPy/object cũ cần được chuyển sang dictionary tensor ở nguồn đáng tin cậy. Nếu cùng tên có cả JSON và PT, JSON được ưu tiên.

Hai view dùng thay đổi độ sáng/tương phản đồng nhất theo clip, giữ thứ tự frame. Train contrastive cần batch≥2 do projection head có BatchNorm và loss cần mẫu âm. Val/test không bỏ batch cuối.

## 6. Chạy huấn luyện và dự đoán khi có dữ liệu thật

### Encoder DINOv3 qua chương trình CA-TCC

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode self_supervised --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0 --batch_size 2 --frame_batch_size 2 --epochs 40
```

Nếu thiếu VRAM, giảm `--frame_batch_size 1`; giữ batch contrastive ít nhất 2. `freeze_dinov3=True` là mặc định của các chế độ cũ. `dinov3_finetune` đã kiểm thử mở một block cuối với chunk=1; xem báo cáo riêng ở đầu README.

### Chuỗi semi-supervised CA-TCC

Dùng cùng `experiment_description`, `run_description` và seed để các giai đoạn tìm đúng checkpoint. Tạo `train_1perc.json` hoặc `train_1perc.pt` từ phần train có nhãn; không dùng val/test làm dữ liệu fine-tune.

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode ft_1p --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode gen_pseudo_labels --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode SupCon --data_perc 1 --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode ft_SupCon_1p --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
```

`gen_pseudo_labels` hiện theo luồng gốc: đọc checkpoint `ft_1p`, ghi `pseudo_train_data.pt`. Đường này gom clip đã resize vào RAM nên chưa phù hợp dataset video lớn. `SupCon --data_perc` chọn checkpoint fine-tune tương ứng; nếu dùng tỷ lệ khác cần tự chuẩn bị pseudo-label từ model tương ứng.

Dự đoán với classification head đã fine-tune:

```powershell
python predict_video.py --video data/FitnessVideo/clips/example.mp4 --checkpoint experiments_logs/FitnessVideo_experiments/dinov3/ft_SupCon_1p_seed_0/saved_models/ckp_last.pt --device cuda:0
```

Script nạp đúng cấu hình/nhãn đã lưu trong checkpoint. Checkpoint `self_supervised` hoặc `SupCon` bị từ chối vì chưa huấn luyện classification head. Checkpoint fine-tune từ smoke test vẫn chỉ cho kết quả minh họa.

Có thể chạy `--training_mode supervised` để huấn luyện trực tiếp từ nhãn, nhưng chế độ đó không sử dụng TC; bằng chứng tích hợp TC nằm ở test self-supervised/SupCon và chuỗi semi-supervised phía trên.

### Giữ encoder gốc

```powershell
python main.py --selected_dataset HAR --backbone_type original --training_mode self_supervised --device cuda:0
```

Lệnh này cần tensor HAR thật trong `data/HAR/`. CNN nhận chuỗi cảm biến; DINOv3 nhận video RGB. Không đưa keypoint YOLO `(B,C,L)` trực tiếp vào ViT. Chọn DINOv3 với HAR hoặc CNN với FitnessVideo sẽ báo lỗi rõ ràng thay vì tự biến đổi dữ liệu sai.

## 7. Kết quả kiểm thử thực tế

Ngày kiểm thử: **17/10/2026**. Nguồn: [GPU report](reports/smoke_gpu.json), [CPU report](reports/smoke_cpu.json).

| Kiểm thử | Đầu ra / nội dung xác minh |
|---|---|
| DINOv3 pretrained từ cache | CLS ảnh `(1,384)`; uint8/float được tiền xử lý nhất quán |
| Video nhiều frame | `(2,18,384)`; hook xác nhận mỗi lần tối đa 2 frame |
| Temporal Adapter | `(2,128,18)`, FP32, đúng device, từ chối clip quá ngắn |
| Head phân loại | logits `(2,3)` |
| Trainer self_supervised | TC hai chiều + NT-Xent; forward, backward và optimizer cập nhật adapter/TC |
| Trainer SupCon | TC hai chiều + SupCon; gradient hữu hạn và khác 0 |
| Trainer supervised / evaluate | Gradient adapter/head; inference trả đủ số mẫu |
| Đóng băng DINOv3 | Luôn eval, không gradient khi train |
| CNN gốc HAR | `(2,9,128) -> features (2,128,18), logits (2,6)`; backward thành công |
| CLI trên GPU | Self-supervised → ft_1p → pseudo-label → SupCon → ft_SupCon_1p → predict |
| Video decoding / checkpoint | Đọc AVI giả qua manifest; nạp checkpoint từ đĩa; 5 pseudo-label cho 5 clip |
| CPU | Các test encoder, adapter, ba chế độ trainer và CNN gốc |
| Bộ nhớ GPU của test chính | Peak allocated **124.19 MiB**, peak reserved **132 MiB**, tổng GPU **4095.5 MiB** |

Số bộ nhớ trên là thống kê allocator PyTorch trong tiến trình smoke test chính, không phải toàn bộ VRAM của driver, ứng dụng khác hoặc các tiến trình CLI con. Không xảy ra OOM với cấu hình kiểm thử.

**Chưa kiểm thử:** huấn luyện nhiều epoch trên dataset thật, độ chính xác/F1 thật, YOLO Pose + RGB fusion, dữ liệu lớn, mở toàn bộ backbone, multi-GPU, codec video ngoài AVI giả. Smoke test CPU không chạy lại chuỗi CLI nhiều giai đoạn.

## 8. Hướng phát triển AI Fitness Coach

Thu thập dữ liệu sau khi được phép; chia train/val/test theo người tập để tránh rò rỉ. Huấn luyện và đánh giá ba lớp trên video thật, rồi nghiên cứu ghép đặc trưng RGB của DINOv3 với keypoint YOLO Pose. Nhận diện loại bài tập, đếm rep và đánh giá tư thế là các bài toán khác nhau; thay backbone ở đây chưa giải quyết toàn bộ các bài toán đó.

Các cải tiến tiếp theo có thể gồm crop người, lấy cửa sổ thời gian, cache CLS để tăng tốc backbone đóng băng, và lưu pseudo-label bằng manifest để tiết kiệm RAM.

## 9. Nguồn gốc

- [CA-TCC — mã nguồn gốc](https://github.com/emadeldeen24/CA-TCC).
- [Bài báo CA-TCC](https://arxiv.org/abs/2208.06616).
- [DINOv3 — mã nguồn chính thức](https://github.com/facebookresearch/dinov3).
- [README gốc và BibTeX của TS-TCC / CA-TCC](docs/README_upstream.md) được giữ nguyên để tham khảo và trích dẫn.
