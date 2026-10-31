# Audit AI/MLOps - 31/10/2026

## Phạm vi và bằng chứng

Đã đọc các entry point `main.py`, `predict_video.py`, `webapp/app.py`; model, loss, TC, trainer, video loader, cấu hình FitnessVideo, exporter, bộ kiểm thử web, frontend và tài liệu hiện có. Các notebook/dataset tools là luồng nghiên cứu dữ liệu riêng, không được đưa vào image. Không có file checkpoint trong `experiments_logs` tại thời điểm audit và không có gói `model_artifacts` phục vụ. Không suy diễn rằng máy hoàn toàn không có weights ở nơi khác: backbone có thể nằm trong Hugging Face cache.

Working tree đã có sửa đổi README/.gitignore và các file nghiên cứu chưa tracked trước tác vụ; không xóa hoặc tự đưa chúng lên remote. Origin thực tế: https://github.com/Hming6789/CA-TCC-DINOv3-Fitness.git.

## Bảng đánh giá trước thay đổi

| Component | Current Implementation | Issue | Proposed Solution | Priority |
|---|---|---|---|---|
| Checkpoint | Không tìm thấy classifier/gói ONNX phục vụ | Không thể chứng minh phân loại video gym | Preview Mode, chặn prediction, không công bố accuracy | Critical |
| Backbone | DINOv3 ViT-S/16, CLS 384 chiều | Weights gated; chưa có provenance downstream | Tách weights khỏi image, mount read-only | High |
| CA-TCC | TC hai chiều, NT-Xent, SupCon và joint fine-tuning | Dễ nhầm TC là module online | Tách sơ đồ offline/online | High |
| API | FastAPI + ONNX CPU, health/status | Chưa có readiness HTTP riêng | Thêm readyz 200/503 | High |
| Docker | Chưa có Dockerfile | Chưa có build/run evidence | Image CPU non-root, context allowlist | High |
| CI/CD | Chưa có workflow | Không có automated gates | Tests, build, smoke, publish sau gates | High |
| Metrics | model_evaluate lấy mean theo batch | Batch cuối nhỏ gây lệch accuracy/loss tổng | Chưa dùng số này làm benchmark; đánh giá theo từng mẫu trước nghiên cứu | Medium |
| Export | Exporter whitelist training modes hẹp hơn CLI | Một số ft_SupCon/train_linear_* bị từ chối | Ghi giới hạn; dùng dinov3_finetune có metadata hợp lệ | Medium |
| Model manifest | Checksums, schema input, purpose | purpose tự khai không chứng minh chất lượng huấn luyện | Yêu cầu provenance/test set độc lập trước phát hành model | High |
| Tests | Toy ONNX graphs và báo cáo synthetic cũ | Functional pass không phải accuracy | Tách functional và scientific validation | High |

## Kiến trúc thực tế

`main.py` điều phối huấn luyện; `predict_video.py` chạy inference PyTorch; `webapp.app:app` là entry point deployment. Web tĩnh HTML/CSS/JS được FastAPI phục vụ cùng origin. API POST `/api/analyze` nhận multipart field `file`; GET `/api/status` mô tả trạng thái.

```mermaid
flowchart LR
  V[Video upload] --> G[Validation: 32 MiB / 30 s / 4K]
  G --> S[18 frame theo thứ tự thời gian]
  S --> P[RGB 224x224 / rescale / normalize]
  P --> D[DINOv3 encoder ONNX CPU]
  D --> E[CLS tensor 1x18x384]
  E --> A[Conv1d 384 to 128 / ReLU / pool]
  A --> H[Flatten 2304 / Linear 3]
  H --> R[Softmax và JSON]
```

Video ít hơn 18 frames bị từ chối. Sampling là round(linspace(0, N-1, 18)), giữ thời gian; OpenCV BGR sang RGB rồi resize bilinear. Training loader trả `(T,3,224,224)` float [0,1]; processor checkpoint xử lý normalization. Web lấy RGB uint8 `(18,224,224,3)`, chuyển từng frame sang float32 `(1,3,224,224)`, chia 255 rồi `(x-mean)/std` từ manifest. Exporter kiểm tra processor không crop/pad, rescale 1/255, kích thước 224. Đối chiếu sampling đã có test thực tế; numerical parity của model export chỉ được xác minh khi chạy exporter với checkpoint hợp lệ.

Encoder cho `(1,384)`, stack thành `(1,18,384)`. Adapter chuyển `(B,T,384)` thành `(B,128,18)` bằng Conv1d kernel=1, ReLU, Dropout, AdaptiveAvgPool1d. Khi T=L=18, pooling giữ chiều dài; adapter không phải temporal Transformer. Flatten 2304 rồi Linear 3. Nhãn cấu hình: 0 Squat, 1 Push-up, 2 Lunge; manifest mang thứ tự nhãn của checkpoint. API trả prediction, softmax, metadata clip và thời gian inference. Softmax không phải điểm chất lượng động tác hay confidence đã calibration.

Nhánh offline có Seq_Transformer trong TC (hidden=100, dự đoán 6 bước), TC hai hướng và context projection. `self_supervised` dùng NT-Xent; `SupCon` dùng supervised contrastive; `dinov3_finetune` dùng CE + 0.1 temporal + 0.1 contextual. Fine-tuning mở block cuối và norm, backbone prefix đóng băng. Các thành phần CA-TCC hiện diện thực sự, nhưng chưa đủ bằng chứng tái lập kết quả khoa học của bài báo. TC không nằm trong graph inference.

## Phụ thuộc và giới hạn

Serving dùng FastAPI 0.162.0, Starlette 0.50.0, Uvicorn 0.34.2, python-multipart 0.0.32, NumPy 2.2.6, ONNX Runtime 1.22.1, OpenCV headless 4.11.0.86. PyTorch/Transformers chỉ dùng training/export. Test web dùng unittest, httpx 0.28.1, ONNX 1.18.0; parity training cần PyTorch và được skip rõ trên runner nhẹ.

Training dependency ranges chưa phải lock hoàn chỉnh; training reproducibility cần lưu environment, seed, split manifest, hash checkpoint và versions. Không tự đổi thuật toán/nhãn/weights trong tác vụ MLOps. Không thực hiện accuracy/F1/confusion matrix vì thiếu classifier và held-out RGB video được xác minh. Dữ liệu pose `(N,52,64)` không được coi là RGB test set.

## Quyết định đóng gói weights

Chọn image chỉ chứa ứng dụng và CPU runtime; gói gồm encoder.onnx, head.onnx, model.json được mount vào `/models:ro`. Không tải weights từ mạng khi request tới và không đóng gói cache/token. Ưu điểm: public image không tái phân phối model, cập nhật model độc lập và image nhỏ hơn. Nhược điểm: máy triển khai phải cấp bundle tương thích trước khi readiness thành công. Bundle có SHA-256 nhưng vẫn cần nguồn tin cậy và provenance huấn luyện.

Đóng weights vào image thuận tiện tự khởi chạy, nhưng tăng dung lượng và gắn version model vào build; chưa chọn vì thiếu classifier và chưa xác minh quyền phân phối. Tải từ registry có thể dùng helper hiện có với HTTPS và checksum, nhưng cần đảm bảo lưu trữ/cache, secret và tính sẵn sàng. DINOv3 nguồn chính thức gắn `dinov3-license` và yêu cầu quyền truy cập; không mặc định weights có giấy phép MIT. Kích thước encoder/export hiện chưa đo vì không có artifact phục vụ; không điền số ước lượng vào kết quả thực nghiệm.

Nguồn: [model card chính thức](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [Docker GitHub Actions](https://docs.docker.com/build/ci/github-actions/).
