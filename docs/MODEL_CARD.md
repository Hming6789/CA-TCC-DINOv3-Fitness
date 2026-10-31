# Model card: VHM Motion serving profile

Status: **Preview - no verified exercise-trained model connected** (31 October 2026).

Intended use: research into classifying one exercise per clip: Squat, Push-up, Lunge. Not form assessment, repetition counting, medical advice or a safety coach.

Architecture: DINOv3 ViT-S/16 CLS features, 18-frame sequence, Conv1d temporal adapter (384 to 128), flatten and three-class linear head. CA-TCC temporal/contextual losses are offline training components. Online runtime: two ONNX CPU graphs.

Input: readable RGB video, 18 or more frames, up to 30 seconds, 32 MiB, 4K; uniformly sample 18 frames and resize to 224x224. Normalize using exported checkpoint processor metadata. Output: label, three softmax probabilities, clip metadata, latency and model version. Probabilities are not calibrated certainty or movement-quality scores.

Version/provenance: application 1.0.0; no downstream model release assigned. Training and held-out dataset provenance not verified. Existing synthetic test reports validate implementation only. No accuracy, precision, recall, macro-F1, confusion matrix or real-video latency is claimed. Toy ONNX packages used by tests are never production weights.

Limitations: three-class closed-set output cannot identify arbitrary exercises, multiple simultaneous actions or out-of-distribution inputs reliably. Camera angle, occlusion, lighting, body diversity and time sampling need empirical evaluation. No calibration, abstention threshold or domain-robustness validation is implemented.

Weights source: [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), governed by DINOv3 License and access conditions. Application image excludes weights; no new license is granted to third-party code or models. Review upstream terms and provenance before any redistribution.

Release gate for a future classifier: documented training split and subject/source-disjoint test split; checkpoint SHA-256 and training mode; label mapping; processor metadata; successful PyTorch/ONNX parity; held-out metrics with class support; CPU resource measurements; signed-off rights to data and weights.
