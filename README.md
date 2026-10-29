# DINOv3 + CA-TCC for Fitness Video Classification

This project integrates **DINOv3 ViT-S/16** into CA-TCC to learn video representations for three exercise classes: **Squat**, **Push-up**, and **Lunge**. The original CNN encoder remains available for time-series experiments.

The integration has been validated with real pretrained weights and synthetic video inputs. The reports document implementation checks rather than exercise recognition benchmarks.

**Latest update — October 30, 2026:** the `dinov3_finetune` mode trains the final DINOv3 transformer block, Temporal Adapter, classification head, and CA-TCC Temporal Contrasting module using a joint loss:

```text
L_total = L_CE + 0.1 * L_temporal + 0.1 * L_SupCon
```

See the [fine-tuning guide](docs/DINOV3_FINETUNING.md) for the exact objective, implementation locations, and commands. Results: [GPU fine-tuning test](reports/finetune_gpu.json), [CPU fine-tuning test](reports/finetune_cpu.json), and [regression test](reports/regression_finetune_gpu.json).

## 1. Architecture

### Original encoder

`models/model.py::base_Model` contains three Conv1d–BatchNorm–ReLU–MaxPool blocks, with Dropout in the first block. Under the HAR configuration:

```text
Input:    (B, 9, 128)
Features: (B, 128, 18)
Logits:   (B, 6)
```

### DINOv3 encoder

The video branch retains the same `forward() -> (logits, features)` interface:

```text
RGB video (B, T, 3, H, W), default T = 18
    -> Uniform temporal sampling and resize to 224 x 224
    -> Checkpoint-specific AutoImageProcessor
    -> DINOv3 ViT-S/16, processed in small frame batches
    -> Per-frame CLS embeddings (B, T, 384)
    -> TemporalAdapter:
         transpose -> Conv1d(384, C, 1) -> ReLU -> Dropout
         -> AdaptiveAvgPool1d(L)
    -> Features (B, C, L) = (B, 128, 18)
         |-- Two views -> normalization -> bidirectional TC -> NT-Xent / SupCon
         |-- Flatten -> Linear(C * L, 3) -> classification logits (B, 3)
```

Temporal Contrasting (TC) is a training branch. Inference uses the encoder, adapter, and classification head. In `dinov3_finetune`, the classification and contrastive branches contribute to a joint objective that also updates the selected DINOv3 blocks.

`FitnessVideo_Configs.Config` inherits the existing HAR dimensions: `final_out_channels=128`, `features_len=18`, `TC.hidden_dim=100`, and `TC.timesteps=6`. The adapter reads C and L from the configuration and requires `T >= L > TC.timesteps`. With the default T=L, temporal pooling preserves the sequence length.

Existing modes freeze DINOv3 using `eval()` and `no_grad()`, including after `model.train()`. The frozen CUDA path uses FP16 autocast and returns FP32 embeddings. Fine-tuning uses FP32 gradients for selected blocks while keeping the frozen prefix in evaluation mode. Video tensors remain on the CPU and the backbone processes frames in chunks.

Preprocessing and CLS extraction follow the [Transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3). The pretrained checkpoint is [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

## 2. Implementation Changes

| File | Main components | Purpose |
|---|---|---|
| `models/dinov3.py` | `DINOv3Backbone`, `TemporalAdapter` | Cached checkpoint loading, preprocessing, frame embeddings, selective unfreezing, and temporal adaptation |
| `models/model.py` | `base_Model.__init__ / forward` | Selects the CNN or DINOv3 encoder while preserving the output interface and original CNN parameter names |
| `models/loss.py` | `DINOv3CATCCLoss` | Combines classification, bidirectional temporal, and supervised contextual losses; exposes individual terms |
| `models/TC.py` | `TC.__init__ / forward` | Retains TC with explicit LogSoftmax dimensions, sequence validation, and input-based device/dtype allocation |
| `config_files/FitnessVideo_Configs.py` | `Config` | Video settings, class mapping, trainable blocks, backbone learning rate, and loss weights |
| `dataloader/video.py` | `VideoDataset`, `load_video`, `video_views` | Video decoding, sampling, tensor/manifest inputs, and temporally consistent augmentation |
| `dataloader/dataloader.py` | `Load_Dataset`, `data_generator` | Selects the appropriate loader, detects time-series channels correctly, and retains all validation/test samples |
| `dataloader/augmentations.py` | `permutation` | Supports unequal segment lengths with current NumPy versions and preserves individual channels |
| `trainer/trainer.py` | `optimizer_groups`, `model_train`, `Trainer`, `training_checkpoint` | Separate learning rates, joint-loss training, gradient clipping, loss history, and best/last checkpoints |
| `trainer/trainer.py` | `prepare_input`, `model_evaluate`, `gen_pseudo_labels` | CPU video inputs, evaluation, and CPU storage for generated pseudo-label data |
| `main.py` | Configuration and training CLI | Backbone/mode selection, checkpoint initialization, loss weights, and complete pseudo-label generation |
| `utils.py` | `copy_Files` | Copies integration modules into experiment snapshots |
| `predict_video.py` | `load_classifier`, `main` | Loads a trained classifier and predicts exercise labels and probabilities |
| `smoke_test_dinov3_catcc.py` | Integration and regression checks | Tests real pretrained weights, gradients, CNN compatibility, CLI stages, and prediction |
| `test_dinov3_finetune.py` | Fine-tuning checks | Validates loss composition, backbone updates, frozen parameters, and checkpoint reconstruction |
| `reports/*.json` | Machine-readable test results | Records integration, fine-tuning, and regression evidence |
| `requirements-dinov3.txt`, `.gitignore`, `.gitattributes` | Dependencies and repository rules | Preserves the installed PyTorch stack and excludes data, credentials, caches, and checkpoints |
| `docs/DINOV3_FINETUNING.md` | Fine-tuning reference | Documents the joint objective and training workflow |
| `docs/README_upstream.md` | Original project documentation | Retains upstream attribution and citation information |

Important code changes include concise Vietnamese comments marked `# NEW` or `# MODIFIED`. The original attention module and NT-Xent/SupCon implementations remain available. Existing local files such as `test_dinov3.py` and `.vscode/` are preserved.

## 3. Environment Setup

Validated environment:

| Component | Version / hardware |
|---|---|
| Conda environment | `catcc-dinov3` |
| Python | 3.10.22 |
| PyTorch | 2.9.1+cu126 |
| Transformers | 4.57.6 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM |

Run in PowerShell:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available())"
python -m pip check
```

The Python executable should resolve to `miniconda3\envs\catcc-dinov3\python.exe`. If Conda is unavailable in the current terminal, invoke the environment directly:

```powershell
& "$HOME\miniconda3\envs\catcc-dinov3\python.exe" -c "import sys; print(sys.executable)"
```

Check dependencies before installing missing packages:

```powershell
python -m pip show transformers huggingface-hub einops opencv-python pandas scikit-learn openpyxl
python -m pip install -r requirements-dinov3.txt
```

The requirements file leaves PyTorch and torchvision installation to the existing CUDA environment. The validated setup already has the checkpoint and processor in the Hugging Face cache. Loading defaults to `local_files_only=True`. Other machines need authorized access to the Meta checkpoint and a prepared local cache. Authentication tokens should be managed outside the source code.

## 4. Run the Tests

These checks use synthetic inputs and the real pretrained checkpoint:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"

# Pretrained backbone and 384-dimensional image embeddings
python smoke_test_dinov3_catcc.py --device cuda:0 --stage backbone

# Full CA-TCC pipeline, including pseudo-label generation and prediction
python smoke_test_dinov3_catcc.py --device cuda:0 --cli --output reports/smoke_gpu.json

# CPU integration checks
python smoke_test_dinov3_catcc.py --device cpu --output reports/smoke_cpu.json

# Joint loss and DINOv3 block fine-tuning
python test_dinov3_finetune.py --device cuda:0 --cli --output reports/finetune_gpu.json
python test_dinov3_finetune.py --device cpu --output reports/finetune_cpu.json

# Regression checks after the fine-tuning update
python smoke_test_dinov3_catcc.py --device cuda:0 --cli --output reports/regression_finetune_gpu.json
```

CLI tests create temporary video, tensor, and checkpoint fixtures, then remove them. If an execution sandbox restricts temporary directories, run the scripts from a regular Python terminal. Metrics generated from these fixtures are implementation checks.

## 5. Video Dataset Format

Place dataset splits under `data/FitnessVideo/`, which is excluded from Git. JSON manifests allow clips to be loaded on demand:

```text
data/FitnessVideo/
    train.json
    val.json
    test.json
    clips/...
```

Each manifest contains a list of clips. Paths are resolved relative to the manifest directory:

```json
[
  {"path": "clips/squat_001.mp4", "label": 0},
  {"path": "clips/pushup_001.mp4", "label": 1},
  {"path": "clips/lunge_001.mp4", "label": 2}
]
```

Class mapping: **0 = Squat, 1 = Push-up, 2 = Lunge**. Each clip must contain at least 18 decodable frames. The loader samples 18 frames uniformly over the clip and resizes them to 224 × 224.

Alternatively, `train.pt`, `val.pt`, and `test.pt` can contain tensor dictionaries:

- `samples`: shape `(N, T, 3, H, W)`, T ≥ 18, RGB `uint8 [0,255]` or floating point `[0,1]`.
- `labels`: shape `(N,)`, dtype `torch.long`, values 0–2.

Supply images before ImageNet normalization; the checkpoint processor handles normalization. Tensor files are loaded with `weights_only=True`; legacy NumPy/object pickles require conversion to tensor dictionaries from a trusted source. If both JSON and PT exist for a split, JSON takes precedence.

Augmentation applies consistent brightness/contrast changes across each clip while preserving frame order. Contrastive training and joint fine-tuning require batch size ≥ 2. Validation and test loaders retain incomplete final batches.

## 6. Training and Inference

### Joint DINOv3 + CA-TCC fine-tuning

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode dinov3_finetune --experiment_description FitnessVideo_experiments --run_description joint_loss --device cuda:0 --batch_size 2 --frame_batch_size 1 --unfreeze_last_blocks 1 --backbone_lr 1e-5 --ce_weight 1 --temporal_weight 0.1 --context_weight 0.1 --epochs 20
```

This mode trains the final ViT block and output normalization, adapter, classifier, and TC module. Add `--init_checkpoint PATH_TO_CHECKPOINT.pt` to initialize both the model and TC from an existing compatible checkpoint. See the [fine-tuning guide](docs/DINOV3_FINETUNING.md) for checkpoint selection and loss logging.

### Self-supervised pretraining

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode self_supervised --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0 --batch_size 2 --frame_batch_size 2 --epochs 40
```

For lower memory use, set `--frame_batch_size 1`. Keep contrastive batches at least 2. Existing modes freeze DINOv3; selective unfreezing is enabled by `dinov3_finetune`.

### Semi-supervised CA-TCC workflow

Use the same experiment description, run description, and seed across stages. Prepare `train_1perc.json` or `train_1perc.pt` from the labeled training subset. Keep validation/test data separate.

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode ft_1p --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode gen_pseudo_labels --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode SupCon --data_perc 1 --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode ft_SupCon_1p --experiment_description FitnessVideo_experiments --run_description dinov3 --device cuda:0
```

`gen_pseudo_labels` follows the original 1% workflow: it reads the `ft_1p` checkpoint and writes `pseudo_train_data.pt`. It collects resized clips in RAM, which can be costly for large datasets. `SupCon --data_perc` selects the corresponding fine-tuned checkpoint; other percentages require matching pseudo-label preparation.

### Prediction

Use a checkpoint with a trained classification head. For the joint fine-tuning workflow:

```powershell
python predict_video.py --video data/FitnessVideo/clips/example.mp4 --checkpoint experiments_logs/FitnessVideo_experiments/joint_loss/dinov3_finetune_seed_0/saved_models/ckp_best.pt --device cuda:0
```

For the semi-supervised workflow:

```powershell
python predict_video.py --video data/FitnessVideo/clips/example.mp4 --checkpoint experiments_logs/FitnessVideo_experiments/dinov3/ft_SupCon_1p_seed_0/saved_models/ckp_last.pt --device cuda:0
```

Prediction restores the saved configuration and class mapping. Checkpoints saved in `self_supervised` or `SupCon` mode are rejected because their classification heads have not been trained in those stages.

`--training_mode supervised` provides classification-only training. TC integration is exercised by the self-supervised, SupCon, and joint fine-tuning modes.

### Original CNN encoder

```powershell
python main.py --selected_dataset HAR --backbone_type original --training_mode self_supervised --device cuda:0
```

This command requires HAR tensors in `data/HAR/`. The CNN consumes sensor time series; DINOv3 consumes RGB video. YOLO keypoint sequences require their own representation path. Unsupported backbone/dataset combinations produce an explicit error.

## 7. Validation Results

Initial integration tests were run on **October 17, 2026**. Fine-tuning and regression tests were run on **October 30, 2026**.

| Check | Verified behavior |
|---|---|
| Pretrained image encoder | CLS embedding `(1,384)`; consistent uint8/float preprocessing |
| Video feature extraction | `(2,18,384)`; at most 2 frames per backbone call in the original integration test |
| Temporal Adapter | FP32 features `(2,128,18)`, correct device, short-clip validation |
| Classification head | Logits `(2,3)` |
| Self-supervised trainer | Bidirectional TC + NT-Xent; adapter/TC forward, backward, and optimizer updates |
| SupCon trainer | Bidirectional TC + SupCon; finite, nonzero gradients |
| Supervised training/evaluation | Adapter/head gradients and complete prediction output |
| Frozen DINOv3 | Evaluation mode and no backbone gradients during training |
| Original HAR CNN | `(2,9,128) -> (2,128,18)` features and `(2,6)` logits; backward succeeds |
| GPU CLI workflow | Pretraining → fine-tuning → pseudo-labels → SupCon → fine-tuning → prediction |
| Decoding/checkpoint checks | Synthetic AVI manifest loading; checkpoint reconstruction; 5 pseudo-labels for 5 clips |
| Joint fine-tuning | Final DINOv3 block updates, frozen prefix preserved, adapter/head/TC gradients verified |
| Fine-tuning checkpoint | Best/last checkpoint saving and equivalent logits after reload |
| CPU | Integration and joint fine-tuning checks pass |

Reports: [initial GPU](reports/smoke_gpu.json), [initial CPU](reports/smoke_cpu.json), [fine-tuning GPU](reports/finetune_gpu.json), [fine-tuning CPU](reports/finetune_cpu.json), [regression GPU](reports/regression_finetune_gpu.json).

In the two-epoch synthetic GPU fine-tuning test, total loss changed from **1.239198 to 1.131357**. Peak PyTorch allocated memory was **544.74 MiB**, including checkpoint reconstruction. The original frozen-backbone test recorded **124.19 MiB** peak allocated and **132 MiB** peak reserved memory on a **4095.5 MiB** GPU. These figures describe the test processes, rather than total driver or system VRAM usage.

Exercise recognition accuracy/F1, full-backbone training, large datasets, YOLO/RGB fusion, multi-GPU training, and additional video codecs require separate evaluation.

## 8. Future Work

Potential extensions include subject-independent dataset splits, person cropping, temporal windows, cached CLS embeddings for frozen backbones, and manifest-based pseudo-label storage. RGB features can also be combined with YOLO Pose keypoints. Exercise classification, repetition counting, and form assessment require separate training and evaluation.

## 9. References and Attribution

- [CA-TCC source repository](https://github.com/emadeldeen24/CA-TCC)
- [CA-TCC paper](https://arxiv.org/abs/2208.06616)
- [Official DINOv3 repository](https://github.com/facebookresearch/dinov3)
- [Original README and TS-TCC / CA-TCC BibTeX entries](docs/README_upstream.md)
