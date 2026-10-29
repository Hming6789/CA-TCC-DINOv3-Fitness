# DINOv3 Fine-Tuning and Joint CA-TCC Loss

Updated: **October 30, 2026**.

The `dinov3_finetune` mode jointly trains the final DINOv3 ViT-S/16 block, Temporal Adapter, classification head, and CA-TCC Temporal Contrasting module.

## Training Objective

Implemented in `models/loss.py::DINOv3CATCCLoss.forward`:

```text
L_CE       = (CE(logits_view1, labels) + CE(logits_view2, labels)) / 2
L_temporal = (TC(view1, view2) + TC(view2, view1)) / 2
L_context  = SupCon(normalize(context_view1), normalize(context_view2), labels)
L_total    = w_ce * L_CE + w_temporal * L_temporal + w_context * L_context
```

Default weights are `w_ce=1.0`, `w_temporal=0.1`, and `w_context=0.1`. Context embeddings come from the TC projection head. TC receives normalized `(B,128,18)` features, while cross-entropy receives raw `(B,3)` logits. Both augmented views share the same labels and preserve frame order. Loss weights are configurable starting values.

Gradients flow from the loss through the classifier/TC and adapter to the selected DINOv3 blocks. The remaining backbone prefix stays frozen. This is a downstream objective for CA-TCC; the original DINOv3 pretraining system uses teacher/student networks and DINO, iBOT, KoLeo, and Gram losses. See [Meta's training implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py).

## Implementation Map

| File / component | Change |
|---|---|
| `models/loss.py::DINOv3CATCCLoss` | Joint objective, weight validation, and individual loss terms |
| `models/dinov3.py::DINOv3Backbone` | Selectively trains final blocks and output normalization; keeps the frozen prefix in evaluation mode |
| `trainer/trainer.py::optimizer_groups` | Adam learning rates: adapter/head `3e-4`, backbone `1e-5` |
| `trainer/trainer.py::model_train` | Two views → bidirectional TC → joint loss → backward → gradient clipping → optimizer |
| `trainer/trainer.py::Trainer`, `training_checkpoint` | Per-epoch loss history; best checkpoint by validation CE; final checkpoint |
| `main.py` | Fine-tuning CLI and initialization of both model and TC from a checkpoint |
| `config_files/FitnessVideo_Configs.py` | Trainable blocks, learning rates, loss weights, and gradient clipping |
| `dataloader/video.py`, `dataloader/dataloader.py` | Two-view augmentation and batch size ≥ 2 |
| `test_dinov3_finetune.py` | Loss composition, branch gradients, parameter updates, checkpoint reconstruction, and CLI checks |

## Run the Checks

Activate the prepared Conda environment:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
python test_dinov3_finetune.py --device cuda:0 --cli --output reports/finetune_gpu.json
python test_dinov3_finetune.py --device cpu --output reports/finetune_cpu.json
python smoke_test_dinov3_catcc.py --device cuda:0 --cli --output reports/regression_finetune_gpu.json
```

Tests use synthetic video inputs. The two-epoch GPU run recorded total loss `1.239198 → 1.131357`, updated the final block, preserved the frozen prefix, and verified gradients in the adapter, classifier, and TC. Reloading the checkpoint reproduced equivalent logits. Peak allocated GPU memory was `544.74 MiB`, including the reconstructed model. CLI checks cover initialization from pretrained weights and from a CA-TCC checkpoint. These are implementation measurements.

## Train on FitnessVideo Splits

Prepare `data/FitnessVideo/train.json`, `val.json`, and `test.json`, or the tensor dictionary `.pt` equivalents described in the main README.

```powershell
python main.py --selected_dataset FitnessVideo --backbone_type dinov3 --training_mode dinov3_finetune --experiment_description FitnessVideo_experiments --run_description joint_loss --device cuda:0 --batch_size 2 --frame_batch_size 1 --unfreeze_last_blocks 1 --backbone_lr 1e-5 --ce_weight 1 --temporal_weight 0.1 --context_weight 0.1 --epochs 20
```

To initialize from an existing CA-TCC checkpoint, add `--init_checkpoint PATH_TO_CHECKPOINT.pt`. Both model and TC states are loaded strictly; the class mapping must match `Squat / Push-up / Lunge`. This starts a new fine-tuning run with a fresh optimizer and epoch count. The mode uses training labels directly and does not require pseudo-label generation.

Outputs are written to `experiments_logs/FitnessVideo_experiments/joint_loss/dinov3_finetune_seed_0/`:

- `loss_history.json`: individual training loss terms and validation CE/accuracy.
- `saved_models/ckp_best.pt`: checkpoint with the lowest validation CE.
- `saved_models/ckp_last.pt`: checkpoint from the final epoch.

Training uses joint loss on augmented views; validation uses CE on the original clips. These values measure different objectives. Checkpoint selection uses validation CE. Final evaluation in `main.py` uses the last checkpoint; use the best checkpoint explicitly for prediction when desired.

```powershell
python predict_video.py --video data/FitnessVideo/clips/example.mp4 --checkpoint experiments_logs/FitnessVideo_experiments/joint_loss/dinov3_finetune_seed_0/saved_models/ckp_best.pt --device cuda:0
```

Data and checkpoints are excluded by Git rules. Source code, documentation, and JSON test reports are tracked.
