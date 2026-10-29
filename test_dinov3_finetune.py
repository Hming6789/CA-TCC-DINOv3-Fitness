"""NEW: Kiểm tra loss và fine-tune block DINOv3 bằng video tổng hợp."""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader

from config_files.FitnessVideo_Configs import Config
from dataloader.video import VideoDataset
from models.loss import DINOv3CATCCLoss
from models.model import base_Model
from models.TC import TC
from trainer.trainer import Trainer, optimizer_groups
from predict_video import load_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cli', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('reports/finetune_smoke.json'))
    args = parser.parse_args()
    print(f'Python: {sys.executable}', flush=True)
    torch.manual_seed(19)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = Config()
    config.freeze_dinov3 = False
    config.frame_batch_size = 1
    config.num_epoch = 2
    model = base_Model(config).to(device)
    tc = TC(config, device).to(device)
    model.train()
    assert not model.encoder.backbone.layer[0].training
    assert model.encoder.backbone.layer[-1].training
    prefix = model.encoder.backbone.layer[0].mlp.up_proj.weight if hasattr(
        model.encoder.backbone.layer[0].mlp, 'up_proj') else next(model.encoder.backbone.layer[0].parameters())
    tuned = next(model.encoder.backbone.layer[-1].parameters())
    prefix_before, tuned_before = prefix.detach().clone(), tuned.detach().clone()
    assert not prefix.requires_grad and tuned.requires_grad

    # Công thức tổng và gradient riêng của CE/temporal/context phải đúng.
    criterion = DINOv3CATCCLoss(config, device)
    l1, l2 = [torch.randn(2, 3, device=device, requires_grad=True) for _ in range(2)]
    c1, c2 = [torch.randn(2, 32, device=device, requires_grad=True) for _ in range(2)]
    t1, t2 = [torch.tensor(v, device=device, requires_grad=True) for v in (0.7, 0.9)]
    loss, components = criterion(l1, l2, torch.tensor([0, 1], device=device), t1, t2, c1, c2)
    torch.testing.assert_close(loss.detach(), components['ce'] + .1 * components['temporal'] + .1 * components['contextual'])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in (l1, l2, c1, c2, t1, t2))
    print('Joint loss formula and gradients: PASS', flush=True)

    source = {'samples': torch.randint(0, 256, (2, 18, 3, 48, 48), dtype=torch.uint8),
              'labels': torch.tensor([0, 1])}
    train = DataLoader(VideoDataset(source, config, 'dinov3_finetune'), batch_size=2)
    validation = DataLoader(VideoDataset(source, config, 'supervised'), batch_size=2)
    groups = optimizer_groups(model, config)
    assert len(groups) == 2 and [g['lr'] for g in groups] == [config.lr, config.backbone_lr]
    optimizer = torch.optim.Adam(groups)
    tc_optimizer = torch.optim.Adam(tc.parameters(), lr=config.lr)
    logger = logging.getLogger('finetune_smoke')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    temp_root = Path('experiments_logs/finetune_tests').resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_name:
        temp = Path(temp_name).resolve()
        assert temp.is_relative_to(temp_root)
        Trainer(model, tc, optimizer, tc_optimizer, train, validation, validation,
                device, logger, config, str(temp), 'dinov3_finetune')
        assert torch.equal(prefix_before, prefix.detach()) and prefix.grad is None
        assert not torch.equal(tuned_before, tuned.detach())
        for module in (model.encoder.backbone.layer[-1], model.temporal_adapter, model.logits,
                       tc.Wk, tc.seq_transformer, tc.projection_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)
        history = json.loads((temp / 'loss_history.json').read_text())
        assert len(history) == 2 and all(set(('ce', 'temporal', 'contextual', 'total')) <= row.keys() for row in history)
        assert (temp / 'saved_models/ckp_best.pt').exists()
        clip = next(iter(validation))[0]
        with torch.no_grad():
            expected = model(clip)[0].cpu()
        restored, saved_config = load_classifier(temp / 'saved_models/ckp_last.pt', device)
        with torch.no_grad():
            actual = restored(clip)[0].cpu()
        torch.testing.assert_close(actual, expected)
        assert saved_config.freeze_dinov3 is False
        report = {'synthetic_only': True, 'device': str(device), 'epochs': 2,
                  'loss_formula': 'CE + 0.1 * temporal + 0.1 * SupCon',
                  'loss_history': [{k: row[k] for k in ('epoch', 'ce', 'temporal', 'contextual', 'total')}
                                   for row in history],
                  'last_block_updated': True, 'frozen_prefix_unchanged': True,
                  'adapter_head_tc_gradients': 'passed', 'checkpoint_roundtrip': 'passed',
                  'optimizer_learning_rates': [g['lr'] for g in groups]}
        if device.type == 'cuda':
            report['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        del model, tc, optimizer, tc_optimizer, restored, groups, tuned, prefix
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if args.cli:
            dataset = temp / 'FitnessVideo'
            dataset.mkdir()
            for split in ('train', 'val', 'test'):
                torch.save(source, dataset / f'{split}.pt')
            common = [sys.executable, 'main.py', '--selected_dataset', 'FitnessVideo',
                      '--training_mode', 'dinov3_finetune', '--data_path', str(temp),
                      '--logs_save_dir', str(temp / 'logs'), '--epochs', '1',
                      '--experiment_description', 'joint', '--device', str(device), '--frame_batch_size', '1']
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            for name, extra in [('initial', []), ('continued', ['--init_checkpoint', str(temp / 'saved_models/ckp_best.pt')])]:
                process = subprocess.run(common + ['--run_description', name] + extra,
                                         capture_output=True, text=True, encoding='utf-8', env=env, timeout=300)
                if process.returncode:
                    raise RuntimeError(process.stdout + process.stderr)
                path = temp / f'logs/joint/{name}/dinov3_finetune_seed_0'
                assert (path / 'loss_history.json').exists() and (path / 'saved_models/ckp_best.pt').exists()
                print(f'CLI {name}: PASS', flush=True)
            report['cli_initial_and_checkpoint_initialization'] = 'passed'
    report['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
