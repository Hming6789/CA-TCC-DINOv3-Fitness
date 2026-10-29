import argparse
import os
import sys
from importlib import import_module
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer.trainer import Trainer, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad

start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description',     default='HAR_experiments',  type=str,   help='Experiment Description')
parser.add_argument('--run_description',            default='test1',            type=str,   help='Experiment Description')
parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels, dinov3_finetune')

parser.add_argument('--selected_dataset',           default='HAR',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')

parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='cuda:0',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')
# NEW: Chọn backbone thực sự, điều chỉnh mức sử dụng VRAM và epoch từ dòng lệnh.
parser.add_argument('--backbone_type', choices=['original', 'dinov3'], default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--frame_batch_size', type=int, default=None)
parser.add_argument('--data_perc', type=int, default=1, choices=[1, 5, 10, 50, 75])
parser.add_argument('--init_checkpoint', default=None, help='Checkpoint CA-TCC để fine-tune tiếp')
parser.add_argument('--unfreeze_last_blocks', type=int, default=1)
parser.add_argument('--backbone_lr', type=float, default=1e-5)
parser.add_argument('--ce_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight', type=float, default=0.1)
parser.add_argument('--context_weight', type=float, default=0.1)
args = parser.parse_args()
print(f'Python: {sys.executable}')

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

configs = import_module(f'config_files.{data_type}_Configs').Config()
configs.backbone_type = args.backbone_type or getattr(configs, 'backbone_type', 'original')
if training_mode == 'dinov3_finetune':
    if configs.backbone_type != 'dinov3':
        parser.error('dinov3_finetune requires FitnessVideo / dinov3')
    if not 1 <= args.unfreeze_last_blocks <= 12 or not np.isfinite(args.backbone_lr) or args.backbone_lr <= 0:
        parser.error('Require 1 <= unfreeze_last_blocks <= 12 and positive finite backbone_lr')
    configs.freeze_dinov3 = False
    configs.unfreeze_last_blocks = args.unfreeze_last_blocks
    configs.backbone_lr = args.backbone_lr
    weights = (args.ce_weight, args.temporal_weight, args.context_weight)
    if not all(np.isfinite(w) and w >= 0 for w in weights) or args.ce_weight <= 0:
        parser.error('Loss weights must be finite/nonnegative; ce_weight must be > 0')
    configs.finetune_ce_weight, configs.finetune_temporal_weight, configs.finetune_context_weight = weights
if args.init_checkpoint and training_mode != 'dinov3_finetune':
    parser.error('--init_checkpoint is supported by dinov3_finetune')
if configs.backbone_type == 'dinov3' and data_type != 'FitnessVideo':
    parser.error('Use --selected_dataset FitnessVideo for RGB video input with DINOv3')
if configs.backbone_type == 'original' and data_type == 'FitnessVideo':
    parser.error('Original CNN expects time series; use HAR/Epilepsy/pFD/sleepEDF')
for argument, field in [('batch_size', 'batch_size'), ('epochs', 'num_epoch'),
                        ('frame_batch_size', 'frame_batch_size')]:
    value = getattr(args, argument)
    if value is not None:
        if value < 1:
            parser.error(f'--{argument} must be positive')
        setattr(configs, field, value)

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug(f'Backbone: {configs.backbone_type}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)
if args.init_checkpoint:
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('video_config', {}).get('class_names') != configs.class_names:
        parser.error('Checkpoint class mapping must match FitnessVideo configuration')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    temporal_contr_model.load_state_dict(checkpoint['temporal_contr_model_state_dict'], strict=True)
    logger.debug(f'Loaded model and TC from: {args.init_checkpoint}')

if "fine_tune" in training_mode or "ft_" in training_mode:
    # load saved model of this experiment
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "gen_pseudo_labels":
    ft_perc = "1p"
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    # MODIFIED: Không shuffle/drop_last để tạo nhãn cho đầy đủ dữ liệu huấn luyện.
    pseudo_dl = torch.utils.data.DataLoader(train_dl.dataset, batch_size=configs.batch_size,
                                           shuffle=False, drop_last=False, num_workers=0)
    gen_pseudo_labels(model, pseudo_dl, device, data_path)
    sys.exit(0)

if "train_linear" in training_mode or "tl" in training_mode:
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

    
if training_mode == "SupCon":
    load_from = os.path.join(       
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{args.data_perc}p_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)      
    pretrained_dict = chkpoint["model_state_dict"]      
    model.load_state_dict(pretrained_dict)     
    
    
from trainer.trainer import optimizer_groups
model_optimizer = torch.optim.Adam(optimizer_groups(model, configs), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode in ("self_supervised", "SupCon", "dinov3_finetune"):
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")
