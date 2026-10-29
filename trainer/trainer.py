import os
import json
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, SupConLoss, DINOv3CATCCLoss


def optimizer_groups(model, config):
    # NEW: Backbone pretrained dùng LR nhỏ; adapter/head giữ LR của CA-TCC.
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("encoder.backbone.") else heads).append(parameter)
    groups = [{"params": heads, "lr": config.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": config.backbone_lr})
    return groups


def training_checkpoint(model, temporal_contr_model, config, training_mode):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "temporal_contr_model_state_dict": temporal_contr_model.state_dict(),
                  "training_mode": training_mode}
    if getattr(model, "backbone_type", "original") == "dinov3":
        fields = ("dinov3_model_id", "freeze_dinov3", "frame_batch_size", "num_frames",
                  "frame_size", "features_len", "final_out_channels", "num_classes",
                  "class_names", "dropout", "unfreeze_last_blocks", "backbone_lr",
                  "finetune_ce_weight", "finetune_temporal_weight", "finetune_context_weight")
        checkpoint["video_config"] = {key: getattr(config, key) for key in fields}
    return checkpoint


def prepare_input(model, data, device):
    # NEW: Video ở CPU; backbone chỉ đưa từng nhóm frame lên GPU, tránh giữ cả hai view.
    if getattr(model, "backbone_type", "original") == "dinov3":
        return data
    return data.float().to(device)


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    history, best_valid_loss = [], float("inf")
    save_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)
        if training_mode == "dinov3_finetune":
            row = {"epoch": epoch, **model.last_loss_components,
                   "validation_ce": float(valid_loss), "validation_accuracy": float(valid_acc)}
            history.append(row)
            logger.debug(f'Fine-tune loss components: {row}')
            with open(os.path.join(experiment_log_dir, "loss_history.json"), "w", encoding="utf-8") as stream:
                json.dump(history, stream, indent=2, allow_nan=False)
            if float(valid_loss) < best_valid_loss:
                best_valid_loss = float(valid_loss)
                torch.save(training_checkpoint(model, temporal_contr_model, config, training_mode),
                           os.path.join(save_dir, "ckp_best.pt"))

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = training_checkpoint(model, temporal_contr_model, config, training_mode)
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode):
    total_loss = []
    total_acc = []
    component_history = []
    joint_loss = DINOv3CATCCLoss(config, device) if training_mode == "dinov3_finetune" else None
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        labels = labels.long().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode in ("self_supervised", "SupCon", "dinov3_finetune"):
            aug1, aug2 = prepare_input(model, aug1, device), prepare_input(model, aug2, device)
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "dinov3_finetune":
            loss, components = joint_loss(predictions1, predictions2, labels,
                                          temp_cont_loss1, temp_cont_loss2,
                                          temp_cont_feat1, temp_cont_feat2)
            component_history.append({key: float(value) for key, value in components.items()})
            predictions = (predictions1 + predictions2) / 2
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
        elif training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, features1.shape[0], config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(prepare_input(model, data, device))
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        if training_mode == "dinov3_finetune":
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(temporal_contr_model.parameters()),
                                           config.grad_clip_norm, error_if_nonfinite=True)
        model_optimizer.step()
        temp_cont_optimizer.step()

    if not total_loss:
        raise ValueError("Training loader produced no batches")
    if component_history:
        model.last_loss_components = {key: sum(row[key] for row in component_history) / len(component_history)
                                      for key in component_history[0]}
    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = prepare_input(model, data, device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs




def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = prepare_input(model, data, device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data.cpu())

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
