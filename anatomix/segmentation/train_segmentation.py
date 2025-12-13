"""
TODO:
1. handle checkpoint saving better (save last only + best) : DONE
2. accelerate transforms by running them on GPU. (Pre-Crop Randomly with worst size being calculated by rotated patch_size by 45°): KO
3. add fast plotting on mid-slices with labels for validation.: DONE


Train on Anatomix Fabien dataset
"""
import logging
import json
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import list_data_collate

from monai.inferers import sliding_window_inference
from anatomix.segmentation.plot_utils import viz_mid_slices, viz_mid_axial_slices_comparison

from tqdm import tqdm

from anatomix.segmentation.segmentation_utils import (
    load_model,
    save_ckp,
    worker_init_fn,
    get_train_transforms,
    get_val_transforms,
    get_val_from_raw_transforms,
    data_handler,
)

from monai.transforms import (
    CutMix
)
from monai.metrics import DiceMetric
torch.multiprocessing.set_sharing_strategy('file_system')


MAX_VAL_BATCHES = 14



def main(opt):
    os.makedirs(
        'finetuning_runs/checkpoints/{}'.format(opt.exp_name),
        exist_ok=True,
    )
    os.makedirs(
        'finetuning_runs/runs/{}/'.format(opt.exp_name),
        exist_ok=True,
    )

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    trimages, trsegs, vaimages, vasegs = data_handler(
        opt.dataset, opt.train_amount, opt.n_iters_per_epoch, opt.batch_size,
    )

    dataset_json = json.load(open(os.path.join(opt.dataset, 'dataset.json')))
    labels_list = dataset_json['labels']

    assert opt.n_classes == len(labels_list)

    print('Training cache: {} images {} segs'.format(len(trimages), len(trsegs)))
    print('Validation set: {} images {} segs'.format(len(vaimages), len(vasegs)))

    train_files = [
        {"image": img, "label": seg} for img, seg in zip(trimages, trsegs)
    ]
    val_files = [
        {"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)
    ]
    # define transforms for image and segmentation
    train_transforms = get_train_transforms(opt.crop_size)
    val_transforms = get_val_from_raw_transforms()

    # create a training data loader
    # transform to Dataset if debug mode
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=8
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )

    # Create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_model = load_model(
        opt.pretrained_ckpt,
        opt.n_classes,
        device,
        freeze_mode="none",
    )

    # Create Dice + CE loss function
    loss_function = monai.losses.DiceCELoss(
        softmax=True, to_onehot_y=True, batch=True, include_background=False,
    )
    # Track Dice loss for validation
    dice_metric = DiceMetric(
        include_background=False,
        reduction="none",  # or "none" for per-sample
        get_not_nans=False,
        num_classes=15  # ← Add this if it helps
    )

    # Create optimizer and scheduler

    optimizer = torch.optim.Adam(
        new_model.parameters(), opt.lr, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.n_epochs
    )

    if opt.nnunet_optimizer:
        optimizer = torch.optim.SGD(
                    new_model.parameters(),
                    lr=opt.lr,
                    momentum=0.99,
                    weight_decay=3e-5,
                    nesterov=True,
                )
        poly_lr = lambda epoch: (1 - epoch / opt.n_epochs) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)


    scaler = torch.GradScaler("cuda")

    # start a typical PyTorch training
    val_interval = opt.val_interval
    best_val_loss = 10000000000
    epoch_loss_values = list()
    pth_best_val_loss = ""
    pth_last_epoch = ""
    amp_enabled = opt.amp_enabled
    writer = SummaryWriter(
        log_dir='finetuning_runs/runs/{}/'.format(opt.exp_name),
        comment='_segmentor',
    )
    print("amp enabled: ", amp_enabled)


    # Training loop
    for epoch in range(opt.n_epochs):
        print("-" * 10)
        print("epoch {:04d}/{:04d}".format(epoch + 1, opt.n_epochs))
        new_model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader, total=len(train_loader)):
            step += 1

            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            # i had to fix it inside monai 1.5.1!!!
            if opt.cutmix:
                cm = CutMix(len(batch_data["image"]), alpha=0.5)
                inputs, labels = cm(inputs, labels)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = new_model(inputs)
                loss = loss_function(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar(
                "train_loss", loss.item(), epoch_len * epoch + step,
            )

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        scheduler.step()

        # Validation and checkpointing loop:
        if (epoch + 1) % val_interval == 0:
            new_model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_dice = 0
                valstep = 0

                for i, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    roi_size = opt.crop_size
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size,
                        new_model, overlap=0.7,
                    )

                    # handle partial annots
                    val_outputs = torch.nn.functional.interpolate(val_outputs, size=val_labels.shape[2:], mode='trilinear')
                    val_images = torch.nn.functional.interpolate(val_images, size=val_labels.shape[2:], mode='trilinear')
                    val_outputs_argmax = val_outputs.argmax(dim=1, keepdim=True)


                    annotated_slices = torch.unique(torch.nonzero(val_labels.squeeze())[:,2])
                    print('num slices: ', len(annotated_slices))
                    subvol_val_labels = val_labels[...,annotated_slices]
                    subvol_val_outputs = val_outputs_argmax[...,annotated_slices]
                    dice_metric(y_pred=subvol_val_outputs, y=subvol_val_labels)
                    dices = dice_metric.aggregate()[0].cpu().numpy().squeeze()
                    dice_metric.reset()  # Reset for next sample
                    case_dices = {label_name:dices[idx-1] for label_name, idx in labels_list.items()}

                    val_dice += dices[~np.isnan(dices)].mean()

                    valstep += 1
                    if i > MAX_VAL_BATCHES:
                        break

                    img = val_images[0, 0].cpu().numpy()
                    gt_mask = val_labels[0, 0].cpu().numpy()
                    if i == 0:
                        img = val_images[0,0].cpu().numpy()
                        labels_pred = val_outputs_argmax.cpu().numpy().squeeze()
                        labels_gt = val_labels.cpu().numpy().squeeze()
                        if len(annotated_slices) < 5:
                            viz_mid_axial_slices_comparison(img, labels_pred, labels_gt, labels_list, writer=writer, tag="Val/ground_truth", global_step=epoch + 1, axis=2, dices=case_dices)
                        else:
                            viz_mid_slices(
                                img, gt_mask, labels_list,
                                writer=writer, tag="Val/ground_truth", global_step=epoch + 1
                            )

                            # Visualize prediction
                            pred_mask = val_outputs.argmax(dim=1)[0].cpu().numpy()
                            viz_mid_slices(
                                img, pred_mask, labels_list,
                                writer=writer, tag="Val/prediction", global_step=epoch + 1
                            )


                val_dice = val_dice / valstep

                if val_dice < best_val_loss:
                    best_val_loss = val_dice
                    best_loss_epoch = epoch + 1
                    if pth_best_val_loss != "":
                        os.remove(pth_best_val_loss)
                    pth_best_val_loss = os.path.join( f"finetuning_runs/checkpoints/{opt.exp_name}", f"best_dict_epoch{epoch + 1:04d}.pth")
                    torch.save(
                        new_model.state_dict(),
                        pth_best_val_loss
                    )
                    print("saved new best loss model")


                print(
                    "current epoch: {} current mean dice: {:.4f}"
                    " best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_dice,
                        best_val_loss.item(), best_loss_epoch,
                    )
                )
                writer.add_scalar(
                    "val_loss_mean_dice_metric", val_dice, epoch + 1
                )


        if (epoch + 1) % val_interval == 0:
            checkpoint = {
                "state_dict": new_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            if pth_last_epoch:
                os.remove(pth_last_epoch)
            pth_last_epoch = 'finetuning_runs/checkpoints/{}/epoch{:04d}.pth'.format(opt.exp_name, epoch+1)
            save_ckp(
                checkpoint,
                pth_last_epoch
            )

    writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--dataset', type=str, default='./dataset/',
        help="Directory where image and label *.nii.gz files are stored.",
    )
    parser.add_argument(
        '--n_epochs', type=int, default=1000,
        help="Number of epochs. "
        "An epoch is defined as n_iters_per_epoch training batches",
    )
    parser.add_argument(
        '--n_iters_per_epoch', type=int, default=75,
        help="Number of training batches per epoch",
    )
    parser.add_argument(
        '--n_classes', type=int, default=14,
        help="Number of classes to segment. Does not include background class",
    )
    parser.add_argument(
        '--val_interval', type=int, default=50,
        help="Do a valid. and checkpointing loop every val_interval epochs",
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help="Adam step size",
    )

    parser.add_argument(
        '--crop_size', type=tuple, default=(160,160,64),
        help="Crop size to train on",
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help="Batch size to train with",
    )
    parser.add_argument(
        '--train_amount', type=int, default=14,
        help="No. of training samples to use for few-shot training",
    )
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default='/home/eperot/oneview_mr/anatomix/model-weights/anatomix.pth',
        help="Default points to model weights path. "
        "Set to 'scratch' for random initialization",
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='decoder',
        help="Prefix to attach to training logs in folder and file names",
    )
    parser.add_argument(
        '--amp_enabled', action='store_false',
        help="If set, use Mixed Precision."
    )
    parser.add_argument(
        '--cutmix', action='store_true',
        help="If set, use cutmix regularizer."
    )
    parser.add_argument(
        '--nnunet_optimizer', action='store_true',
        help="If set, use nnunet's optimizer."
    )

    args = parser.parse_args()

    main(args)
