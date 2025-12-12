"""
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
from anatomix.segmentation.plot_utils import viz_mid_slices

from tqdm import tqdm

from anatomix.segmentation.segmentation_utils import (
    load_model,
    find_and_sort_files,
    worker_init_fn
)

from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
)

from plot_utils import viz_mid_slices, viz_mid_axial_slices





def validate_on_slices(dataset="/home/eperot/nnUNet_raw/baseline_mr_val/", exp_name='baseline_mr', viz=False):
    images, segs = find_and_sort_files(dataset, 'test')

    val_files = [
        {"image": img, "label": seg} for img, seg in zip(images, segs)
    ]

    val_transforms = Compose(
        [
            LoadImaged(keys=['image','label']),
            EnsureChannelFirstd(keys=['image','label']),
            EnsureTyped(keys=['image','label']),
            Orientationd(keys=['image','label'], axcodes='IPL'),
            Spacingd(keys=["image", "label"], pixdim=[3,1.5,1.5]),
            ScaleIntensityd(keys="image")
        ]
    )

    crop_size = 128
    device = 'cuda:0'

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
        shuffle=True
    )

    checkpoint_filepath = f'finetuning_runs/checkpoints/{exp_name}/best_dict_epoch0444.pth'
    new_model = load_model(
        "scratch",
        15,
        device,
        freeze_mode='none'
    )
    new_model.load_state_dict(torch.load(checkpoint_filepath, weights_only=True))
    new_model.eval()
    valloss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=False, reduction="none")

    dataset_json = json.load(open(os.path.join(dataset, 'dataset.json')))
    labels_list = dataset_json['labels']

    demo_dir = f'finetuning_runs/demo_outputs/{exp_name}/'
    os.makedirs(demo_dir, exist_ok=True)

    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            roi_size = (crop_size, crop_size, crop_size)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size,
                new_model, overlap=0.7,
            )



            if viz:
                img = val_images[0,0].cpu().numpy()
                labels = val_outputs.argmax(dim=1)[0].cpu().numpy()
                viz_mid_slices(
                    img,
                    labels,
                    labels_list,
                    filename=f'{demo_dir}/demo_output_sample{i}_image_output.png')


if __name__ == '__main__':
    import fire;fire.Fire(validate_on_slices)