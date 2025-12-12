"""
"""
import logging
import json
import os
import numpy as np
import torch
import pandas as pd
import glob
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import list_data_collate

from monai.inferers import sliding_window_inference
from anatomix.segmentation.plot_utils import viz_mid_slices, viz_mid_axial_slices

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

from monai.transforms import Resize


from plot_utils import viz_mid_slices, viz_mid_axial_slices_comparison, viz_mid_slices_by_dice



from padding import divisible_pad_torch, unpad_torch


def no_sliding_window_inference(inputs, model):
    vol, pad = divisible_pad_torch(inputs, 32)
    out = unpad_torch(model(vol), pad)
    return out


def tta_no_sliding_window_inference(inputs, model, flips=None):
    """
    Run sliding_window_inference with simple flip-based TTA and average logits.
    inputs: tensor (B,C,H,W,D)
    flips: list of tuples of spatial axes to flip (use axes 2,3,4 for H,W,D).
           default = 8 combinations: no-flip + all axis flips.
    Returns averaged logits tensor.
    """
    if flips is None:
        flips = [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
    agg = None
    n = 0
    for f in flips:
        if f:
            inp = torch.flip(inputs, dims=f)
        else:
            inp = inputs
        out = no_sliding_window_inference(inp, model)
        if f:
            out = torch.flip(out, dims=f)  # inverse transform
        out = out.detach().float()
        agg = out if agg is None else agg + out
        n += 1
    return agg / float(n)



def tta_sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=0.7, flips=None):
    """
    Run sliding_window_inference with simple flip-based TTA and average logits.
    inputs: tensor (B,C,H,W,D)
    flips: list of tuples of spatial axes to flip (use axes 2,3,4 for H,W,D).
           default = 8 combinations: no-flip + all axis flips.
    Returns averaged logits tensor.
    """
    if flips is None:
        flips = [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
    agg = None
    n = 0
    for f in flips:
        if f:
            inp = torch.flip(inputs, dims=f)
        else:
            inp = inputs
        out = sliding_window_inference(inp, roi_size, sw_batch_size, model, overlap=overlap)
        if f:
            out = torch.flip(out, dims=f)  # inverse transform
        out = out.detach().float()
        agg = out if agg is None else agg + out
        n += 1
    return agg / float(n)



def validate_on_slices(dataset="/home/eperot/nnUNet_raw/baseline_mr_val/", exp_name='baseline_mr_v2', viz=False, mode='test'):
    images, segs = find_and_sort_files(dataset, mode)

    val_files = [
        {"image": img, "label": seg} for img, seg in zip(images, segs)
    ]

    # this code confirms that baseline_mr_val is NOT preproc
    # import nibabel as nib
    # for img in images:
    #     img = nib.load(img)
    #     print(img.header.get_zooms())
    #     breakpoint()

    val_transforms = Compose(
        [
            LoadImaged(keys=['image','label']),
            EnsureChannelFirstd(keys=['image','label']),
            EnsureTyped(keys=['image','label']),
            Orientationd(keys=['image','label'], axcodes='RAS'),
            # Spacingd(keys=["image", "label"], mode=('bilinear', 'nearest'), pixdim=[1.5,1.5,3]), # THIS IS BROKEN FOR SPARSE ANNOTS
            Spacingd(keys=["image"], mode='bilinear', pixdim=[1.5, 1.5, 3]), # do not resize the labels
            ScaleIntensityd(keys="image")
        ]
    )


    roi_size = (160,160,80)
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

    checkpoint_filepath = glob.glob(f'finetuning_runs/checkpoints/{exp_name}/best_dict*.pth')[-1]
    print(checkpoint_filepath)
    new_model = load_model(
        "scratch",
        14,
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

    all_dices = []

    import matplotlib.pyplot as plt

    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            sw_batch_size = 4
            # val_outputs = tta_sliding_window_inference(
            #     val_images, roi_size, sw_batch_size,
            #     new_model, overlap=0.7,
            # )

            val_outputs = tta_no_sliding_window_inference(val_images, new_model)


            # Put Back prediction and image back to original pixdim
            val_outputs = torch.nn.functional.interpolate(val_outputs, size=val_labels.shape[2:], mode='trilinear')
            val_images = torch.nn.functional.interpolate(val_images, size=val_labels.shape[2:], mode='trilinear')


            # select only correct slices
            annotated_slices = torch.unique(torch.nonzero(val_labels.squeeze())[:,2])
            subvol_val_labels = val_labels[:,:,:,:,annotated_slices]
            subvol_val_outputs = val_outputs[:,:,:,:,annotated_slices]

            dices = 1-valloss_function(subvol_val_outputs, subvol_val_labels).squeeze()
            dices = dices.cpu().numpy()


            if dices.mean() == 1:
                print(val_labels.sum().item())
                breakpoint()

            case_dices = {'case': f'case_{i}'}
            for label_name, idx in labels_list.items():
                if idx == 0:
                    continue
                dice_value = dices[idx-1]
                # if dice_value < 0.1:
                #     print(label_name, (subvol_val_labels==idx).sum())
                #     lab = (subvol_val_labels==idx).cpu().numpy().squeeze()
                #     pred =(subvol_val_outputs.argmax(dim=1)==idx).cpu().numpy().squeeze()
                #     fig, ax = plt.subplots(1)
                #     ax.imshow(lab[...,0])
                #     ax.contour(pred[...,0], levels=0, color='red')
                #     plt.show()
                #     plt.savefig('test.png')
                #     plt.close()
                #     breakpoint()
                case_dices[label_name] = dice_value
                print(f"{label_name}: Dice {dice_value:.4f}")

            all_dices += [case_dices]

            if viz:
                img = val_images[0,0].cpu().numpy()
                labels_pred = val_outputs.argmax(dim=1)[0].cpu().numpy()
                labels_gt = val_labels.cpu().numpy().squeeze()
                print("num annotated slices: ", len(annotated_slices))
                if len(annotated_slices) < 5 and len(annotated_slices) > 0:
                    viz_mid_axial_slices_comparison(img, labels_pred, labels_gt, labels_list,  filename=f'{demo_dir}/demo_output_sample{i}_image_output.png', axis=2) # take axis=2 if RAS
                else:
                    viz_mid_slices(
                        img,
                        labels_pred,
                        labels_list,
                        filename=f'{demo_dir}/demo_output_sample{i}_image_output.png')

    df = pd.DataFrame(all_dices)
    # Calculate average row
    avg_row = {'case': 'average'}
    avg_avg = 0
    for col in df.columns:
        if col != 'case':
            avg_row[col] = df[col].mean()
            avg_avg += df[col].mean() / len(df.columns)

    print("Final Average: ", avg_avg)
    # Insert average as first row
    df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    # Save to CSV
    csv_path = f'{demo_dir}/dice_scores.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved dice scores to {csv_path}")


if __name__ == '__main__':
    import fire;fire.Fire(validate_on_slices)