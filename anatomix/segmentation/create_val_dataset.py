"""
Load relevant dataset in amos_pp
"""
import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from deep_learning.load_and_export import loaders
import nibabel as nib

import monai
from monai.transforms import (
    Compose,
    LoadImage,
    Orientation,
    Spacing,
    EnsureChannelFirst,
    EnsureType,

    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
)



val_transforms_ipl = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes='IPL'),
    ]
)


val_transforms_ipl_dict = Compose(
    [
        LoadImaged(keys=['image','label']),
        EnsureChannelFirstd(keys=['image','label']),
        EnsureTyped(keys=['image','label']),
        Orientationd(keys=['image','label'],axcodes='IPL'),
    ]
)

from plot_utils import viz_mid_axial_slices


def load_2d_seglists_cases(out_dir):
    amos_test_image_path = os.path.join("/net/frbucawnas02/export/hadokenalgo/datasets/Abdomen/amos22/imagesTs/")
    tsimages = sorted(glob.glob(os.path.join(amos_test_image_path, '*.nii.gz')))
    ts_images_filenames_by_case_num = {}
    for tsimg in tsimages:
        case_num = int(os.path.basename(tsimg).split('_')[1].split('.nii.gz')[0])
        ts_images_filenames_by_case_num[case_num] = tsimg


    dir_path = os.path.join("/net/frbucawnas02/export/hadokenalgo/projects/oneview_for_mr/gt/gt_2D_test/")
    seglists = glob.glob(dir_path + '*.seglist')

    baseline_mr_json = os.path.join("/home/eperot/nnUNet_raw/Dataset908_baselineMR/dataset.json")
    mr = json.load(open(baseline_mr_json, 'r'))
    labels = mr['labels']

    ts_images_by_case_num = {}

    cases_masks = {}
    for seglist in tqdm(seglists, total=len(seglists)):
        case_num = int(os.path.basename(seglist).split('_')[1].split('.nii.gz')[0])
        label = os.path.basename(seglist).split('_')[-1].split('.seglist')[0]
        if label in ['trabecularBone', 'ascite']:
            continue

        if case_num not in ts_images_by_case_num:
            ts_images_by_case_num[case_num] = val_transforms_ipl(ts_images_filenames_by_case_num[case_num])

        vol = ts_images_by_case_num[case_num]
        mask = loaders.seglist2mask(seglist, vol.shape[1:])


        label_num = labels.get(label, 0) # by default put to background

        img = cases_masks.get(case_num, np.zeros(mask.shape, dtype=np.int32))
        img[mask] = label_num
        cases_masks[case_num] = img

    # verify visually
    # for case_num, label in tqdm(cases_masks.items(), total=len(cases_masks)):
    #     vol = ts_images_by_case_num[case_num].cpu().numpy().squeeze()
    #     viz_mid_axial_slices(vol, label, labels, f"viz/test#{case_num}.png")



    # dump niftis
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labelsTs'), exist_ok=True)

    for case_num, label in tqdm(cases_masks.items(), total=len(cases_masks)):
        vol = ts_images_by_case_num[case_num]

        affine = vol.meta['affine'].numpy()

        img_path = os.path.join(out_dir, 'imagesTs', f'amos_{case_num:04d}.nii.gz')
        label_path = os.path.join(out_dir, 'labelsTs', f'amos_{case_num:04d}.nii.gz')

        img_nifti = nib.Nifti1Image(vol.cpu().numpy().squeeze(), affine)
        label_nifti = nib.Nifti1Image(label, affine)
        nib.save(img_nifti, img_path)
        nib.save(label_nifti, label_path)

        # test load
        # input_filenames = {'image': img_path, 'label': label_path}
        # reloaded = val_transforms_ipl_dict(input_filenames)
        # viz_mid_axial_slices(reloaded['image'].cpu().numpy().squeeze(), reloaded['label'].cpu().numpy().squeeze(), labels, f"viz/test#{case_num}.png")








if __name__ == '__main__':
    import fire;fire.Fire(load_2d_seglists_cases)