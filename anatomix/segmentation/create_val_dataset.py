"""
Load relevant dataset in amos_pp
"""
import os
import glob
import json
import numpy as np
import common_io as cio
import nibabel as nib
import matplotlib.pyplot as plt


import monai
from monai.transforms import (
    Compose,
    LoadImage,
    Orientation,
    Spacing,
    EnsureChannelFirst,
    EnsureType,
    SaveImage
)



val_transforms_ipl_31515 = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes='IPL'),
        Spacing(pixdim=[3,1.5,1.5]),
    ]
)

def load_2d_seglists_cases():
    amos_test_image_path = os.path.join("/net/frbucawnas02/export/hadokenalgo/datasets/Abdomen/amos22/imagesTs/")
    tsimages = sorted(glob.glob(os.path.join(amos_test_image_path, '*.nii.gz')))
    ts_images_by_case_nums = {}
    for tsimg in tsimages:
        case_num = int(os.path.basename(tsimg).split('_')[1].split('.nii.gz')[0])
        ts_images_by_case_nums[case_num] = tsimg

    dir_path = os.path.join("/net/frbucawnas02/export/hadokenalgo/projects/oneview_for_mr/gt/gt_2D_test/")
    seglists = glob.glob(dir_path + '*.seglist')

    baseline_mr_json = os.path.join("/home/eperot/nnUNet_raw/Dataset908_baselineMR/dataset.json")
    mr = json.load(open(baseline_mr_json, 'r'))
    labels = mr['labels']

    # case_slices = {}

    cases_masks = {}
    for seglist in seglists:
        case_num = int(os.path.basename(seglist).split('_')[1].split('.nii.gz')[0])
        label = os.path.basename(seglist).split('_')[-1].split('.seglist')[0]
        if label in ['trabecularBone', 'ascite']:
            continue
        mask = cio.load_seg(seglist)>0

        label_num = labels.get(label, 0)

        img = cases_masks.get(case_num, np.zeros(mask.shape, dtype=np.int32))
        img[mask] = label_num
        cases_masks[case_num] = img


        # slice_num = np.stack(np.where(img>0), axis=-1)
        # slices = np.unique(slice_num[:,0])
        # case_slices[case_num] = slices

        # viz
        # mip = img.max(axis=0)
        # plt.imsave(f"viz/test#{case_num}.png", mip, cmap='tab20')

    for case_num, mask in cases_masks.items():

        img_nifti_path = ts_images_by_case_nums[case_num]

        img = val_transforms_ipl_31515(img_nifti_path)


        print(img.shape, mask.shape)

        if img.shape != mask.shape:
            print(f'something is wrong with case {case_num}')
            continue









if __name__ == '__main__':
    import fire;fire.Fire(load_2d_seglists_cases)