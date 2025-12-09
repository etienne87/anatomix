import os
import re
from glob import glob

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
)

from plot_utils import viz_mid_slices


def find_and_sort_files(basedir):
    def natural_sort_key(s):
        """Sort strings containing numbers in natural order"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    # Load all training image and segmentation paths
    trimages = sorted(
        glob(
            os.path.join(basedir, './imagesTr/*.nii.gz'),
        ),
        key=natural_sort_key
    )
    trsegs = sorted(
        glob(
            os.path.join(basedir, './labelsTr/*.nii.gz'),
        ),
        key=natural_sort_key
    )
    return trimages, trsegs



def preproc_amos22(dataset, res=[3,1.5,1.5], out_dir=''):

    images, segs = find_and_sort_files(dataset)

    val_files = [
        {"image": img, "label": seg} for img, seg in zip(images, segs)
    ]
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes='IPL'),
            Spacingd(keys=["image", "label"], pixdim=res),
        ]
    )

    dataset = monai.data.Dataset(data=val_files, transform=transforms)

    for data in dataset:
        img = data['image'].cpu().numpy().squeeze()
        lab = data['label'].cpu().numpy().squeeze()


        viz_mid_slices(img, lab, filename="test.png")
        breakpoint()



if __name__ == '__main__':
    import fire;fire.Fire(preproc_amos22)