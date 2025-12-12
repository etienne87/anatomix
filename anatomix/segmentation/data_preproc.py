import os
import re
import tqdm
import json
from glob import glob

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    SaveImaged
)

from plot_utils import viz_mid_slices, viz_mid_axial_slices
from segmentation_utils import find_and_sort_files

def save_json(dataset_path, labels, num):
    data_dict = {
        'channel_names':{
            '0': 'NoNorm'
        },
        'labels': labels,
        "numTraining": num,
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(dataset_path, 'dataset.json'), 'w') as f:
        json.dump(data_dict, f)



def preproc(dataset, out_dir, res=[3,1.5,1.5]):

    images, segs = find_and_sort_files(dataset, 'test')

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

    os.makedirs(out_dir, exist_ok=True)

    dataset_json = json.load(open(os.path.join(dataset, 'dataset.json')))
    labels_list = dataset_json['labels']
    # in amos i had this issue that organ was the value?
    labels_list = {v: int(k) for k, v in labels_list.items()} if list(labels_list.keys())[0].isdigit() else labels_list
    save_json(out_dir, labels_list, len(images))

    save_transform = Compose(
        [
            SaveImaged(
                keys=["image"],
                output_dir=os.path.join(out_dir, 'imagesTs'),
                output_postfix="",
                resample=False,
                separate_folder=False,
            ),
            SaveImaged(
                keys=["label"],
                output_dir=os.path.join(out_dir, 'labelsTs'),
                output_postfix="",
                resample=False,
                separate_folder=False,
            ),
        ]
    )

    dataset = monai.data.Dataset(data=val_files, transform=transforms)
    #dataloader = monai.data.DataLoader(dataset, batch_size=1, num_workers=8)

    for idx, data in enumerate(tqdm.tqdm(dataset, total=len(dataset))):
        save_transform(data)

        if idx==0:
            img = data['image'].cpu().numpy().squeeze()
            lab = data['label'].cpu().numpy().squeeze()
            viz_mid_slices(img, lab, filename="test.png")


def viz(dataset, mode='test'):
    images, segs = find_and_sort_files(dataset, mode)

    val_files = [
        {"image": img, "label": seg} for img, seg in zip(images, segs)
    ]
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            # Orientationd(keys=["image", "label"], axcodes='IPL'),
            # Spacingd(keys=["image", "label"], pixdim=[1,1,1]),
        ]
    )

    dataset_json = json.load(open(os.path.join(dataset, 'dataset.json')))
    labels = dataset_json['labels']

    dataset = monai.data.Dataset(data=val_files, transform=transforms)


    os.makedirs('viz', exist_ok=True)
    for idx, data in enumerate(tqdm.tqdm(dataset, total=len(dataset))):
        img = data['image'].cpu().numpy().squeeze()
        lab = data['label'].cpu().numpy().squeeze()
        # viz_mid_slices(img, lab, labels, filename=f"viz/test#{idx}.png")
        viz_mid_axial_slices(img, lab, labels, filename=f"viz/test#{idx}.png")


if __name__ == '__main__':
    import fire;fire.Fire()