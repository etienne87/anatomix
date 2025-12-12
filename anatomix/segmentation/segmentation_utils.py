import torch
import os
import numpy as np
import re

from monai.networks.blocks import UnetOutBlock
from glob import glob

from monai.transforms import (
    ScaleIntensityd,
    Compose,
    LoadImaged,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandSpatialCropd,
    RandAffined,
    RandAxisFlipd,
    EnsureTyped,
    EnsureChannelFirstd,
)

from anatomix.model.network import Unet


# -----------------------------------------------------------------------------
# Loading pretrained model


def load_model(pretrained_ckpt, n_classes, device, freeze_mode):
    """
    Load and configure a U-Net model for semantic segmentation.

    This function creates a U-Net model and optionally loads pretrained weights.
    It adds a final output layer for the specified number of segmentation classes.

    Parameters
    ----------
    pretrained_ckpt : str
        Path to pretrained model checkpoint, or 'scratch' for random initialization
    n_classes : int
        Number of segmentation classes (excluding background)
    device : torch.device
        Device to load the model on ('cuda' or 'cpu')

    Returns
    -------
    new_model : torch.nn.Sequential
        Configured model with pretrained weights (if specified) and output layer
    """
    # Initialize base U-Net model
    model = Unet(3, 1, 16, 4, ngf=16).to(device)

    if pretrained_ckpt == 'scratch':
        print("Training from random initialization.")
        pass
    else:
        print("Transferring from proposed pretrained network.")
        model.load_state_dict(torch.load(pretrained_ckpt))

    if freeze_mode == "backbone":
        print("Freezing backbone weights.")
        for param in model.parameters():
            param.requires_grad = False

    if freeze_mode == "encoder":
        first_decoder_idx = model.decoder_idx[0]
        print(f"Freezing encoder weights. Everything before layer index: {first_decoder_idx}")
        for num_layer, (name, param) in enumerate(model.named_parameters()):
            if num_layer < first_decoder_idx:
                print('freeze layer: ', name)
                param.requires_grad = False

    if freeze_mode == "stem":
        first_decoder_idx = model.encoder_idx[0]
        print(f"Freezing stem weights. Everything before layer index: {first_decoder_idx}")
        for num_layer, (name, param) in enumerate(model.named_parameters()):
            if num_layer < first_decoder_idx:
                print('freeze layer: ', name)
                param.requires_grad = False


    # Add final classification layer
    fin_layer = UnetOutBlock(3, 16, n_classes + 1, False).to(device)
    new_model = torch.nn.Sequential(model, fin_layer)
    new_model.to(device)


    trainable = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in new_model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    return new_model


# -----------------------------------------------------------------------------
# Misc. utilities

def save_ckp(state, checkpoint_dir):
    """
    Save model checkpoint to disk.

    Parameters
    ----------
    state : dict
        Model state dictionary to save
    checkpoint_dir : str
        Directory path to save checkpoint file
    """
    torch.save(state, checkpoint_dir)


def worker_init_fn(worker_id):
    """
    Initialize worker for data loading.

    Sets random seed for data augmentation transforms in worker processes.

    Parameters
    ----------
    worker_id : int
        ID of the worker process
    """
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(
            worker_info.seed % (2 ** 32)
        )
    except AttributeError:
        pass


# -----------------------------------------------------------------------------
# augmentation definitions

def get_train_transforms(crop_size: tuple=(128,128,128)):
    """
    Get training data transforms based on the specified dataset.

    This function returns a composition of data transformation
    functions for training a model. These are just base augmentations.
    For actual augmentations per dataset, refer to App. B of the submission.
    This will be made dataset-specific for public release.

    Parameters
    ----------
    crop_size : int
        The size of the crop to be applied to the images.

    Returns
    -------
    train_transforms : Compose
        A composed transform object containing the specified
        transformations for the training dataset.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size,)*3
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)

    # is this pipeline optimal?
    # Initial Crop should be larger than crop size
    # Should Add Random Flips
    # Should Add Random Elastic Deformation
    # => close match to nnUNet
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=crop_size,
                random_size=False,
            ),
            RandAxisFlipd(["image", "label"], prob=0.15, lazy=True),
            RandGaussianNoised(keys=["image"], prob=0.33),
            RandBiasFieldd(
                keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)
            ),
            RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
            RandAdjustContrastd(keys=["image"], prob=0.33),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.33,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),

            RandGaussianSharpend(keys=["image"], prob=0.33),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.2, 0.2, 0.2),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=crop_size,
                padding_mode='zeros',
            ),
            ScaleIntensityd(keys="image"),
        ]
    )
    return train_transforms


def get_val_transforms():
    crop_size = 256
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
        ]
    )
    return val_transforms



def natural_sort_key(s):
    """Sort strings containing numbers in natural order"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def find_and_sort_files(basedir, mode='train'):
    # Load all training image and segmentation paths
    img_dir = 'imagesTr' if mode == 'train' else 'imagesTs'
    label_dir = 'labelsTr' if mode == 'train' else 'labelsTs'
    images = sorted(
        glob(
            os.path.join(basedir, f'./{img_dir}/*.nii.gz'),
        ),
        key=natural_sort_key
    )
    segs = sorted(
        glob(
            os.path.join(basedir, f'./{label_dir}/*.nii.gz'),
        ),
        key=natural_sort_key
    )
    return images, segs



# -----------------------------------------------------------------------------
# Dataset handling
def data_handler(
    basedir, finetuning_amount=3, iters_per_epoch=75, batch_size=3, seed=12345,
):
    """
    Handle data loading and preparation for few-shot segmentation training.

    This function loads training and validation image/segmentation pairs from the
    specified directory structure, randomly selects a subset for few-shot training,
    and repeats the training data as needed to match the desired iterations per epoch.

    Parameters
    ----------
    basedir : str
        Base directory containing imagesTr, labelsTr, imagesVal, and labelsVal folders
    finetuning_amount : int, optional
        Number of training image pairs to use for few-shot learning. Default is 3
    iters_per_epoch : int, optional
        Number of training iterations per epoch. Default is 75
    batch_size : int, optional
        Batch size for training. Default is 3
    seed : int, optional
        Random seed for reproducible data selection. Default is 12345

    Returns
    -------
    tuple
        Lists of file paths: (training images, training segmentations,
                             validation images, validation segmentations)
    """

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
    # Verify we have matching pairs of images and segmentations
    assert len(trimages) > 0
    assert len(trimages) == len(trsegs)

    # Randomly select subset of training data for few-shot learning
    trimages = np.random.RandomState(seed=seed).permutation(trimages).tolist()
    trsegs = np.random.RandomState(seed=seed).permutation(trsegs).tolist()


    # dumb check for file mismatches
    # import tqdm
    # for img, lab in tqdm.tqdm(zip(trimages, trsegs), total=len(trimages)):
    #     basename1 = os.path.basename(img).split('_0000.nii.gz')[0]
    #     basename2 = os.path.basename(lab).split('.nii.gz')[0]
    #     if basename1 != basename2:
    #         print(f"Mismatch: {basename1} vs {basename2}")
    #         continue

    # Select val from the rest

    # Select train from the beginning
    trimages = trimages
    trsegs = trsegs

    vaimages = trimages
    vasegs = trsegs


    # I don't have any validation data for now, so commenting this out
    # Calculate repeats needed to achieve desired iterations per epoch
    samples_per_epoch = iters_per_epoch * batch_size
    repeats = max(1, samples_per_epoch // finetuning_amount)

    # Repeat training data to match desired samples per epoch
    trimages = trimages * repeats
    trsegs = trsegs * repeats

    # # Load validation data paths
    # vaimages = sorted(
    #     glob(
    #         os.path.join(basedir, './imagesVal/*.nii.gz'),
    #     )
    # )
    # vasegs = sorted(
    #     glob(
    #         os.path.join(basedir, './labelsVal/*.nii.gz'),
    #     )
    # )

    return trimages, trsegs, vaimages, vasegs