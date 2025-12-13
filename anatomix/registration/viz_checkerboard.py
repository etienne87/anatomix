import numpy as np
import matplotlib.pyplot as plt


def create_checkerboard(image1, image2, tile_size=20):
    """
    Create a checkerboard pattern alternating between two images.

    Args:
        image1: numpy array (H, W)
        image2: numpy array (H, W) - must be same shape as image1
        tile_size: size of each checker square in pixels

    Returns:
        Checkerboard composite image
    """
    H, W = image1.shape

    # Create a checkerboard mask
    y_tiles = (np.arange(H) // tile_size) % 2
    x_tiles = (np.arange(W) // tile_size) % 2

    # XOR pattern: alternates between 0 and 1
    mask = (y_tiles[:, None] + x_tiles[None, :]) % 2

    # Combine images using the mask
    checkerboard = np.where(mask, image1, image2)

    return checkerboard


def viz_mid_slices(vol1, vol2, filename=None, figsize=None):
    fig, ax = plt.subplots(3, 2, figsize=figsize)  # Changed to 4 columns
    for i in range(3):
        slice1 = np.take(vol1, vol1.shape[i] // 2, axis=i)
        slice2 = np.take(vol2, vol2.shape[i] // 2, axis=i)

        ax[i][0].imshow(slice1, cmap="gray")
        ax[i][0].set_title("Fixed" if i == 0 else "")
        ax[i][1].imshow(slice2, cmap="gray")
        ax[i][1].set_title("Moving" if i == 0 else "")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def minmax(arr, minclip=None, maxclip=None):
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def viz_mid_slices_checkerboard(vol1, vol2, tile_size=32, do_min_max=True, filename=None):
    fig, ax = plt.subplots(3, 3, dpi=300, figsize=(12, 9))  # Changed to 4 columns
    for i in range(3):
        slice1 = np.take(vol1, vol1.shape[i] // 2, axis=i)
        slice2 = np.take(vol2, vol2.shape[i] // 2, axis=i)

        if do_min_max:
            slice1 = minmax(slice1)
            slice2 = minmax(slice2)

        # Create checkerboard between slice1 (fixed) and slice3 (warped)
        checkerboard_orig = create_checkerboard(slice1, slice2, tile_size=tile_size)

        ax[i][0].imshow(slice1, cmap="gray")
        ax[i][0].set_title("Fixed" if i == 0 else "")
        ax[i][1].imshow(slice2, cmap="gray")
        ax[i][1].set_title("Moving" if i == 0 else "")
        ax[i][2].imshow(checkerboard_orig, cmap="gray")
        ax[i][2].set_title("Checkerboard-fixed-moving" if i == 0 else "")

        # Remove axes for cleaner visualization
        for j in range(3):
            ax[i][j].axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def viz_mid_slices_checkerboard_compare(vol1, vol1_reg_v1, vol1_reg_v2, tile_size=32, do_min_max=True, names=['v1','v2']):
    fig, ax = plt.subplots(3, 2, dpi=300, figsize=(12, 9))  # Changed to 4 columns
    for i in range(3):
        slice1 = np.take(vol1, vol1.shape[i] // 2, axis=i)
        slice2 = np.take(vol1_reg_v1, vol1_reg_v1.shape[i] // 2, axis=i)
        slice3 = np.take(vol1_reg_v2, vol1_reg_v2.shape[i] // 2, axis=i)

        if do_min_max:
            slice1 = minmax(slice1)
            slice2 = minmax(slice2)
            slice3 = minmax(slice3)

        # Create checkerboard between slice1 (fixed) and slice3 (warped)
        checkerboard_v1 = create_checkerboard(slice1, slice2, tile_size=tile_size)
        checkerboard_v2 = create_checkerboard(slice1, slice3, tile_size=tile_size)

        ax[i][0].imshow(checkerboard_v1, cmap="gray")
        ax[i][0].set_title(names[0] if i == 0 else "")
        ax[i][1].imshow(checkerboard_v2, cmap="gray")
        ax[i][1].set_title(names[1] if i == 0 else "")

        # Remove axes for cleaner visualization
        for j in range(2):
            ax[i][j].axis("off")

    plt.tight_layout()
    plt.show()
