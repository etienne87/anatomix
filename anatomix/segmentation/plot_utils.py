import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import io
from PIL import Image



# Predefined color map for first 15 labels
BASE_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000'
]

def create_colors(n_labels):
    """Create color palette, extending if needed using a colormap."""
    if n_labels <= len(BASE_COLORS):
        return BASE_COLORS[:n_labels]

    # Use predefined colors first, then generate additional ones
    colors = BASE_COLORS.copy()
    n_additional = n_labels - len(BASE_COLORS)

    # Generate additional colors from a colormap
    cmap = plt.cm.get_cmap('tab20')
    for i in range(n_additional):
        color = mcolors.rgb2hex(cmap((i % 20) / 20))
        colors.append(color)

    return colors

# Create colors for all possible labels upfront (adjust max as needed)
LABEL_COLORS = create_colors(50)  # Support up to 50 labels


def add_legends(fig, label_dict):
    legend_elements = [Patch(facecolor=LABEL_COLORS[label_val % len(LABEL_COLORS)],
                            label=f'{label_name}')
                      for label_name, label_val in label_dict.items()]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.98, 0.5), frameon=True)


def add_legends_v2(fig, label_dict, dice_scores=None, dice_threshold=0.7):
    """
    Add legend to figure with optional dice scores.

    Args:
        fig: Matplotlib figure
        label_dict: Dictionary mapping label names to label values
        dice_scores: Optional dictionary mapping label names/values to dice scores
        dice_threshold: Threshold below which to highlight labels (default 0.7)
    """
    legend_elements = []

    for label_name, label_val in label_dict.items():
        if dice_scores is not None:
            dice = dice_scores.get(label_name, dice_scores.get(label_val, None))
            if dice is not None:
                dice_str = f" (Dice: {dice:.3f})"
                marker = " ⚠️" if dice < dice_threshold else ""
            else:
                dice_str = ""
                marker = ""
        else:
            dice_str = ""
            marker = ""

        legend_elements.append(
            Patch(facecolor=LABEL_COLORS[label_val % len(LABEL_COLORS)],
                  label=f'{label_name}{dice_str}{marker}')
        )

    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.98, 0.5), frameon=True)

def ax_viz_mid_slice(ax, slice_vol, slice_mask, label_dict):
    ax.imshow(slice_vol, cmap="gray")
    for _, label_val in label_dict.items():
        # Create binary mask for this label
        label_mask = (slice_mask == label_val).astype(float)
        if np.any(label_mask):  # Only draw if label exists in this slice
            color = LABEL_COLORS[label_val % len(LABEL_COLORS)]
            ax.contour(label_mask, levels=[0.5], colors=color, linewidths=0.5)


def viz_mid_axial_slices(vol, label, labels, filename=None, axis=0):
    slice_num = np.stack(np.where(label>0), axis=-1)
    slices = np.unique(slice_num[:,axis])
    fig, ax = plt.subplots(len(slices),1, figsize=(8*len(slices), 10))
    for j in range(len(slices)):
        vol_slice = np.take(vol, slices[j], axis=axis)
        pred_slice = np.take(label, slices[j], axis=axis)
        ax_viz_mid_slice(ax[j], vol_slice, pred_slice, labels)
    add_legends(fig, labels, dices)
    plt.savefig(filename)



def viz_mid_slices(vol, mask, labels=None, filename=None, writer=None, tag="", global_step=0):
    assert vol.shape == mask.shape

    # Handle labels parameter
    if labels is None:
        # Get unique labels from mask
        label_values = np.unique(mask)
        label_values = label_values[label_values != 0]  # Exclude background (0)
        label_dict = {f'Label {int(val)}': int(val) for val in label_values}
    elif isinstance(labels, dict):
        # Filter out background if present
        label_dict = {name: val for name, val in labels.items() if val != 0}
    else:
        # Assume it's a list/array of label values
        label_values = [val for val in labels if val != 0]
        label_dict = {f'Label {int(val)}': int(val) for val in label_values}

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    for i in range(3):
        slice_vol = np.take(vol, vol.shape[i] // 2, axis=i)
        slice_mask = np.take(mask, mask.shape[i] // 2, axis=i)

        ax[i].imshow(slice_vol, cmap="gray")

        for __, label_val in label_dict.items():
            # Create binary mask for this label
            label_mask = (slice_mask == label_val).astype(float)
            if np.any(label_mask):  # Only draw if label exists in this slice
                color = LABEL_COLORS[label_val % len(LABEL_COLORS)]
                ax[i].contour(label_mask, levels=[0.5], colors=color, linewidths=0.5)

        ax[i].set_title("Fixed" if i == 0 else "")
        ax[i].axis('off')

    # Create legend
    legend_elements = [Patch(facecolor=LABEL_COLORS[label_val % len(LABEL_COLORS)],
                            label=f'{label_name}')
                      for label_name, label_val in label_dict.items()]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.98, 0.5), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend



    # Log to TensorBoard if writer is provided
    if writer is not None:
        # Convert figure to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)

        # Convert to CHW format for TensorBoard (height, width, channels) -> (channels, height, width)
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))

        # Add to TensorBoard
        writer.add_image(tag, image_array, global_step)
        buf.close()

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()




def viz_mid_axial_slices_comparison(vol, label_pred, label_gt, labels, filename=None, axis=0, dices=None):
    """
    Visualize predicted and ground truth labels side by side for all axial slices containing labels.

    Args:
        vol: 3D volume array
        label_pred: 3D predicted segmentation mask
        label_gt: 3D ground truth segmentation mask
        labels: Label dictionary
        filename: Optional filename to save the figure
    """
    # Get slices that contain either prediction or ground truth labels
    slice_num_gt = np.stack(np.where(label_gt > 0), axis=-1)
    slices_gt = np.unique(slice_num_gt[:, axis]) if len(slice_num_gt) > 0 else []
    slices = np.unique(np.concatenate([slices_gt]))

    fig, axes = plt.subplots(len(slices), 2, figsize=(16, 8*len(slices)))

    # Handle case of single slice
    if len(slices) == 1:
        axes = axes.reshape(1, -1)

    for j in range(len(slices)):
        # Prediction column
        vol_slice = np.take(vol, slices[j], axis=axis)
        pred_slice = np.take(label_pred, slices[j], axis=axis)
        gt_slice = np.take(label_gt, slices[j], axis=axis)
        ax_viz_mid_slice(axes[j, 0], vol_slice, pred_slice, labels)
        axes[j, 0].set_title(f"Slice {slices[j]} - Prediction" if j == 0 else "")

        # Ground truth column
        ax_viz_mid_slice(axes[j, 1], vol_slice, gt_slice, labels)
        axes[j, 1].set_title(f"Slice {slices[j]} - Ground Truth" if j == 0 else "")

    add_legends_v2(fig, labels, dices)

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()



def viz_mid_slices_by_dice(vol, mask_pred, mask_gt, dice_scores, labels=None, dice_threshold=0.7,
                           filename=None, writer=None, tag="", global_step=0):
    """
    Visualize predicted and ground truth masks side by side, highlighting labels with dice scores below threshold.

    Args:
        vol: 3D volume array
        mask_pred: 3D predicted segmentation mask
        mask_gt: 3D ground truth segmentation mask
        dice_scores: Dictionary mapping label names/values to their dice scores
        labels: Label dictionary or list
        dice_threshold: Dice score threshold below which labels are highlighted (default 0.7)
        filename: Optional filename to save the figure
        writer: Optional TensorBoard writer
        tag: Tag for TensorBoard logging
        global_step: Global step for TensorBoard logging
    """
    assert vol.shape == mask_pred.shape == mask_gt.shape

    # Handle labels parameter
    if labels is None:
        # Get unique labels from both masks
        label_values = np.unique(np.concatenate([mask_pred.flatten(), mask_gt.flatten()]))
        label_values = label_values[label_values != 0]  # Exclude background (0)
        label_dict = {f'Label {int(val)}': int(val) for val in label_values}
    elif isinstance(labels, dict):
        # Filter out background if present
        label_dict = {name: val for name, val in labels.items() if val != 0}
    else:
        # Assume it's a list/array of label values
        label_values = [val for val in labels if val != 0]
        label_dict = {f'Label {int(val)}': int(val) for val in label_values}

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    for i in range(3):
        slice_vol = np.take(vol, vol.shape[i] // 2, axis=i)
        slice_pred = np.take(mask_pred, mask_pred.shape[i] // 2, axis=i)
        slice_gt = np.take(mask_gt, mask_gt.shape[i] // 2, axis=i)

        # Prediction column
        axes[i, 0].imshow(slice_vol, cmap="gray")
        for label_name, label_val in label_dict.items():
            label_mask = (slice_pred == label_val).astype(float)
            if np.any(label_mask):
                color = LABEL_COLORS[label_val % len(LABEL_COLORS)]

                # Check if dice score is below threshold
                dice = dice_scores.get(label_name, dice_scores.get(label_val, 1.0))
                linewidth = 2.0 if dice < dice_threshold else 0.5
                linestyle = '--' if dice < dice_threshold else '-'

                axes[i, 0].contour(label_mask, levels=[0.5], colors=color,
                                  linewidths=linewidth, linestyles=linestyle)
        axes[i, 0].set_title("Prediction" if i == 0 else "")
        axes[i, 0].axis('off')

        # Ground truth column
        axes[i, 1].imshow(slice_vol, cmap="gray")
        for label_name, label_val in label_dict.items():
            label_mask = (slice_gt == label_val).astype(float)
            if np.any(label_mask):
                color = LABEL_COLORS[label_val % len(LABEL_COLORS)]

                # Check if dice score is below threshold
                dice = dice_scores.get(label_name, dice_scores.get(label_val, 1.0))
                linewidth = 2.0 if dice < dice_threshold else 0.5
                linestyle = '--' if dice < dice_threshold else '-'

                axes[i, 1].contour(label_mask, levels=[0.5], colors=color,
                                  linewidths=linewidth, linestyles=linestyle)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis('off')

    # Create legend with dice scores
    legend_elements = []
    for label_name, label_val in label_dict.items():
        dice = dice_scores.get(label_name, dice_scores.get(label_val, None))
        if dice is not None:
            dice_str = f" (Dice: {dice:.3f})"
            marker = " ⚠️" if dice < dice_threshold else ""
        else:
            dice_str = ""
            marker = ""

        legend_elements.append(
            Patch(facecolor=LABEL_COLORS[label_val % len(LABEL_COLORS)],
                  label=f'{label_name}{dice_str}{marker}')
        )

    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.98, 0.5), frameon=True, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend

    # Log to TensorBoard if writer is provided
    if writer is not None:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)

        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))

        writer.add_image(tag, image_array, global_step)
        buf.close()

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()