from train_segmentation import (
    data_handler,
    get_val_transforms,
    monai,
    DataLoader,
    list_data_collate,
    worker_init_fn,
    torch,
    load_model,
    sliding_window_inference,
    viz_mid_slices,
    json,
    os
    #...
)
import pandas as pd


def val(opt):

    _, __, vaimages, vasegs = data_handler(
        opt.dataset, opt.train_amount, opt.n_iters_per_epoch, opt.batch_size,
    )

    val_files = [
        {"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)
    ]

    # define transforms for image and segmentation
    val_transforms = get_val_transforms()

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
        freeze_mode='none',
    )

    dir_save = f'finetuning_runs'

    checkpoint_filepath = f'{dir_save}/checkpoints/{opt.exp_name}/best_dict_epoch0444.pth'
    new_model.load_state_dict(torch.load(checkpoint_filepath, weights_only=True))
    new_model.eval()


    dataset_json = json.load(open(os.path.join(opt.dataset, 'dataset.json')))
    labels_list = dataset_json['labels']

    # fake one (we should put it inside the checkpoint perhaps)
    # dataset_json = json.load(open(os.path.join('/home/eperot/nnUNet_raw/Dataset903_baselineCT_oneview_without_clahe/', 'dataset.json')))
    # labels_list = dataset_json['labels']

    valloss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=False, reduction="none")
    demo_dir = f'{dir_save}/demo_outputs/{opt.exp_name}/'
    os.makedirs(demo_dir, exist_ok=True)
    all_dices = []
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            roi_size = (opt.crop_size, opt.crop_size, opt.crop_size)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size,
                new_model, overlap=0.7,
            )

            # handle partial annots
            annotated_slices = torch.unique(torch.nonzero(val_labels.squeeze())[:,0])
            subvol_val_labels = val_labels[:,:,annotated_slices]
            subvol_val_outputs = val_outputs[:,:,annotated_slices]
            dices = 1-valloss_function(subvol_val_outputs, subvol_val_labels).squeeze()

            #dices = 1-valloss_function(val_outputs, val_labels).squeeze()
            dices = dices.cpu().numpy()
            case_dices = {'case': f'case_{i}'}
            for label_name, idx in labels_list.items():
                if idx == 0:
                    continue
                dice_value = dices[idx-1]
                case_dices[label_name] = dice_value
                print(f"{label_name}: Dice {dice_value:.4f}")

            all_dices += [case_dices]

            if opt.viz:
                img = val_images[0,0].cpu().numpy()
                labels = val_outputs.argmax(dim=1)[0].cpu().numpy()
                viz_mid_slices(
                    img,
                    labels,
                    labels_list,
                    filename=f'{demo_dir}/demo_output_sample{i}_image_output.png')


    df = pd.DataFrame(all_dices)
    # Calculate average row
    avg_row = {'case': 'average'}
    for col in df.columns:
        if col != 'case':
            avg_row[col] = df[col].mean()
    # Insert average as first row
    df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    # Save to CSV
    csv_path = f'{demo_dir}/dice_scores.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved dice scores to {csv_path}")
