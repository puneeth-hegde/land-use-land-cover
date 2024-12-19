import os
import torch
from torchvision import transforms
from utils.model_utils import load_model
from utils.image_utils import save_image, mask_to_label

def segment_images():
    # Define paths
    processed_dir = './data/processed'
    labels_dir = './data/labels'

    os.makedirs(labels_dir, exist_ok=True)

    for region in ['region_1', 'region_2', 'region_3']:
        region_folder = os.path.join(processed_dir, region)
        region_labels_folder = os.path.join(labels_dir, region)
        os.makedirs(region_labels_folder, exist_ok=True)

        # Load the pre-trained UNet model
        model = load_model('./models/unet_model.pth')

        for image_file in os.listdir(region_folder):
            image_path = os.path.join(region_folder, image_file)
            image = load_image(image_path)

            # Preprocess and segment the image
            segmented_image = model(image)

            # Convert the segmented mask to a label
            label = mask_to_label(segmented_image)

            # Save the label image
            save_image(label, os.path.join(region_labels_folder, image_file))

    print("Image segmentation completed.")
