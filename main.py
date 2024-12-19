import os
from scripts.preprocess import preprocess_images
from scripts.segment_images import segment_images
from scripts.analyze import analyze_area
from scripts.train_unet import train_unet
from scripts.dashboard import create_dashboard

def main():
    # Preprocess images (chronologically sort and resize if needed)
    preprocess_images()

    # Segment images and create labels
    segment_images()

    # Analyze areas of geographical aspects
    analyze_area()

    # Train the UNet model
    train_unet()

    # Create the dashboard
    create_dashboard()

if __name__ == '__main__':
    main()
