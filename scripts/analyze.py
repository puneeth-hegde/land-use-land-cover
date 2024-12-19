import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# Function to get the unique classes from the segmentation mask (automatically reads them)
def get_classes_from_image(image_path):
    # Open the mask image using PIL
    img = Image.open(image_path)
    
    # Convert the image into a numpy array
    img_array = np.array(img)
    
    # Get the unique pixel values in the mask image, which represent the classes
    unique_classes = np.unique(img_array)
    
    return unique_classes, img_array

# Function to calculate the area of each class in the segmentation mask
def get_area_data(region, year):
    region_path = Path(f'data/raw/{region}')
    image_files = list(region_path.glob(f'*_{year}.tif'))  # Assuming image names have year suffix
    
    if not image_files:
        return None
    
    image_file = image_files[0]
    
    # Get the unique classes from the image mask
    classes, img_array = get_classes_from_image(image_file)
    
    # Calculate the area for each class (by counting the pixels that belong to each class)
    areas = {cls: np.sum(img_array == cls) for cls in classes}
    
    return areas

# Function to analyze the areas for a region and store the results
def analyze_region(region):
    years = get_years(region)
    analysis_results = []
    
    for year in years:
        area_data = get_area_data(region, year)
        
        if area_data:
            analysis_results.append({'Region': region, 'Year': year, **area_data})
    
    return analysis_results

# Function to get list of years for a particular region
def get_years(region):
    region_dir = Path(f'data/raw/{region}')
    image_files = list(region_dir.glob('*.tif'))  # Find all TIFF images for this region
    
    years = set()
    for image in image_files:
        year = image.stem.split('_')[-1]  # Assuming image filename ends with the year
        years.add(int(year))
    
    return sorted(years)

# Function to save the analysis results to a CSV file
def save_analysis_results(analysis_results):
    # Create the analysis_results directory if it doesn't exist
    analysis_dir = Path('data/analysis_results')
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the results as a CSV file
    results_file = analysis_dir / 'area_analysis1.csv'
    
    # Convert the results into a DataFrame
    df = pd.DataFrame(analysis_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(results_file, index=False)
    print(f"Analysis results saved to {results_file}")

# Main function to analyze all regions and save the results
def main():
    regions = ['region_1', 'region_2', 'region_3']  # List of regions you are analyzing
    
    all_analysis_results = []
    
    for region in regions:
        print(f"Analyzing {region}...")
        region_analysis = analyze_region(region)
        all_analysis_results.extend(region_analysis)
    
    # Save the aggregated analysis results
    save_analysis_results(all_analysis_results)

if __name__ == "__main__":
    main()
