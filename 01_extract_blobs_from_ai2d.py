# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import cv2
import json
from pathlib import Path
from hdf5writer import DataWriter
from skimage import feature
from tqdm import tqdm

"""
Usage:
	1. Download the AI2D corpus from https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip
	2. Extract the AI2D corpus into the directory "ai2d"
	3. Run the script using the command below:
	
		python 01_extract_blobs_from_ai2d.py
  
	4. The blobs will be placed in the directory "png_blobs/"
"""

# Set up paths to AI2D annotation and images
ai2d_json_dir = Path("ai2d/annotations/")
ai2d_img_dir = Path("ai2d/images/")
ai2d_rst_dir = Path("ai2d/ai2d-rst")

# Calculate the number of JSON files
ai2d_json = list(ai2d_json_dir.glob("*.json"))

# Create target directory
Path("png_blobs").mkdir()

# Set up progress bar
with tqdm(total=len(ai2d_json)) as pbar:

    # Loop over the AI2D JSON files
    for (b, i) in enumerate(range(0, len(ai2d_json), 20)):

        # Fetch files from list
        json_files = ai2d_json[i: i + 20]

        # Loop over AI2D annotation
        for ann_file in json_files:

            # Get path to image; cast to string for OpenCV
            img_file = str(ai2d_img_dir / ann_file.stem)

            # Load and convert image
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Open file containing JSON annotation
            with open(ann_file, 'r') as json_file:

                # Load JSON
                ann = json.load(json_file)

                # Get blobs
                blobs = ann['blobs']

                # Loop over blobs
                for blob in blobs.keys():
                    # Get polygon and cast to NumPy array
                    polygon = np.array([blobs[blob]['polygon']])

                    # Create empty mask
                    mask = np.zeros((img.shape[0], img.shape[1]),
                                    dtype=np.uint8)

                    # Fill the masked area with white pixels (255)
                    cv2.fillPoly(mask, polygon, (255))

                    # Add masks to the images with different colour spaces
                    masked_img = cv2.bitwise_and(img, img, mask=mask)

                    # Add mask as alpha channel
                    masked_img = cv2.merge([masked_img, mask])

                    # Convert to BGRA
                    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGRA)

                    # Get filename
                    filename = f"{ann_file.stem}_{blob}.png"

                    # Get bounding box
                    bbox = cv2.boundingRect(polygon)

                    # Crop the image
                    cropped_bbox = masked_img[bbox[1]: bbox[1] + bbox[3],
                                   bbox[0]: bbox[0] + bbox[2]]

                    # Write cropped image to disk
                    cv2.imwrite(f"png_blobs/{filename}", cropped_bbox)

        # Update progress bar
        pbar.update(20)
