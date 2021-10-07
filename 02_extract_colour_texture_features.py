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
	1. Download the AI2D-RST corpus from https://korp.csc.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip
	2. Extract the AI2D-RST corpus; move into the directory "json", and copy the directory "ai2d-rst"
	   and the file "categories_ai2d-rst.json" into the directory with the AI2D dataset ("ai2d")
	3. Run the script using the command below:
	
		python 02_extract_colour_texture_features.py
		
	4. The resulting features will be stored into a HDF5 file named "blob_features_ycbcr_hsv_gray_lbp.h5".
	
"""


# Set up paths to AI2D annotation, images and categories
ai2d_json_dir = Path("ai2d/annotations/")
ai2d_img_dir = Path("ai2d/images/")
ai2d_rst_dir = Path("ai2d/ai2d-rst")
ai2d_cats = Path("ai2d/categories.json")
ai2d_rst_cats = Path("ai2d/categories_ai2d-rst.json")

# Set up HDF5 database
db = DataWriter(n_data=len(list(Path("png_blobs/").glob("*.png"))),
                output_path="blob_features_ycbcr_hsv_gray_lbp.h5",
                bufsize=256)

# Calculate the number of JSON files
ai2d_json = list(ai2d_json_dir.glob("*.json"))

# Load categories for both AI2D and AI2D-RST
with open(ai2d_cats, 'r') as ai2d_cats_file:

    # Load JSON
    ai2d_categories = json.load(ai2d_cats_file)

    # Close file
    ai2d_cats_file.close()

with open(ai2d_rst_cats, 'r') as ai2d_rst_cats_file:

    # Load JSON
    ai2d_rst_categories = json.load(ai2d_rst_cats_file)

    # Close file
    ai2d_rst_cats_file.close()

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

            # Load image
            img = cv2.imread(img_file)

            # Convert BGR colour space to YCbCr, HSV and grayscale
            img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Open file containing JSON annotation
            with open(ann_file, 'r') as json_file:

                # Set up placeholder lists for features and filenames (fn)
                ycbcr_batch, hsv_batch, lbp_batch, fn_batch = [], [], [], []
                gray_batch = []
                ai2d_cat_batch, ai2d_rst_cat_batch = [], []

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
                    masked_img_y = cv2.bitwise_and(img_ycbcr, img_ycbcr,
                                                   mask=mask)
                    masked_img_h = cv2.bitwise_and(img_hsv, img_hsv,
                                                   mask=mask)
                    masked_img_g = cv2.bitwise_and(img_gray, img_gray,
                                                   mask=mask)

                    # Calculate a histogram for HSV
                    hist_3d_hsv = cv2.calcHist([img_hsv],  # image
                                               [0, 1, 2],  # channels
                                               mask=mask,  # mask
                                               histSize=[4, 4, 4],  # n of bins
                                               ranges=[0, 180, 0, 256, 0, 256]
                                               )

                    # Normalize histogram for HSV
                    hist_3d_hsv = cv2.normalize(hist_3d_hsv,    # source
                                                hist_3d_hsv,    # target
                                                0, 1,           # range
                                                cv2.NORM_MINMAX
                                                )

                    # Calculate a histogram for YCbCr
                    hist_3d_y = cv2.calcHist([img_ycbcr],  # image
                                             [0, 1, 2],  # channels
                                             mask=mask,  # mask
                                             histSize=[4, 4, 4],  # n of bins
                                             ranges=[0, 256, 0, 256, 0, 256]
                                             )

                    # Normalize histogram for YCbCr
                    hist_3d_y = cv2.normalize(hist_3d_y,    # source
                                              hist_3d_y,    # target
                                              0, 1,         # range
                                              cv2.NORM_MINMAX
                                              )

                    # Calculate a histogram for grayscale image
                    hist_1d_g = cv2.calcHist([img_gray],    # image
                                             [0],           # channel
                                             mask=mask,     # mask
                                             histSize=[64],     # n of bins
                                             ranges=[0, 256]
                                             )

                    # Normalize histogram for grayscale image
                    hist_1d_g = cv2.normalize(hist_1d_g,    # source
                                              hist_1d_g,    # target
                                              0, 1,         # range
                                              cv2.NORM_MINMAX
                                              )

                    # Calculate local binary patterns (LBP)
                    lbp_gray = feature.local_binary_pattern(masked_img_g,
                                                            24,  # points
                                                            3,  # radius
                                                            method='uniform')

                    # Calculate histogram for LBP with mask
                    (hist_lbp, _) = np.histogram(lbp_gray[mask > 0],
                                                 bins=range(0, 24 + 3),
                                                 range=(0, 24 + 2),
                                                 density=True)

                    # Get filename
                    filename = f"{ann_file.stem}_{blob}.png"

                    # Fetch AI2D and AI2D-RST categories
                    ai2d_cat = ai2d_categories[ann_file.stem]

                    try:
                        ai2d_rst_cat = ai2d_rst_categories[ann_file.stem]

                    except KeyError:

                        ai2d_rst_cat = 'none'

                    # Append histograms and features to the queue lists
                    ycbcr_batch.append(hist_3d_y.flatten())
                    hsv_batch.append(hist_3d_hsv.flatten())
                    lbp_batch.append(hist_lbp.flatten())
                    gray_batch.append(hist_1d_g.flatten())
                    fn_batch.append(filename)
                    ai2d_cat_batch.append(ai2d_cat)
                    ai2d_rst_cat_batch.append(ai2d_rst_cat)

                # Close file
                json_file.close()

                # Write files to database
                db.add(ycbcr=ycbcr_batch,
                       hsv=hsv_batch,
                       gray=gray_batch,
                       lbp=lbp_batch,
                       filename=fn_batch,
                       ai2d_cat=ai2d_cat_batch,
                       ai2d_rst_cat=ai2d_rst_cat_batch
                       )

        # Update progress bar
        pbar.update(20)

# Close database
db.close()
