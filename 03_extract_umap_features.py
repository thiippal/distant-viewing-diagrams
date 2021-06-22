# -*- coding: utf-8 -*-

# Import libraries
import h5py
import umap
import numpy as np
import pandas as pd


# Define path to HDF5 database
db_file = "blob_features_ycbcr_hsv_gray_lbp.h5"

# Load database
db = h5py.File(db_file, 'r')

# Assign features and identifiers to variables
ycbcr_hist = db['ycbcr_hist']
hsv_hist = db['hsv_hist']
lbp_hist = db['lbp_hist']
gray_hist = db['gray_hist']
filenames = db['filename']
ai2d_category = db['ai2d_category']
ai2d_rst_category = db['ai2d_rst_category']

# Define variables
n_neighbours = 200
min_dist = 0.99
n_dims = 10

# Stack features horizontally
features = np.hstack([lbp_hist, gray_hist])

# Initialize and fit UMAP to the data
result_umap = umap.UMAP(n_components=n_dims,
                        n_neighbors=n_neighbours,
                        min_dist=min_dist,
                        verbose=True,
                        random_state=42).fit_transform(features)

# Set up a DataFrame to hold the UMAP features
df = pd.DataFrame(data=result_umap)

# Add diagram and element IDs and AI2D/RST categories to the DataFrame
df['filename'] = filenames
df['ai2d_category'] = ai2d_category
df['ai2d_rst_category'] = ai2d_rst_category

# Write DataFrame to disk
df.to_pickle(f'lbp+gray_{n_dims}-dim_umap_features_{min_dist}_{n_neighbours}.pkl')
