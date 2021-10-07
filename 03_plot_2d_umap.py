# -*- coding: utf-8 -*-

# Import libraries
import h5py
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


# Define a function for processing logo images
def plot_image(path):

    # Read image
    img = Image.open(path)

    # Resize to thumbnail size
    img.thumbnail((15, 15), Image.LANCZOS)

    return OffsetImage(img)


# Initialize seaborn
sns.set_style("whitegrid")

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
n_dims = 2

# Set up pandas DataFrame to hold the results
df = pd.DataFrame()

# Add diagram and element IDs and AI2D/RST categories to the DataFrame
df['filename'] = filenames
df['ai2d_category'] = ai2d_category
df['ai2d_rst_category'] = ai2d_rst_category

# Stack features horizontally
features = np.hstack([gray_hist, lbp_hist])

# HDF5 insists on storing the data as bytes (prefix 'b'); convert to UTF-8 str
df['filename'] = df['filename'].str.decode('utf-8')
df['ai2d_category'] = df['ai2d_category'].str.decode('utf-8')
df['ai2d_rst_category'] = df['ai2d_rst_category'].str.decode('utf-8')

# Initialize and fit UMAP to the data
result_umap = umap.UMAP(n_components=n_dims,
                        n_neighbors=n_neighbours,
                        min_dist=min_dist,
                        verbose=True,
                        random_state=42).fit_transform(features)

# Store UMAP results into the DataFrame
df['umap_1'] = result_umap[:, 0]
df['umap_2'] = result_umap[:, 1]

# Save UMAP features to disk
df.to_pickle(f'lbp+gray_{n_dims}-dim_umap_features_{min_dist}_{n_neighbours}.pkl')

# Set up figure
fig = plt.figure(constrained_layout=True, figsize=(12, 8))

# Set up main axis
main_ax = fig.add_subplot(111)
main_ax.grid(False)

# Plot UMAP features (otherwise thumbnails are not rendered)
main_ax.scatter(x=df['umap_1'], y=df['umap_2'])

# Begin looping over the data to plot
for x, y, img in zip(df['umap_1'], df['umap_2'], df['filename']):

    # Create annotation box
    ab_m = AnnotationBbox(plot_image(f"png_blobs/{img}"), (x, y), frameon=False)

    # Add annotation box to the axis
    main_ax.add_artist(ab_m)

# Plot to render everything
plt.plot()

# Save visualisation to disk
plt.savefig(f"umap_gray+lbp_{n_neighbours}_{min_dist}.jpg",
            bbox_inches="tight",
            dpi=600)
