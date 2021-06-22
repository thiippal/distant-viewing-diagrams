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
    img.thumbnail((30, 30), Image.LANCZOS)

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

# Set up pandas DataFrame to hold the results
df = pd.DataFrame()

# Add diagram and element IDs and AI2D/RST categories to the DataFrame
df['filename'] = filenames
df['ai2d_category'] = ai2d_category
df['ai2d_rst_category'] = ai2d_rst_category

# Stack features horizontally
features = np.hstack([gray_hist, lbp_hist])

# Scale down for demo
# features = features[:500]
# df = df[:500]

# Initialize and fit UMAP to the data
result_umap = umap.UMAP(n_components=2,
                        n_neighbors=n_neighbours,
                        min_dist=min_dist,
                        verbose=True,
                        random_state=42).fit_transform(features)

# Store UMAP results into the DataFrame
df['umap_1'] = result_umap[:, 0]
df['umap_2'] = result_umap[:, 1]

# Save UMAP features to disk
df.to_hdf("umap_features.h5", "df")

exit()

# Set up figure and gridspec
fig = plt.figure(constrained_layout=True, figsize=(12, 8))
gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3, 1])

# Set up main axis
main_ax = fig.add_subplot(gs[0, :])
main_ax.grid(False)

# Set up the zoom-in axes below the main axis
zoom_1 = fig.add_subplot(gs[1, 0])
zoom_2 = fig.add_subplot(gs[1, 1])
zoom_3 = fig.add_subplot(gs[1, 2])
zoom_1.grid(False)

# Plot UMAP features (otherwise thumbnails are not rendered)
main_ax.scatter(x=df['umap_1'], y=df['umap_2'])

# Add first subplot for zooming in
zoom_1.margins(0)
zoom_1.set_xlim(22, 24)
zoom_1.set_ylim(32, 34)
zoom_1.scatter(x=df['umap_1'], y=df['umap_2'])

# Begin looping over the data to plot
for x, y, img in zip(df['umap_1'], df['umap_2'], df['filename']):

    # Create annotation box
    ab_m = AnnotationBbox(plot_image(f"png_blobs/{img}"), (x, y), frameon=False)
    ab_z1 = AnnotationBbox(plot_image(f"png_blobs/{img}"), (x, y), frameon=False)
    ab_z1.set_clip_on(True)


    # Add annotation box to the axis
    main_ax.add_artist(ab_m)
    zoom_1.add_artist(ab_z1)
    
plt.show()

exit()

plt.plot()

plt.savefig(f"umap_gray+lbp_{n_neighbours}_{min_dist}.jpg", bbox_inches="tight", dpi=600)
