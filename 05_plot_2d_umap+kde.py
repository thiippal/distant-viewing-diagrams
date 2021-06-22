# -*- coding: utf-8 -*-

# Import libraries
import h5py
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Initialize seaborn
sns.set(style="white")

# Define path to HDF5 database
db_file = "features/blob_features_ycbcr_hsv_gray_lbp.h5"

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

# Initialize and fit UMAP to the data
result_umap = umap.UMAP(n_components=2,
                        n_neighbors=n_neighbours,
                        min_dist=min_dist,
                        verbose=True,
                        random_state=42).fit_transform(features)

# Store UMAP results into the DataFrame
df['umap_1'] = result_umap[:, 0]
df['umap_2'] = result_umap[:, 1]

# Replace empty spaces with NaN and filter DataFrame for AI2D-RST diagrams
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(subset=['ai2d_rst_category'])

# Optionally, write features to disk
# df.to_csv('ai2d_blobs_umap_features.csv', index=False)

# Define target categories
target_categories = ['cross-section', 'cut-out', 'illustration']

# Filter the data for target categories
df = df.loc[df['ai2d_rst_category'].isin(target_categories)]

# Define colors for target categories
color_map = {'cross-section': 'g', 'cut-out': 'r', 'illustration': 'b'}

# Plot target categories on marginal X- and Y-axes
for target_cat in target_categories:

    # Plot joint plot
    jp = sns.jointplot(x=df.loc[df['ai2d_rst_category'] == target_cat, 'umap_1'],
                       y=df.loc[df['ai2d_rst_category'] == target_cat, 'umap_2'],
                       kind='hex', color=color_map[target_cat], space=0,
                       joint_kws={'gridsize': 12},
                       marginal_kws={'kde': True, 'bins': 20})

    # Set axis sizes manually
    jp.ax_marg_x.set_xlim(-5, 17.5)
    jp.ax_marg_y.set_ylim(-12.5, 10)

    jp.savefig(f"{target_cat}_gray+lbp_hex+bin+kde.pdf")