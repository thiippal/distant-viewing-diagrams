# -*- coding: utf-8 -*-

# Import libraries
import h5py
import umap
import numpy as np
import pandas as pd
import seaborn as sns


# Initialize seaborn
sns.set(style="white")

# Read UMAP features from pickle
df = pd.read_pickle('lbp+gray_2-dim_umap_features_0.99_200.pkl')

# Replace 'none' with NaN and filter DataFrame for AI2D-RST diagrams
df = df.replace(r'none', np.nan, regex=True)
df = df.dropna(subset=['ai2d_rst_category'])

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