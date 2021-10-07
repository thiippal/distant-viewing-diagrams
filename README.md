# Semiotically-grounded distant viewing of diagrams: insights from two multimodal corpora

## Description

This repository contains code associated with the article *Semiotically-grounded distant viewing of diagrams: insights from two multimodal corpora* by Tuomo Hiippala and John Bateman, published in [Digital Scholarship in the Humanities](https://doi.org/10.1093/llc/fqab063) (open access).

## Preliminaries

To reproduce the results reported in the article, you must first download the following data:

 1. The Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset ([direct download](http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip))
 2. The AI2D-RST corpus ([direct download](https://korp.csc.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip))

You should also [create a fresh virtual environment](https://docs.python.org/3/library/venv.html) for Python 3.8+ and install the libraries defined in `requirements.txt` using the following command:

`pip install -r requirements.txt`

## Codebase

`01_extract_blobs_from_ai2d.py` extracts graphic elements classified as "blobs" from the AI2D corpus and stores them into a directory named `png_blobs`.

`02_extract_colour_texture_features.py` extracts colour histograms and local binary patterns from the blobs. The results are stored into a HDF5 file named `blob_features_ycbcr_hsv_gray_lbp.h5`.

`03_extract_umap_features.py` learns two-dimensional UMAP features for the 90-dimensional colour and texture features.

`04_plot_2d_umap.py` plots the two-dimensional UMAP features.

`05_plot_2d_umap+kde.py` plots a joint plot with UMAP features and kernel density estimations for selected diagram categories (e.g. cross-sections, cut-outs and illustrations).

`06_plot_alluvial.R` plots an alluvial graph that maps the AI2D diagram categories to AI2D-RST diagram categories. You will need the `ggplot2` and `ggalluvial` libraries to run this script.

## Contact

Questions? Open an issue on GitHub or e-mail me at tuomo dot hiippala @ helsinki dot fi.
