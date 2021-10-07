# -*- coding: utf-8 -*-

# Import packages
import h5py

# Set the datatype for strings
dt = h5py.special_dtype(vlen=str)


class DataWriter:
    def __init__(self, n_data, output_path, bufsize=250):

        # Open the HDF5 database for writing
        self.db = h5py.File(output_path, 'w')

        # Create datasets that hold the extracted features and filenames
        self.ycbcr_hist = self.db.create_dataset("ycbcr_hist", (n_data, 64),
                                                 dtype="float")
        self.hsv_hist = self.db.create_dataset("hsv_hist", (n_data, 64),
                                               dtype="float")
        self.gray_hist = self.db.create_dataset("gray_hist", (n_data, 64),
                                                dtype="float")
        self.lbp_hist = self.db.create_dataset("lbp_hist", (n_data, 26),
                                               dtype="float")
        self.filenames = self.db.create_dataset("filename",
                                                (n_data,),
                                                dtype=dt)
        self.ai2d_category = self.db.create_dataset("ai2d_category",
                                                    (n_data,),
                                                    dtype=dt)
        self.ai2d_rst_category = self.db.create_dataset("ai2d_rst_category",
                                                        (n_data,),
                                                        dtype=dt)

        # Store buffer size and initialize buffer and index
        self.bufsize = bufsize
        self.buffer = {"ycbcr_hist": [], "hsv_hist": [], "gray_hist": [],
                       "lbp_hist": [], "filename": [],
                       "ai2d_cat": [], "ai2d_rst_cat": []}
        self.ix = 0

    def add(self, ycbcr, hsv, gray, lbp, filename, ai2d_cat, ai2d_rst_cat):

        # Add features and identifiers to the buffer
        self.buffer["ycbcr_hist"].extend(ycbcr)
        self.buffer["hsv_hist"].extend(hsv)
        self.buffer["gray_hist"].extend(gray)
        self.buffer["lbp_hist"].extend(lbp)
        self.buffer["filename"].extend(filename)
        self.buffer["ai2d_cat"].extend(ai2d_cat)
        self.buffer["ai2d_rst_cat"].extend(ai2d_rst_cat)

        # Check if the buffer needs to be written to the HDF5 file
        if len(self.buffer["filename"]) >= self.bufsize:

            self.flush()

    def flush(self):

        # Get current index
        i = self.ix + len(self.buffer["filename"])

        # Write buffer to disk
        self.ycbcr_hist[self.ix:i] = self.buffer["ycbcr_hist"]
        self.hsv_hist[self.ix:i] = self.buffer["hsv_hist"]
        self.gray_hist[self.ix:i] = self.buffer["gray_hist"]
        self.lbp_hist[self.ix:i] = self.buffer["lbp_hist"]
        self.filenames[self.ix:i] = self.buffer["filename"]
        self.ai2d_category[self.ix:i] = self.buffer["ai2d_cat"]
        self.ai2d_rst_category[self.ix:i] = self.buffer["ai2d_rst_cat"]

        # Update index
        self.ix = i

        # Reset buffer
        self.buffer = {"ycbcr_hist": [],
                       "hsv_hist": [],
                       "gray_hist": [],
                       "lbp_hist": [],
                       "filename": [],
                       "ai2d_cat": [],
                       "ai2d_rst_cat": []
                       }

    def close(self):

        # Check if any data remains in the buffer
        if len(self.buffer["filename"]) > 0:

            # Flush buffer
            self.flush()

        # Close the dataset
        self.db.close()
