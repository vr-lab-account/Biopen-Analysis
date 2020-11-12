# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:44:21 2020

@author: Vishnu
"""

import glob
from skimage import io
from scipy import ndimage
import numpy as np



#Imports all images (glucose, buffer, and rhodamine for NAD(P)H and Fluo-4) into arrays
NADH_g_flist = glob.glob('E:/Rizzo Lab/Experiments/Biopen/INS1_Clusters_Carb_NADH_07_24_20/Unsmoothed/NADH_Images/*.tif')
im_NADH_g = np.array([np.array(io.imread(fname)) for fname in NADH_g_flist])

NADH_b_flist = glob.glob('E:/Rizzo Lab/Experiments/Biopen/INS1_Clusters_Carb_NADH_07_24_20/Unsmoothed/Buffer_Images/*.tif')
im_NADH_b = np.array([np.array(io.imread(fname)) for fname in NADH_b_flist])

Fluo4_g_flist = glob.glob('E:/Rizzo Lab/Experiments/Biopen/Biopen_Verap_NADH_Fluo4_08_05_20/Unsmoothed/Fluo4_Glucose_Images/*.tif')
im_Fluo4_g = np.array([np.array(io.imread(fname)) for fname in Fluo4_g_flist])

Fluo4_b_flist = glob.glob('E:/Rizzo Lab/Experiments/Biopen/Biopen_Verap_NADH_Fluo4_08_05_20/Unsmoothed/Fluo4_Buffer_Images/*.tif')
im_Fluo4_b = np.array([np.array(io.imread(fname)) for fname in Fluo4_b_flist])

#Array of islet identifiers for each image.
islet_id = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16']

#Sets the number of iterations to the number of images in the array
iternum = len(im_NADH_g)

for i in range(0,iternum):
    #Smoothes all images using a median filter.
    bilateral_ng = ndimage.median_filter(im_NADH_g[i],3)
    io.imsave("E:/Rizzo Lab/Experiments/Biopen/New_analysis_GCK_islets_02_09_19/NADH_Images/"+islet_id[i]+"_glucose.tif", bilateral_ng)
    
    bilateral_nb = ndimage.median_filter(im_NADH_b[i],3)
    io.imsave("E:/Rizzo Lab/Experiments/Biopen/INS1_Clusters_Carb_NADH_07_24_20/Buffer_Images/"+islet_id[i]+"_buffer.tif", bilateral_nb)
    
    bilateral_fg = ndimage.median_filter(im_Fluo4_g[i],3)
    io.imsave("E:/Rizzo Lab/Experiments/Biopen/Biopen_Verap_NADH_Fluo4_08_05_20/Fluo4_Glucose_Images/"+islet_id[i]+"_glucose_Fluo4.tif", bilateral_fg)
    
    bilateral_fb = ndimage.median_filter(im_Fluo4_b[i],3)
    io.imsave("E:/Rizzo Lab/Experiments/Biopen/Biopen_Verap_NADH_Fluo4_08_05_20/Fluo4_Buffer_Images/"+islet_id[i]+"_buffer_Fluo4.tif", bilateral_fb)