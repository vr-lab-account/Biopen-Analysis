# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:19:03 2020

@author: Vishnu Rao
"""
import glob
from skimage import io
import biopen_analysis_functions_2 as baf
import numpy as np

parent_path = 'E:/Rizzo Lab/Experiments/Biopen/INS1_Clusters_Carb_NADH_07_24_20/'

#Imports all images (glucose, buffer, and rhodamine for NAD(P)H) into arrays
glucose_flist = glob.glob(parent_path+'NADH_Images/*.tif')
im_glucose = np.array([np.array(io.imread(fname)) for fname in glucose_flist])

buffer_flist = glob.glob(parent_path+'Buffer_Images/*.tif')
im_buffer = np.array([np.array(io.imread(fname)) for fname in buffer_flist])

rhod_flist = glob.glob(parent_path+'Rhodamine_Images/*.tif')
im_rhod = np.array([np.array(io.imread(fname)) for fname in rhod_flist])

#Sets background intensity and upper limit for glucose and buffer images
back_inten_NADH = 800
upperlim_NADH = 8000

#Array of cluster/islet identifiers for each image
islet_id = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15']

#Paths were images/spreadsheets will be saved
path1 = parent_path + 'Analysis/Glucose/'
path2 = parent_path + 'Analysis/Buffer/'
path3 = parent_path + 'Analysis/Stim_unstim/'

#Sets the number of iterations to the number of images in the array
iternum = len(im_glucose)

for i in range(0,iternum):    
    #Background substraction and norrmalization of glucose stimulated clusters/islets
    im_g_minus_back, islet_region = baf.clusterregion_backsub(im_glucose[i], back_inten_NADH, upperlim_NADH)
    smaller_islet_region, top_border, bottom_border, left_border, right_border = baf.smaller_isletregion(im_g_minus_back, islet_region)
    im_g_align = baf.ba_align(im_g_minus_back, top_border, bottom_border, left_border, right_border)
    im_g_final = im_g_align * islet_region * smaller_islet_region
    im_g_norm = baf.baseline_normalize(im_g_final)

    #Determines cluster/islet diameter
    baf.islet_diam(top_border, bottom_border, left_border, right_border, path3, islet_id[i])

    #Determines of stimulated and unstimulated regions as well as stimulation edge
    stim, unstim = baf.stim_unstim_regions(im_rhod[i], islet_region, smaller_islet_region, path3, islet_id[i])
    islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols = baf.islet_edge_stimulation_edge(im_g_norm, stim, left_border, right_border, path3, islet_id[i])
    
    #Determines max response vs distance to cluster/islet edge in stimulated regions and max response vs distance to stimulation edge in unstimulated regions for glucose stimulation
    baf.islet_edge_stim_edge_vs_response(im_g_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path1, islet_id[i])
    #Saves normalized image
    io.imsave(path3+islet_id[i]+'_'+'glucose_norm.tif', im_g_norm.astype(np.float32))
    
    
    #Background substraction and norrmalization of buffer stimulated clusters/islets
    im_b_minus_back, islet_region = baf.clusterregion_backsub(im_buffer[i], back_inten_NADH, upperlim_NADH)
    im_b_align = baf.ba_align(im_b_minus_back, top_border, bottom_border, left_border, right_border)
    im_b_final = im_b_align * islet_region * smaller_islet_region
    im_b_norm = baf.baseline_normalize(im_b_final)
    
    #Determines max response vs distance to cluster/islet edge in stimulated regions and max response vs distance to stimulation edge in unstimulated regions for glucose stimulation
    baf.islet_edge_stim_edge_vs_response(im_b_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path2, islet_id[i])
    #Saves normalized image
    io.imsave(path3+islet_id[i]+'_'+'buffer_norm.tif', im_b_norm.astype(np.float32))
    
    
    #Deteremines response in stimulated and unstimulated regions for glucose and buffer stimulation (NADH)
    baf.stim_unstim_response (im_g_norm, stim, unstim, path1, islet_id[i])
    baf.stim_unstim_response (im_b_norm, stim, unstim, path2, islet_id[i])
