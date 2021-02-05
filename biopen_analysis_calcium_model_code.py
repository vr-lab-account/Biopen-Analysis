# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:19:03 2020

@author: Vishnu Rao
"""
import glob
from skimage import io
import biopen_analysis_functions_model_code as baf
import numpy as np

#Sets parent path for image inputs and result outputs
parent_path = 'E:/Rizzo Lab/Experiments/Biopen/Biopen_NADH_Fluo4_06_19_20/'

#Imports all images (glucose, buffer, and rhodamine for NAD(P)H and Fluo-4) into arrays
NADH_g_flist = glob.glob(parent_path+'NADH_Glucose_Images/*.tif')
im_NADH_g = np.array([np.array(io.imread(fname)) for fname in NADH_g_flist])

NADH_b_flist = glob.glob(parent_path+'NADH_Buffer_Images/*.tif')
im_NADH_b = np.array([np.array(io.imread(fname)) for fname in NADH_b_flist])

rhod_flist = glob.glob(parent_path+'Rhodamine_Images/*.tif')
im_rhod = np.array([np.array(io.imread(fname)) for fname in rhod_flist])

Fluo4_g_flist = glob.glob(parent_path+'Fluo4_Glucose_Images/*.tif')
im_Fluo4_g = np.array([np.array(io.imread(fname)) for fname in Fluo4_g_flist])

Fluo4_b_flist = glob.glob(parent_path+'Fluo4_Buffer_Images/*.tif')
im_Fluo4_b = np.array([np.array(io.imread(fname)) for fname in Fluo4_b_flist])

#Sets background intensity and upper limit for NAD(P)H and Fluo4 images
back_inten_NADH = 700
upperlim_NADH = 7000
back_inten_Fluo4 = 300
upperlim_Fluo4 = 16383 *.95

#Paths were images/spreadsheets will be saved
path1 = parent_path + 'Analysis/NADH_glucose/'
path2 = parent_path + 'Analysis/NADH_buffer/'
path3 = parent_path + 'Analysis/Stim_unstim_2/'
path4 = parent_path + 'Analysis/Fluo4_glucose/'
path5 = parent_path + 'Analysis/Fluo4_buffer/'


#Array of cluster/islet identifiers for each image. M1I1 refers to mouse 1 islet 1.
islet_id = ['M1I1', 'M1I2', 'M1I3', 'M1I4', 'M1I5', 'M1I6', 'M2I1', 'M2I2', 'M2I3', 'M2I4', 'M2I5', 'M3I1', 'M3I2', 'M3I3', 'M3I4', 'M3I5']

#Sets the number of iterations to the number of images in the array
iternum = len(im_NADH_g)

for i in range(0,iternum):
    #Background substraction and norrmalization of glucose stimulated clusters/islets (NADH)
    im_NADH_g_minus_back, islet_region = baf.isletregion_backsub(im_NADH_g[i], back_inten_NADH, upperlim_NADH)
    smaller_islet_region, top_border, bottom_border, left_border, right_border = baf.smaller_isletregion(im_NADH_g_minus_back, islet_region)
    im_NADH_g_minus_back_aligned = baf.ba_align(im_NADH_g_minus_back, top_border, bottom_border, left_border, right_border)
    im_NADH_g_final = im_NADH_g_minus_back_aligned * islet_region * smaller_islet_region
    im_NADH_g_norm = baf.baseline_normalize(im_NADH_g_final)
    #Saves normalized images
    io.imsave(path3+islet_id[i]+'_'+'NADH_norm.tif', im_NADH_g_norm.astype(np.float32))
      
    #Background substraction and norrmalization of buffer stimulated clusters/islets (NADH)
    im_NADH_b_minus_back, trash_islet_region = baf.isletregion_backsub(im_NADH_b[i], back_inten_NADH, upperlim_NADH)
    im_NADH_b_minus_back_aligned = baf.ba_align(im_NADH_b_minus_back, top_border, bottom_border, left_border, right_border)
    im_NADH_b_final = im_NADH_b_minus_back_aligned * islet_region * smaller_islet_region
    im_NADH_b_norm = baf.baseline_normalize(im_NADH_b_final)
    #Saves normalized images
    io.imsave(path3+islet_id[i]+'_'+'NADH_norm_buffer.tif', im_NADH_b_norm.astype(np.float32))
    
    #Background substraction and norrmalization of glucose stimulated clusters/islets (Fluo4)
    im_Fluo4_g_minus_back, trash_islet_region = baf.isletregion_backsub(im_Fluo4_g[i], back_inten_Fluo4, upperlim_Fluo4)
    im_Fluo4_g_minus_back_aligned = baf.ba_align(im_Fluo4_g_minus_back, top_border, bottom_border, left_border, right_border)
    im_Fluo4_g_final = im_Fluo4_g_minus_back_aligned * islet_region * smaller_islet_region
    im_Fluo4_g_norm = baf.baseline_normalize(im_Fluo4_g_final)
    #Saves normalized images
    io.imsave(path3+islet_id[i]+'_'+'Fluo4_norm.tif', im_Fluo4_g_norm.astype(np.float32))
    
    #Background substraction and norrmalization of buffer stimulated clusters/islets (Fluo4)
    im_Fluo4_b_minus_back, trash_islet_region = baf.isletregion_backsub(im_Fluo4_b[i], back_inten_Fluo4, upperlim_Fluo4)
    im_Fluo4_b_minus_back_aligned = baf.ba_align(im_Fluo4_b_minus_back, top_border, bottom_border, left_border, right_border)
    im_Fluo4_b_final = im_Fluo4_b_minus_back_aligned * islet_region * smaller_islet_region
    im_Fluo4_b_norm = baf.baseline_normalize(im_Fluo4_b_final)  
    #Saves normalized images
    io.imsave(path3+islet_id[i]+'_'+'Fluo4_norm_buffer.tif', im_Fluo4_b_norm.astype(np.float32))
    
    #Determines stimulated and unstimulated regions as well as stimulation and islet edge
    stim, unstim = baf.stim_unstim_regions(im_rhod[i], islet_region, smaller_islet_region, path3, islet_id[i])
    islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols = baf.islet_edge_stimulation_edge(im_NADH_g_norm, stim, left_border, right_border, path3, islet_id[i])   
    
    #Determines islet diameter
    baf.islet_diam(top_border, bottom_border, left_border, right_border, path3, islet_id[i])
    
    #Deteremines response in stimulated and unstimulated regions, max response vs distance to cluster/islet edge in stimulated regions, and
    #max response vs distance to stimulation edge in unstimulated regions for glucose stimulation (NADH)
    baf.stim_unstim_response (im_NADH_g_norm, stim, unstim, path1, islet_id[i])
    baf.islet_edge_stim_edge_vs_response(im_NADH_g_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path1, islet_id[i])
    
    #Deteremines response in stimulated and unstimulated regions, max response vs distance to cluster/islet edge in stimulated regions, and
    #max response vs distance to stimulation edge in unstimulated regions for buffer stimulation (NADH)
    baf.stim_unstim_response (im_NADH_b_norm, stim, unstim, path2, islet_id[i])
    baf.islet_edge_stim_edge_vs_response(im_NADH_b_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path2, islet_id[i])

    
    #Deteremines response in stimulated and unstimulated regions, max response vs distance to cluster/islet edge in stimulated regions, and
    #max response vs distance to stimulation edge in unstimulated regions for glucose stimulation (Fluo4)
    baf.stim_unstim_response (im_Fluo4_g_norm, stim, unstim, path4, islet_id[i])
    baf.islet_edge_stim_edge_vs_response(im_Fluo4_g_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path4, islet_id[i])

    
    #Deteremines response in stimulated and unstimulated regions, max response vs distance to cluster/islet edge in stimulated regions, and
    #max response vs distance to stimulation edge in unstimulated regions for buffer stimulation (Fluo4)
    baf.stim_unstim_response (im_Fluo4_b_norm, stim, unstim, path5, islet_id[i])
    baf.islet_edge_stim_edge_vs_response(im_Fluo4_b_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim, path5, islet_id[i])


