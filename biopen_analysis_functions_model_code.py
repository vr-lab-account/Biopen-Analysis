# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:11:54 2020

@author: Vishnu Rao
"""

import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
import pandas as pd
import imreg_dft as ird
from scipy import ndimage

def ba_align (im, top_border, bottom_border, left_border, right_border):
    #This function aligns the images within a timecourse using the imreg_dft algorithm. Images are aligned to the frame before biopen stimulation is begun.
    
    #Gets dimensions of timecourse image
    d,h,w = im.shape
    
    #Excludes non-islet fluorescence in the field of view so that only islets are aligned with each other
    align_islet_region = np.zeros((h,w))
    align_islet_region[top_border+30:bottom_border+30, left_border+30:right_border+30] = 1
    im_align_region = im * align_islet_region
    
    #Aligns each image in the timecourse with the frame right before stimulation 
    im_regmat = np.zeros(shape=(d,h,w), dtype=np.uint16)
    if d>1:
        for count in range(0,d):
            result = ird.translation(im_align_region[10], im_align_region[count])
            tvec = result["tvec"].round(4)
            timg = ird.transform_img(im[count], tvec=tvec)
            im_regmat[count] = timg
    
    #Returns the aligned timecourse
    return im_regmat


def isletregion_backsub (im, back_inten=1000, upper_limit = 16383):
    #This function subtracts the background intensity from images and determines the islet region using otsu thresholding
    
    #Gets dimensions of timecourse image
    d1,h1,w1 = im.shape
    
    #Initializes background array
    background = np.zeros((d1,h1,w1))
        
    #Calculates and subtracts out background based on a set intensity threshold provided by the user
    for a in range (0,d1):
        background[a,:,:] = np.mean(im[a][im[a]<back_inten])
    im_minus_back = im - background

    #Defines region of the image corresponding to the islet based on otsu thresholding. The frame before stimulation is used for thresholding.
    im_temp = im_minus_back[10]
    im_temp[im_temp > upper_limit] = 0
    thresh = threshold_otsu(im_temp)
    islet_region = (im_minus_back[10]>thresh).astype(int)
    
    #Returns background subtracted image and binary (0s and 1s) array of islet region
    return im_minus_back, islet_region;


def clusterregion_backsub (im, back_inten=1000, upper_limit = 16383):
    #This function subtracts the background intensity and determines the cluster region.
    
    #Gets dimensions of timecourse image
    d1,h1,w1 = im.shape
    
    #Initializes background array
    background = np.zeros((d1,h1,w1))
        
    #Calculates and subtracts out background based on a set intensity threshold provided by the user
    for a in range (0,d1):
        background[a,:,:] = np.mean(im[a][im[a]<back_inten])
    im_minus_back = im - background

    #Defines region of the image corresponding to the cluster based on a defined threshold. The frame before stimulation is used for thresholding.
    im_temp = im[10]
    im_temp[im_temp > upper_limit] = 0
    thresh = 800
    islet_region = (im_temp>thresh).astype(int)
    
    #Returns background subtracted cluster and binary (0s and 1s) array of islet region
    return im_minus_back, islet_region;


def smaller_isletregion (im_minus_back, islet_region):
    #***IMPORTANT***: For this function to work, the islet must be near the center of the image.
    #This function determines the top, bottom, left, and right borders of an islet. Uses these borders (creates a square) to set a more stringent boundary.
    #The stringent boundary exclude any other fluorescence that might be in the field of view.
    
    #Gets dimensions of timecourse image
    d1,h1,w1 = im_minus_back.shape
    
    #Removes background from image (i.e. sets background pixels to 0)
    im_noback = islet_region * im_minus_back

    #Determines more stringent islet borders to exclude other fluorescence in the field of view. Searches from middle and goes in all four directions
    #until a row/column of 150 middle zeros is hit.
    half_width = int(w1/2)
    half_height = int(h1/2)
    
    top_mat = (im_noback[10][:half_height, (half_width-150):(half_width+150)]) == 0
    bottom_mat = (im_noback[10][half_height:, (half_width-150):(half_width+150)]) == 0
    left_mat = (im_noback[10][(half_height-150):(half_height+150), :half_width]) == 0
    right_mat = (im_noback[10][(half_height-150):(half_height+150), half_width:]) == 0
    
    top_border = max(np.where(np.all(top_mat, axis = 1))[0])
    bottom_border = min(np.where(np.all(bottom_mat, axis = 1))[0]) + half_height
    left_border = max(np.where(np.all(left_mat, axis = 0))[0])
    right_border = min(np.where(np.all(right_mat, axis = 0))[0]) + half_width

    #Creates a more stringent region that the islet can be in based on determined borders
    smaller_islet_region = np.zeros((h1,w1))
    smaller_islet_region[top_border:bottom_border, left_border:right_border] = 1
    
    #Returns stringent islet region and islet borders (top and bottom borders are row positions, left and right borders are column positions)
    return smaller_islet_region, top_border, bottom_border, left_border, right_border;


def islet_diam (top_border, bottom_border, left_border, right_border, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the islet diameter in um
    
    #The islet diameter is averaged from height and width of the islet
    diameter = ((bottom_border-top_border) + (right_border-left_border))/2/2.7533
    diameter_df = pd.DataFrame(data={'Islet diameter' : [diameter]})
    
    #The diameter is exported to a .csv file
    diameter_df.to_csv(path+isletid+'_'+'diameter.csv')


def baseline_normalize (im_final):
    #This function normalizes images in the timecourse based on baseline intensities for each pixel.
    
    #The first five frames are tossed. Then baseline intensities are calculated for each pixel using timepoints 6-11 (frames 0-11 are before stimulation).
    #Values are normalized to baseline avoiding division by 0.
    baseline = im_final[5:11].sum(axis=0)/6
    im_norm = np.divide(im_final, baseline, out=np.zeros_like(im_final).astype(float), where=baseline!=0)
    
    #Returns normalized image
    return im_norm


def stim_unstim_regions (im_rhod, islet_region, smaller_islet_region, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the stimulated and unstimulated regions of islets based on rhodamine stimulation. The area and percent of the islet stimulated is also calculated.
    
    #Frames 1 and 15 are used for thresholding (Frame 15 is the first frame after biopen stimulation has ended)
    R1 = im_rhod[0]
    R15 = im_rhod[14]
    
    #Removes the background from images then calculates the average baseline intensity value of the islet in frame 1 (excluding any zeros)
    Rhod1 = R1 * islet_region * smaller_islet_region
    Rhod15 = R15 * islet_region * smaller_islet_region
    avgbaseline = np.average(Rhod1[Rhod1>0])
    
    #Calculates total number of pixels from frame 15 with a value greater than 0
    totalpix_stim = len(Rhod15[Rhod15>0])
    
    #Divides pixel values in frame 15 by pixel values in frame 1. Excludes pixels with a value of 0 so no division by zero occurs.
    Rhod_divide = np.divide(Rhod15, Rhod1, out=np.zeros_like(Rhod15).astype(float), where=Rhod1!=0)
    
    #Uses two thresholds to determine the pixels stimulated: three times the average baseline intensity and a three fold change in the intensity of an individual pixel.
    #The stimulated pixel array is smoothed using a median filter and the number of stimulated pixels is caclulated.
    stim = np.logical_and(Rhod_divide>3, Rhod15>(avgbaseline*3))
    stim_filter = ndimage.median_filter(stim.astype(np.uint8),3)
    stim_2 = (stim_filter == 1)
    stimpix = len(Rhod15[stim_2])

    #Calculates the percent of islet pixels stimulated and exports values to .csv
    pcentstim = stimpix/totalpix_stim
    stimvals_df = pd.DataFrame(data={'Stimulated pixels' : [stimpix], 'Total pixels' : [totalpix_stim], 'Percent stimulated' : [pcentstim]})
    stimvals_df.to_csv(path+isletid+'_'+'stimvals_smooth.csv')

    #True/false matrix of smoothed stimulated pixels is converted to an image and exported
    stimimg = (stim_2*islet_region*smaller_islet_region).astype(np.uint8)
    stimimg*=255
    io.imsave(path+isletid+'_'+'stimulated_smooth.tif', stimimg)
    
    #True/false matrix of unstimulated pixels is converted to an image and exported
    unstim = np.invert(stim) * islet_region * smaller_islet_region
    unstimimg = ndimage.median_filter(unstim.astype(np.uint8),3)
    unstimimg*=255
    io.imsave(path+isletid+'_'+'unstimulated_smooth.tif', unstimimg)
    
    #Returns matrices of stimulated and unstimulated regions
    return stim_2, unstim;


def glucose_gradient (im_rhod, islet_region, smaller_islet_region, stim, unstim, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the glucose gradient across the islet following rhodamine stimulation. Glucose concentration is linearly mapped to fluorescence intensity.
    
    #Frames 11 and 15 are used for setting bounds (Frame 11 is the last frame before biopen stimulation and frame 15 is the first frame after biopen stimulation has ended)
    R11 = im_rhod[10]
    R15 = im_rhod[14]
    
    #Removes the background from images
    Rhod11 = R11 * islet_region * smaller_islet_region
    Rhod15 = R15 * islet_region * smaller_islet_region
    
    #Subtracts pixel values in frame 15 (after stimulation)  by pixel values in frame 11 (before stimulation)
    Rhod_subtract = Rhod15-Rhod11
    Rhod_subtract[Rhod_subtract<0] = 0
    
    #Determines top 10% of pixels by intensity in stimulated regions and bottom 10% of pixels by intensity in unstimulated regions. This is used to set boundaries (2 mM and 20 mM)
    stim_pixels = len(stim[stim>0])
    unstim_pixels = len(unstim[unstim>0])
    
    stimulated = Rhod_subtract * stim
    unstimulated = Rhod_subtract * unstim
    
    stim_sort = np.sort((stimulated[stimulated>0]), axis = None)
    top_10 = int(stim_pixels * 0.1)
    top_10_avg = np.mean(stim_sort[(len(stim_sort)-top_10):len(stim_sort)])
    
    unstim_sort = np.sort((unstimulated[unstimulated>0]), axis = None)
    bottom_10 = int(unstim_pixels * 0.1)
    bottom_10_avg = np.mean(unstim_sort[:bottom_10])
    
    #Calculates slope and y-intercept from boundaries
    slope = (top_10_avg-bottom_10_avg)/(20-2)
    yint = bottom_10_avg-slope*2
    
    #Creates glucose gradient image
    gluc_grad = ((Rhod_subtract-yint)/slope) * islet_region * smaller_islet_region
    io.imsave(path+isletid+'_'+'glucose_gradient.tif', gluc_grad)
    
    return


def stim_unstim_response (im_norm, stim, unstim, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the average response in stimulated and unstimulated regions of the islet.
    
    #Gets dimensions of timecourse image
    d1,h1,w1 = im_norm.shape
    
    #Average responses in stimulated and unstimulated region are calculated excluding zeros
    unstim_nozeros = unstim*im_norm
    unstim_nozeros[unstim_nozeros == 0] = np.nan  
    unstim_reshape = np.reshape(unstim_nozeros, (d1, (h1*w1)))
    
    stim_nozeros = stim*im_norm
    stim_nozeros[stim_nozeros == 0] = np.nan
    stim_reshape = np.reshape(stim_nozeros, (d1, (h1*w1)))
    
    #Average responses over time are exported to a .csv file
    averageresponse_df = pd.DataFrame({"unstim": np.nanmean(unstim_reshape, axis = 1), "stim": np.nanmean(stim_reshape, axis = 1)})
    averageresponse_df.to_csv(path+isletid+'_'+'averageresponse.csv')
    
    return
    

def islet_edge_stimulation_edge (im_norm, stim, left_border, right_border, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the islet edge as well as the stimulation edge
    
    #Gets dimensions of stimulated pixels image
    h1,w1 = stim.shape
    
    #Stores row positions of the bottom half of the islet boundary. Also creates an islet edge array that will be exported later on.
    islet_edge_cols = range(left_border, right_border)
    im_tf = (im_norm[10][:, left_border:right_border]>0).astype(np.uint8)
    im_flip_df = pd.DataFrame(data = np.flip(im_tf,0))
    islet_edge_rows = im_flip_df.idxmax(axis = 0)
    islet_edge_rows = h1-1-islet_edge_rows
    islet_edge = np.zeros((h1,w1))
    
    #Assigns 1s to x,y (column, row) positions demarcating the islet edge. Rest of the array is 0s. 
    for i in range(0,len(islet_edge_rows)):
        islet_edge[islet_edge_rows[i],left_border+i] = 1
    
    #Bins every two rows and calculates sum (of two pixels). If sum equals 2 then two consective pixels in that column were stimulated.
    bins = 2
    stim_reshape = np.reshape(stim[:(h1//bins)*bins, :], (-1, bins, w1)).sum(axis=1)
    
    #Cycles through each column. Determines the first row of four consecutive (i.e. four consecutive rows within a column) stimulated pixels.
    #Stores row position in an array and also creates an array for the stimulation edge.
    stim_edge = np.zeros((h1,w1))
    stim_edge_rows = np.zeros((right_border-left_border))
    for i in range(left_border,right_border):
        row_edge_array = np.argwhere(stim_reshape[:,i] == 2)
        row_edge_array_short = row_edge_array[:len(row_edge_array)-1]
        row_edge_consec = row_edge_array_short[(row_edge_array[1:]-row_edge_array[:-1]) == 1]
        if len(row_edge_consec) > 0:
            stim_edge[min(row_edge_consec)*2,i] = 1
            stim_edge_rows[i-left_border] = min(row_edge_consec)*2 
        elif len(row_edge_array) > 0:
            stim_edge[max(row_edge_array)*2,i] = 1
            stim_edge_rows[i-left_border] = max(row_edge_array)*2
    
    #Determines column positions that match the row positions determined above
    stim_edge_cols = np.argwhere(stim_edge_rows > 0) + left_border
    stim_edge_rows_2 = stim_edge_rows[stim_edge_rows > 0]
    stim_edge_cols_2 = np.reshape(stim_edge_cols, (len(stim_edge_rows_2)))
    
    #Exports islet and stimulation edge images
    io.imsave(path+isletid+'_'+'islet_edge.tif', islet_edge.astype(np.uint8))
    io.imsave(path+isletid+'_'+'stim_edge.tif', stim_edge.astype(np.uint8))
    
    #Returns arrays of row and column positions of islet and stimulation edges
    return islet_edge_rows, islet_edge_cols, stim_edge_rows_2, stim_edge_cols_2


def glucose_at_stim_edge (gluc_grad, stim_edge, path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #Determines glucose concentration at stimulation edge
    glucose_at_stim_edge = np.mean(gluc_grad[stim_edge>0])
    
    #Glucose concentration is exported to .csv
    gluc_stim_edge_df = pd.DataFrame({"concentration": [glucose_at_stim_edge]})
    gluc_stim_edge_df.to_csv(path+isletid+'_'+'glucose_stim_edge.csv')
    
    return


def islet_edge_stim_edge_vs_response (im_norm, islet_edge_rows, islet_edge_cols, stim_edge_rows, stim_edge_cols, unstim, stim,
                                      path = 'C:/Users/fishn/OneDrive/Desktop/', isletid = 'D1I1'):
    #This function determines the minimum distance from each pixel in the islet to the islet edge and stimulation edge.
    #The max response for each pixel is calculated and mapped to the minimum distance. The function also calculated the time
    #to max response of pixels showing a response
    
    #Gets dimensions of timecourse image
    d1,h1,w1 = im_norm.shape

    #Creates arrays of row and column positions for each pixel
    dist_rows = np.zeros((h1,w1))
    np.transpose(dist_rows)[:] = range(0,h1)
    dist_cols = np.zeros((h1,w1))
    dist_cols[:] = range(0,w1)
    
    #Sets array depth to number of pixels in stimulation edge and islet edge
    d2 = len(stim_edge_rows)
    d3 = len(islet_edge_rows)

    #Creates two distance matrices, one for the stimulated edge and one for the islet edge
    dist_mat_stim_e = np.zeros((d2,h1,w1))    
    dist_mat_islet_e = np.zeros((d3,h1,w1))
    
    #Conversion factor for pixels to um
    pixels_per_um = 2.7533

    #Determines distance of every pixel to each point on stimulation edge, each frame denotes a point on the stimulation edge
    for i in range(0,d2):
        dist_mat_stim_e[i,:,:] = np.sqrt((dist_rows-stim_edge_rows[i])**2 + (dist_cols-stim_edge_cols[i])**2)  
    
    #Determines distance of every pixel to each point on islet edge, each frame denotes a point on the islet edge
    for i in range(0,d3):
        dist_mat_islet_e[i,:,:] = np.sqrt((dist_rows-islet_edge_rows[i])**2 + (dist_cols-islet_edge_cols[i])**2)

    #Takes the min among slices at each pixel position to determine the minimum distance to stimulation edge for each pixel
    dist_mat_shortest_se = np.amin(dist_mat_stim_e, axis = 0)    
    #Takes the min among slices at each pixel position to determine the minimum distance to islet edge for each pixel
    dist_mat_shortest_ie = np.amin(dist_mat_islet_e, axis = 0)
    
    
    #For unstimulated regions:
    #Maps distance to stimulation edge with max responses. Max responses are determined from all responses after stimulation. Distance is rounded to the nearest integer
    #(no decimal points). Each column contains all the pixels in a frame.
    dist_vs_resp_unstim = np.zeros((h1*w1,d1-13+1))
    dist_vs_resp_unstim[:, 0] = np.ravel(np.around(dist_mat_shortest_se * unstim / pixels_per_um))
    dist_vs_resp_unstim[:, 1:] = np.transpose(np.reshape(im_norm[13:, :, :]*unstim, (-1, h1*w1)))
    
    #Removes rows with all values less than one, i.e. pixels with no response
    resp_unstim = dist_vs_resp_unstim[:, 1:d1-13+1]
    resp_unstim = resp_unstim[~np.all(resp_unstim < .9, axis=1)]
    
    #Determines average time to max of unstimulated pixels
    time_max_unstim = np.mean(resp_unstim.argmax(axis = 1))+3
    
    #Removes zeros that refer to pixels not within unstimulated region
    dist_vs_resp_unstim = dist_vs_resp_unstim[~np.all(dist_vs_resp_unstim == 0, axis=1)]
    
    #Groups responses by distance and takes the mean response at each timepoint
    dist_vs_resp_unstim_df = pd.DataFrame(data=dist_vs_resp_unstim)
    dist_vs_resp_unstim_df.rename(columns={dist_vs_resp_unstim_df.columns[0]: "distance" }, inplace = True)
    d_vs_r_unstim_grouped = dist_vs_resp_unstim_df.groupby('distance', as_index=False).mean()
    
    #Determines the max response (of the means) for each distance and the time after stimulation where this occurs
    d_vs_r_unstim_np = d_vs_r_unstim_grouped.to_numpy()
    max_response_unstim = np.max(d_vs_r_unstim_np[:, 1:], axis = 1)
    max_response_time_unstim = d_vs_r_unstim_np[:, 1:].argmax(axis = 1)+3
    
    #Organizes distance vs max response values in a dataframe
    dist_vs_maxresp_unstim_df = pd.DataFrame({"distance": d_vs_r_unstim_np[:, 0], "max response": max_response_unstim, "time of max response": max_response_time_unstim})
    
    #Exports distance vs max response spreadsheet
    dist_vs_maxresp_unstim_df.to_csv(path+isletid+'_'+'dist_vs_maxresp_unstim.csv')
    d_vs_r_unstim_grouped.to_csv(path+isletid+'_'+'d_v_r_unstim_all.csv')
    
    
    #For stimulated regions:
    #Maps distance to islet edge with responses. Max responses are determined from all responses after stimulation. Distance is rounded to the nearest integer
    #(no decimal points). Each column contains all the pixels in a frame.
    dist_vs_resp_stim = np.zeros((h1*w1,d1-13+1))
    dist_vs_resp_stim[:, 0] = np.ravel(np.around(dist_mat_shortest_ie * stim / pixels_per_um))
    dist_vs_resp_stim[:, 1:] = np.transpose(np.reshape(im_norm[13:, :, :]*stim, (-1, h1*w1)))
    
    #Removes rows with all values less than one, i.e. pixels with no response
    resp_stim = dist_vs_resp_stim[:, 1:d1-13+1]
    resp_stim = resp_stim[~np.all(resp_stim < .9, axis=1)]
    
    #Determines average time to max of stimulated pixels
    time_max_stim = np.mean(resp_stim.argmax(axis = 1))+3
    
    #Removes zeros that refer to pixels not within stimulated region
    dist_vs_resp_stim = dist_vs_resp_stim[~np.all(dist_vs_resp_stim == 0, axis=1)]
    
    #Groups responses by distance and takes the mean response at each timepoint
    dist_vs_resp_stim_df = pd.DataFrame(data=dist_vs_resp_stim)
    dist_vs_resp_stim_df.rename(columns={dist_vs_resp_stim_df.columns[0]: "distance" }, inplace = True)
    d_vs_r_stim_grouped = dist_vs_resp_stim_df.groupby('distance', as_index=False).mean()
    
    #Determines the max response (of the means) for each distance and the time after stimulation where this occurs
    d_vs_r_stim_np = d_vs_r_stim_grouped.to_numpy()
    max_response_stim = np.max(d_vs_r_stim_np[:, 1:], axis = 1)
    max_response_time_stim = d_vs_r_stim_np[:, 1:].argmax(axis = 1)+3
    
    #Organizes distance vs max response values in a dataframe
    dist_vs_maxresp_stim_df = pd.DataFrame({"distance": d_vs_r_stim_np[:, 0], "max response": max_response_stim, "time of max response": max_response_time_stim})
    
    #Exports distance vs max response spreadsheet
    dist_vs_maxresp_stim_df.to_csv(path+isletid+'_'+'dist_vs_maxresp_stim.csv')
    d_vs_r_stim_grouped.to_csv(path+isletid+'_'+'d_v_r_stim_all.csv')
    
    #Exports average time to max for stimulated and unstimulated regions
    time_max_df = pd.DataFrame({"stim": [time_max_stim], "unstim": [time_max_unstim]})
    time_max_df.to_csv(path+isletid+'_'+'average_time_to_max.csv')
    