#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:57:47 2016

Splitting up my microscopy image processing into multiple files by purpose.
Grabbing low hanging fruit fixes as I go. Or even adding very simple features
(like default parameter values).

This file is for the image segmentation routines. Thresholding, normalizing,
aligning, labeling regions, etc.

@author: Nicholas Sherer
"""

import numpy as np
# import skimage.transform as sktr
# import skimage.restoration as skre
# import skimage.draw as skdr
import skimage.feature as skfe
import skimage.filters as skf
import skimage.measure as skme
import skimage.morphology as skmo
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as ndint
import scipy.signal as spsig
# from functools import partial
from skimage import img_as_ubyte
from cv2 import fastNlMeansDenoising as nlMnsDns
from skimage._shared._warnings import expected_warnings


__version__ = .1

def findRegionCenters(mask, min_size=100, min_separation=10,
                      min_dist_fr_bg=10, border_size=5):
    """
    Return points in a binary image at local maximum distance from background.

    Foreground regions and holes in the foreground less than the min_size are
    ignored. Currently the default parameters are tuned for 5 micron beads at
    100x magnification.

    Parameters
    ----------
    mask : ndarray of bool_like
        Binary input image.
    min_size: integer
        Regions in mask smaller than min_size pixels will be ignored when
        finding the centers of regions.
    min_separation: integer
        The minimum separation in pixels between two local maximum. If maxima
        are closer than this only one of them will be returned.
    min_dist_fr_bg: integer
        The minimum distance in pixels from the background a local maximum must
        be for it to be returned as a region center.
    border_size: integer
        The minimum distance in pixels from the boundary a local maximum must
        be for it to be returned as a center.

    Returns
    -------
    centers : ndarray
        (row, column) coordinates of peaks

    Notes
    -----
    This functions optional inputs are mostly those for the skimage.feature
    corner_peaks function and are just passed into it. See its documentation
    for more details.
    """
    clean_mask = skmo.remove_small_holes(mask, min_size=min_size)
    clean_mask = skmo.remove_small_objects(clean_mask, min_size=min_size)
    mask_edt = ndi.distance_transform_edt(clean_mask)
    centers = skfe.corner_peaks(mask_edt, min_distance=min_separation,
                                threshold_abs=min_dist_fr_bg,
                                exclude_border=border_size)
    return centers


def warpIm2Im(fr_image, to_image, affine_transform):
    """
    The affine transform is the one you get by calling the estimate
    method of the affine transform object of skimage.transform on
    points going from the to_im to the fr_im. In other words, it is
    the inverse of the matrix transformation that maps coordinates
    in the fr_im to the to_im which is what the function is doing.
    Don't ask why scipy.ndimage works that way it just does.
    """
    size = np.shape(to_image)
    scaling_and_rotation = affine_transform.params[0:2, 0:2]
    translation = affine_transform.params[0:2, 2]
    # The order of interpolation for the warp is 0 because we're typically
    # warping either masks of integers or bools so spline interpolation is
    # bad.
    warp = ndint.affine_transform(fr_image, scaling_and_rotation, translation,
                                  size, order=0)
    return warp


def interpNans(im):
    """
    This function interpolates over any NANs in an image. It does
    this in place in memory, which is why it does not return a value.
    """
    mask = np.isnan(im)
    im[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                         im[~mask])


def normAndDenoisePc(image_list):
    """
    This corrects for unevenness in illumination by summing
    across a collection to get a blurry idea of the illumination
    and then dividing each image by that to correct for
    the uneven illumination. Then it denoises each image (meant
    for phase contrast images).
    """
    image_array = np.array(image_list)
    mean_illumination = np.mean(image_array, 0)
    normed_image_list = [image/mean_illumination for image in image_list]
    with expected_warnings(['precision']):
        image_ubyte_list = [img_as_ubyte(image/np.max(image)) for image in
                            normed_image_list]
    denoised_image_list = \
        [nlMnsDns(image, None, np.uint8(.95*np.std(image)), 7, 11) for image in
         image_ubyte_list]
    return denoised_image_list


def localMinLeftOfGlobalMax(image, bins, comparison_width):
    """
    This function finds a cutoff between pixels for e coli
    in a histogram and pixels of background. It does this by
    assuming the background is taking up most of the field of view
    and taking the maximum of the histogram to represent the mean
    of the background (roughly). Then it finds the closest local
    minimum below this maximum and uses it as a cutoff. This tactic seems to
    work well for e.coli in phase contrast or for finding the dark halos around
    fluorescent beads in brightfield.
    """
    bins, edges = np.histogram(image, bins)
    peak_index = spsig.argrelextrema(bins, np.greater_equal,
                                     order=comparison_width)[0]
    trough_index = spsig.argrelextrema(bins, np.less_equal,
                                       order=comparison_width)[0]
    peak_value = bins[peak_index]
    bg_peak_index = \
        peak_index[np.where(peak_value == np.max(peak_value))][0]
    trough_left_bg = trough_index[trough_index < bg_peak_index]
    try:
        low_threshold_index = trough_left_bg[-1]
    except IndexError:
        low_threshold_index = 0
        # if there is no min left of max everything is below
    low_threshold_location = edges[low_threshold_index]
    return low_threshold_location


def thresholdMask(image, bins=np.arange(256), comparison_width=5,
                  min_size=200):
    """
    Thresholds an image using localMinLeftofGlobalMax and returns the mask that
    results from this thresholding.
    """
    threshold = localMinLeftOfGlobalMax(image, bins, comparison_width)
    mask = image < threshold
    clean_mask = skmo.remove_small_objects(mask, min_size=min_size)
    return clean_mask


def findMedianBg(image_list, sigma=40):
    """
    This function calculates a background illumination by taking the median by
    pixel across a stack of images and then blurring the result. It
    returns this background normalized to have a mean intensity of 1 so that it
    won't affect the overall scale of the images to be illumination corrected,
    but will correct for inhomogeneities in illumination.
    """
    background = skf.gaussian(np.median(np.array(image_list), 0), sigma=sigma)
    normalized_background = background / np.mean(background)
    return normalized_background


def properties2list(regionprops_list_list, fields):
    """
    This function takes a list of regionprops listed by FOV and returns
    the attributes in a dictionary with keys of the attribute name whose fields
    are lists or numpy arrays depending on whether the attribute is a scalar.
    """
    return_dict = {}
    return_dict['FOV'] = []
    return_dict['label'] = []
    for field in fields:
        return_dict[field] = []
    FOV = 0
    for regionprops_list in regionprops_list_list:
        for regionprops in regionprops_list:
            return_dict['FOV'].append(FOV)
            return_dict['label'].append(regionprops['label'])
            for field in fields:
                try:
                    return_dict[field].append(getattr(regionprops, field))
                except AttributeError:
                    raise
        FOV = FOV + 1
    supported_types = set(np.typeDict.values())
    supported_types.update([float, int, complex])
    supported_types.discard(np.object_)
    supported_types.discard(np.void)
    for key in return_dict:
        if (type(return_dict[key][0]) in supported_types):
            return_dict[key] = np.array(return_dict[key])
    return return_dict


def medianAbsDev(array):
    '''This function returns the median absolute deviation of a numpy array.'''
    return np.median(np.abs(array-np.median(array)))


def aboveNMADselect(array, n):
    '''This function returns a boolean array with True at any positions in
    array that are more than n times the median absolute deviation from the
    median.'''
    mad_array = medianAbsDev(array)
    return (np.abs(array - np.median(array)) > n*mad_array)


def removeNonCirles(masks, n=np.inf, eccentricity_c=.6, solidity_c=.95):
    '''This function takes in a collection of image masks and discards all
    regions that aren't sufficiently circular i.e. are too eccentric and not
    solid enough. The parameter n can be set to discard circles whose areas and
    perimeters are n times the median absolute deviation from the median area
    and perimeter of the regions.'''
    labels = [skme.label(mask) for mask in masks]
    rprops = [skme.regionprops(label) for label in labels]
    prop_list = properties2list(rprops, ['eccentricity', 'solidity', 'area',
                                         'perimeter'])
    eccentricity_criteria = prop_list['eccentricity'] > eccentricity_c
    solidity_criteria = prop_list['solidity'] < solidity_c
    area_criteria = aboveNMADselect(prop_list['area'], n)
    perimeter_criteria = aboveNMADselect(prop_list['perimeter'], n)
    rejects = (eccentricity_criteria + solidity_criteria + area_criteria +
               perimeter_criteria) >= 1
    FOV_rejects = prop_list['FOV'][rejects]
    label_rejects = prop_list['label'][rejects]
    for FOV, label in zip(FOV_rejects, label_rejects):
        labels[FOV][labels[FOV] == label] = 0
    return [label > 0 for label in labels]


def findBeadsBF(image, thr):
    '''
    Find beads in brightfield by thresholding for the dark rings around them.
    '''
    outline = image < thr
    clean = skmo.remove_small_objects(outline, min_size=64)
    filled = skmo.remove_small_holes(clean, min_size=10000)
    bead_mask = np.logical_xor(clean, filled)
    bead_mask_clean = skmo.remove_small_objects(bead_mask, min_size=300)
    return bead_mask_clean
