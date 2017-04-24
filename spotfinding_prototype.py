#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:14:45 2017

Some prototype functions for finding spots, setting thresholds for finding
spots manually, and visualization and widget controls to go with all this.

@author: Nicholas Sherer
"""

import numpy as np
from copy import deepcopy
import skimage.measure as skme
import skimage.morphology as skmo
from skimage._shared._warnings import expected_warnings

import ipywidgets as ipyw
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import segmentation as mseg


def spotfindingLabels(mask_list, target_FOV, camera_transform):
    '''
    Return dilated and filled in labeled masks warped to a particular FOV.
    '''
    with expected_warnings(['Only one label']):
        expanded_masks = [skmo.binary_dilation(mask, selem=skmo.disk(10))
                          for mask in mask_list]
        expanded_masks = [skmo.remove_small_holes(mask, min_size=2500)
                          for mask in expanded_masks]
        expanded_labels = [skme.label(mask) for mask in expanded_masks]
        expanded_labels_w = [mseg.warpIm2Im(label, target_FOV,
                                            camera_transform)
                             for label in expanded_labels]
        expanded_labels_w = [skmo.remove_small_objects(label, min_size=30)
                             for label in expanded_labels_w]
        return expanded_labels_w


def maskBboxesandCoordinates(label_list):
    '''
    Return a dictionary mapping unique objects in an image sequence to their
    FOV, the bbox that contains them in that FOV, and the coordinates they
    occupy.
    '''
    mask_rprops = [skme.regionprops(label) for label in label_list]
    mask_coords = mseg.properties2list(mask_rprops, ['coords', 'bbox'])
    return mask_coords


def intensityValuesRegion(image_list, mask_coords, index):
    '''
    Return sorted list of intensities of an object in an image sequence.
    '''
    FOV = mask_coords['FOV'][index]
    x_coords = mask_coords['coords'][index][:, 0]
    y_coords = mask_coords['coords'][index][:, 1]
    return np.sort(image_list[FOV][x_coords, y_coords], axis=None)


def regionView(label_list, image_list, mask_coords, index):
    '''
    Return a view of an object without the rest of the image.
    '''
    FOV = mask_coords['FOV'][index]
    label = mask_coords['label'][index]
    mask = label_list[FOV] == label
    region = mask*image_list[FOV]
    bbox = mask_coords['bbox'][index]
    min_row, min_col, max_row, max_col = bbox
    return region[min_row:max_row, min_col:max_col]


def regionImagesAndIntensities(label_list, pc_image_list, TIRF_image_list):
    '''
    Return a dictinary containing slices of phase contrast and TIRF images of
    connected blobs of ecoli and sorted array of the intensity values of the
    TIRF image.
    '''
    mask_coords = maskBboxesandCoordinates(label_list)
    region_num = len(mask_coords['label'])
    pc_regions = [regionView(label_list, pc_image_list, mask_coords, i)
                  for i in range(region_num)]
    TIRF_regions = [regionView(label_list, TIRF_image_list, mask_coords, i)
                    for i in range(region_num)]
    reg_intensities = [intensityValuesRegion(TIRF_image_list, mask_coords, i)
                       for i in range(region_num)]
    return_dict = {'pc': pc_regions, 'TIRF': TIRF_regions,
                   'intensities': reg_intensities}
    return return_dict


def showOverlay(image, overlay, subplot, cmap=colormap.bwr):
    '''
    Plot an overlay of a mask on top an image.
    '''
    subplot.imshow(image)
    my_cmap = cmap
    my_cmap.set_under('w', alpha=0)
    subplot.imshow(overlay, cmap=my_cmap, clim=[.9, 1])


def showInverseOverlay(image, overlay, subplot, cmap=colormap.binary):
    '''
    Plot the inverse of an overlay of a mask on top of an image.
    '''
    subplot.imshow(image)
    my_cmap = cmap
    my_cmap.set_over('w', alpha=0)
    subplot.imshow(overlay, cmap=my_cmap, clim=[0, .1])


def plotHistogramThreshold(data, threshold, subplot):
    '''
    Plot a histogram of some data and draw a vertical line separating the data
    in two.
    '''
    nbins = np.sqrt(data.size).astype('int')
    bin_size = ((np.max(data)-np.min(data))/nbins).astype('int')
    subplot.hist(data, bins=nbins)
    subplot.axvline(threshold, color='red')
    subplot.set_yscale('log')
    subplot.set_ylabel('frequency')
    subplot.set_label('data, binsize= {: d}'.format(bin_size))


def halfSampleMode(sorted_array):
    '''
    Estimate the mode of a continuous random variable from a sorted array of
    draws of that random variable.
    '''
    size = sorted_array.size
    if size == 1:
        return sorted_array
    if size == 2:
        return (sorted_array[0]+sorted_array[1])/2
    half = np.int(size / 2) + 1
    small_interval = sorted_array[half-1]-sorted_array[0]
    small_index = half-1
    for i in range(half, size-1):
        new_interval = sorted_array[i] - sorted_array[i-half+1]
        if new_interval < small_interval:
            small_interval = new_interval
            small_index = i
    return halfSampleMode(sorted_array[small_index-half+1:small_index+1])


class manualSpotThresholder(object):

    def __init__(self, pc_regions, TIRF_regions, intensities, fsize,
                 load_thresholds=None, load_overlays=None, start_region=0):
        self.pc_regions = pc_regions
        self.TIRF_regions = TIRF_regions
        self.intensities = intensities
        self.region_num = len(pc_regions)
        self.fsize = fsize
        self.max_intensity = np.max(np.array([np.max(ri)
                                    for ri in self.intensities]))
        self.modes = [halfSampleMode(intensity) for intensity in intensities]
        self.init_guess = [4*mode - 3*np.min(intensity) for mode, intensity in
                           zip(self.modes, self.intensities)]

        if load_thresholds is None:
            self.thresholds = deepcopy(self.init_guess)
        else:
            self.thresholds = load_thresholds

        if load_overlays is None:
            self.overlays = [np.zeros_like(region) for region in pc_regions]
        else:
            self.overlays = load_overlays

        self.r_slider = ipyw.IntSlider(min=0, max=self.region_num-1, step=1,
                                       value=start_region,
                                       continuous_update=False)
        self.t_slider = ipyw.IntSlider(min=0, max=self.max_intensity, step=200,
                                       value=self.thresholds[start_region],
                                       continuous_update=False)
        self.box = ipyw.Box()
        self.box.children = [self.r_slider, self.t_slider]
        self.updatePlots(self.r_slider.value, self.t_slider.value)

        def onRegionChange(region):
            old_t_value = self.t_slider.value
            self.t_slider.value = self.thresholds[region]
            if old_t_value == self.t_slider.value:
                self.updatePlots(region, old_t_value)

        def onThresholdChange(threshold):
            region = self.r_slider.value
            self.thresholds[region] = threshold
            self.overlays[region] = self.TIRF_regions[region] > threshold
            self.updatePlots(region, threshold)

        ipyw.interactive(onRegionChange, region=self.r_slider)
        ipyw.interactive(onThresholdChange, threshold=self.t_slider)

    def updatePlots(self, index, threshold):
        fig = plt.figure(figsize=self.fsize)
        pc_subplot = fig.add_subplot(131)
        TIRF_subplot = fig.add_subplot(132)
        hist_subplot = fig.add_subplot(133)
        showOverlay(self.pc_regions[index], self.overlays[index], pc_subplot)
        TIRF_subplot.imshow(self.TIRF_regions[index])
        plotHistogramThreshold(self.intensities[index], threshold,
                               hist_subplot)
        hist_subplot.vlines(self.modes[index], 0, self.intensities[index].size,
                            color='green')
