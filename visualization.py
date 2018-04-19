#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:04:29 2016

Splitting up my microscopy image processing into multiple files by purpose.
Grabbing low hanging fruit fixes as I go. Or even adding very simple features
(like default parameter values).

This file is for the various plotting and graphing I do.

@author: Nicholas Sherer
"""


import ipywidgets as ipyw
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as colormap
import numpy as np
import skimage.morphology as skmo

from segmentation import warpIm2Im


def showImages(images, figsize=None):
    """
    Just a simple function to display a set of images side by side.
    """
    fig = plt.figure(figsize=figsize)
    image_num = len(images)
    if figsize is None:
        figsize = (image_num*6, 6)
    plts = []
    for index, image in enumerate(images):
        plts.append(fig.add_subplot(1, image_num, index+1))
        plts[index].imshow(image)


def inspectImages(image_lists, figsize=None):
    """
    This function is just a quick shortcut to making a slider for inspecting a
    group of related lists of images (such as TIRF and brightfield images of
    the same field of view). To make sense, the lists should contain fields of
    view in the same order from the same experiment.
    """
    sl_min = 0
    sl_max = len(image_lists[0])
    if figsize is None:
        figsize = (len(image_lists)*6, 6)

    def displayImages(image_num):
        images = []
        for image_list in image_lists:
            images.append(image_list[image_num])
        showImages(images, figsize)

    image_num = ipyw.IntSlider(value=sl_min, min=sl_min, max=sl_max-1,
                               continuous_update=False, description='image #')
    widget = ipyw.interactive(displayImages, image_num=image_num)
    return widget


def adjustAlignment(image_list, mask_list, trans):
    """
    This function makes the widget for hand tuning the alignment between
    cameras and returns the value of the hand tuned alignment.
    """

    my_cmap = colormap.binary
    my_cmap.set_over('w', alpha=0)

    s_0 = trans.scale
    s_err = .01*s_0
    s_st = .001*s_0
    theta_0 = trans.rotation
    theta_err = .005
    theta_st = .0001
    delta_x_0, delta_y_0 = trans.translation
    delta_err = 5
    delta_st = .1

    image_sl = ipyw.IntSlider(value=0, min=0, max=len(image_list)-1,
                              continuous_update=False, description='image #')
    dil_sl = ipyw.IntSlider(value=1, min=0, max=10, continuous_update=False,
                            description='dilation size')

    scale_slider = ipyw.FloatSlider(value=s_0, min=s_0 - s_err,
                                    max=s_0 + s_err, step=s_st,
                                    continuous_update=False,
                                    description='scale')
    theta_slider = ipyw.FloatSlider(value=theta_0, min=theta_0 - theta_err,
                                    max=theta_0 + theta_err, step=theta_st,
                                    continuous_update=False,
                                    description='rotation')
    delta_x_slider = ipyw.FloatSlider(value=delta_x_0,
                                      min=delta_x_0 - delta_err,
                                      max=delta_x_0 + delta_err,
                                      step=delta_st, continuous_update=False,
                                      description='vertical shift')
    delta_y_slider = ipyw.FloatSlider(value=delta_y_0,
                                      min=delta_y_0 - delta_err,
                                      max=delta_y_0 + delta_err,
                                      step=delta_st, continuous_update=False,
                                      description='horizontal shift')

    def changeTransform(transform, scale, theta, delta_x, delta_y):
        """
        This functions changes the augmented transformation matrix to the one
        given by the scaling, rotation, and translation parameters input
        """
        transform.params[0, 0] = scale*np.cos(theta)
        transform.params[1, 1] = scale*np.cos(theta)
        transform.params[0, 1] = scale*-np.sin(theta)
        transform.params[1, 0] = scale*np.sin(theta)
        transform.params[0, 2] = delta_x
        transform.params[1, 2] = delta_y

    def applyTransform(index, dil_size, scale, theta, delta_x, delta_y):
        changeTransform(trans, scale, theta, delta_x, delta_y)
        warp_mask = warpIm2Im(mask_list[index], image_list[index], trans)
        warp_mask = skmo.binary_dilation(warp_mask, selem=skmo.disk(dil_size))
        fig = plt.figure(figsize=(24, 16))
        img_view = fig.add_subplot(1, 2, 1)
        align_view = fig.add_subplot(1, 2, 2)
        img_view.imshow(image_list[index])
        align_view.imshow(image_list[index])
        align_view.imshow(warp_mask, cmap=my_cmap, clim=[0, .1])

    box1 = ipyw.Box()
    box1.children = [image_sl, dil_sl]
    box2 = ipyw.Box()
    box2.children = [scale_slider, theta_slider, delta_x_slider,
                     delta_y_slider]
    tabwidget = ipyw.Tab()
    tabwidget.children = [box1, box2]
    tabwidget.set_title(0, 'image # and dilation size')
    tabwidget.set_title(1, 'transformation parameters')
    ipyw.interactive(applyTransform, index=image_sl, dil_size=dil_sl,
                     scale=scale_slider, theta=theta_slider,
                     delta_x=delta_x_slider, delta_y=delta_y_slider)
    return tabwidget


def plotConnectingLine(fig, coord1, axes1, coord2, axes2):
    """
    Draws a line between points coord1 and coord2 in respective subplots axes1
    and axes2 of a figure fig
    """
    plt.draw()
    inverted = fig.transFigure.inverted()
    point1 = np.array([coord1[1], coord1[0]])
    coord1_fig = inverted.transform(axes1.transData.transform(point1))
    point2 = np.array([coord2[1], coord2[0]])
    coord2_fig = inverted.transform(axes2.transData.transform(point2))
    line = matplotlib.lines.Line2D((coord1_fig[0], coord2_fig[0]),
                                   (coord1_fig[1], coord2_fig[1]),
                                   transform=fig.transFigure)
    fig.lines.append(line)


def showKeypointpairs(image1, image2, keypoints1, keypoints2, figsize=(18, 9)):
    """
    This function takes two images and a set of keypoints for each image and
    shows the images side by side with the keypoints numbered. If the two
    images have the same number of keypoints, then lines are drawn connecting
    keypoints by the order they are given in (0th point in keypoints1 connected
    to the 0th point in keypoints2, 1st in keypoints1 to 1st in keypoints2,
    etc.)
    """
    # first set up the plots and axes and display the images
    image1_height, image1_width = image1.shape
    image2_height, image2_width = image2.shape

    fig = plt.figure(figsize=figsize)
    subplot1 = fig.add_subplot(121)
    subplot1.set_ylim([0, image1_height])
    subplot1.set_xlim([0, image1_width])
    subplot2 = fig.add_subplot(122)
    subplot2.set_ylim([0, image2_height])
    subplot2.set_xlim([0, image2_width])

    subplot1.imshow(image1)
    subplot2.imshow(image2)

    # next plot the keypoints for each image
    i = 0
    for point in keypoints1:
        subplot1.text(point[1], point[0], str(i))
        subplot1.plot(point[1], point[0], 'r*')
        i = i+1

    j = 0
    for point in keypoints2:
        subplot2.text(point[1], point[0], str(j))
        subplot2.plot(point[1], point[0], 'r*')
        j = j+1

    # if the number of keypoints match, draw lines between them
    if len(keypoints1) == len(keypoints2):
        pairs = zip(keypoints1, keypoints2)
        for pair in pairs:
            plotConnectingLine(fig, pair[0], subplot1, pair[1], subplot2)


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
