#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:01:32 2018

@author: kuhlmanlab
"""

import numpy as np


def translate(x_0, y_0, z_0, x, y, z):
    '''
    Translate points (x, y, z) by (x_0, y_0, z_0).
    '''
    return x - x_0, y - y_0, z - z_0


def rotate(u_x, u_y, theta, x, y, z):
    '''
    Rotate points (x, y, z) around axis with unit vector (u_x, u_y, u_z) by
    angle theta (counterclockwise, unit vector always points to positive z).
    '''
    u_z = np.sqrt(1 - u_x**2 - u_y**2)
    u_vector = np.array([u_x, u_y, u_z])
    u_vector_t = np.transpose(np.atleast_2d(u_vector))
    R = np.cos(theta)*np.identity(3) + \
        (1 - np.cos(theta))*u_vector*u_vector_t + \
        np.sin(theta) * np.array([[0, -u_z, u_y],
                                  [u_z, 0, -u_x],
                                  [-u_y, u_x, 0]])
    return [np.squeeze(x) for x in
            np.vsplit(np.matmul(R, np.vstack((x, y, z))), 3)]


class laser(object):
    '''
    A laser is described by a function which given a position returns the laser
    intensity at that position. Electromagnetic phase is ignored.
    '''

    def __init__(self, field_intensity):
        self.field_intensity = field_intensity

    def intensity_at(self, cell):
        return self.field_intensity(*cell.position)

    @staticmethod
    def _gaussian_intensity(I_0, w_0, z_R, x, y, z):
        '''
        Intensity profile of a gaussian laser beam. 00 TEM mode.
        '''
        w_z = 1 + (z / z_R)**2
        r_2 = x**2 + y**2
        return I_0 / w_z * np.exp(-2*r_2 / (w_0**2 * w_z))

    @classmethod
    def gaussian_beam_init(cls, I_0, w_0, z_R, u_x, u_y, theta, x_0, y_0, z_0):
        '''
        Create a laser beam with a gaussian intensity profile. Center of the
        beam is x_0, y_0, z_0, and it's rotated from the z-axis around the
        (u_x, u_y, u_z) axis by angle theta.
        '''
        def fixed_beam(x, y, z):
            return cls._gaussian_intensity(I_0, w_0, z_R,
                                           *rotate(u_x, u_y, theta,
                                                   *translate(x_0, y_0, z_0,
                                                              x, y, z)))
        return cls(fixed_beam)

    def intensity_plane(self, x_0, x_f, y_0, y_f, z):
        '''
        Return a 2-d slice of laser intensity.
        '''
        xx, yy = np.meshgrid(np.arange(x_0, x_f), np.arange(y_0, y_f))
        if callable(z):
            zz = z(xx, yy)
        else:
            zz = z*np.ones_like(xx)
        image = self.field_intensity(xx.flatten(), yy.flatten(), zz.flatten())
        image = image.reshape((y_f-y_0, x_f-x_0))
        return image


class camera(object):
    '''
    Camera object holding parameters of imaging chip and optics.
    '''

    def __init__(self, efficiency, exposure_time, signal_per_photon,
                 saturation_level, circuit_noise):
        self.efficiency = efficiency
        self.exposure_time = exposure_time
        self.signal_per_photon = signal_per_photon
        self.saturation_level = saturation_level
        self.circuit_noise = circuit_noise

    def image_cell(self, cell, laser_intensity, duration):
        '''
        Given a cell, laser_intensity, and duration of exposure return camera
        signal (stochastic).
        '''
        total_photons = cell.total_signal(laser_intensity, duration)
        photons = np.random.binomial(total_photons,
                                     min(self.exposure_time/duration,
                                         1))
        camera_signal = self.efficiency * self.signal_per_photon * photons + \
            np.random.poisson(self.circuit_noise)
        return min(self.saturation_level, camera_signal)


def background(object):
    '''
    Placeholder. Just has a constant autofluorescence.
    '''

    def __init__(self, autofluorescence):
        self.autofluorescence = autofluorescence

    def total_signal(self, laser_intensity, duration, area):
        return np.random.poisson(self.autofluorescence * area *
                                 laser_intensity * duration)


class cell(object):
    '''
    Represents a cell full of fluorescent proteins.

    Cell position requires all 3 coordinates. Not just 2 because slide may be
    slightly tilted etc.
    '''

    def __init__(self, position, area, autofluorescence,
                 fluorescent_proteins):
        self.position = position
        self.area = area
        self.autofluorescence = autofluorescence
        self.fluorescent_proteins = fluorescent_proteins

    def protein_signal(self, laser_intensity, duration):
        '''
        Emit photons from all proteins in cell with laser and aggregate their
        signal.
        '''
        photons = 0
        for protein in self.fluorescent_proteins:
            photons = photons + protein.emit(laser_intensity, duration)
        return photons

    def autofluorescence_signal(self, laser_intensity, duration):
        '''
        Emit photons due to autofluorescence of cell.
        '''
        return np.random.poisson(self.area * self.autofluorescence *
                                 laser_intensity * duration)

    def total_signal(self, laser_intensity, duration):
        '''
        Emit total signal of cell.
        '''
        return self.protein_signal(laser_intensity, duration) + \
            self.autofluorescence_signal(laser_intensity, duration)


class fluorescent_protein(object):
    '''
    Fluorescent proteins described by their brightness, stability, and a piece
    of state of whether or not they've photobleached.
    '''

    def __init__(self, photons_per_time_per_intensity,
                 bleaching_per_time_per_intensity, is_bleached=False):
        self.brightness = photons_per_time_per_intensity
        self.bleaching_rate = bleaching_per_time_per_intensity
        self.is_bleached = is_bleached

    def emit(self, laser_intensity, duration):
        '''
        Emit a stochastic number of photons depending on laser intensity and
        duration of exposure. The protein may irreversibly bleach leaving it
        unable to fluoresce.
        '''
        if self.is_bleached is False:
            bleach_time = np.random.exponential(self.bleaching_rate)
            if bleach_time < duration:
                emission_time = bleach_time
                self.is_bleached = True
            else:
                emission_time = duration
            return np.random.poisson(self.brightness * emission_time *
                                     laser_intensity)
        else:
            return 0


def photobleach_curve(laser, camera, cell, frame_count):
    '''
    Given a laser, camera, and cell generate a photobleaching curve for
    frame_count frames. You can't run this repeatedly on the same cell because
    the proteins will all be photobleached.
    '''
    intensity = []
    for i in range(frame_count):
        intensity.append(camera.image_cell(cell, laser.intensity_at(cell),
                                           camera.exposure_time))
    return np.array(intensity)
