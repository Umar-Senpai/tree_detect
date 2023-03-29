#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

'''
The present set of methods allow the computation of the angle and distance from a given origin 
to a line defined by a set of points
'''
xd = np.array([[1, 0, 0]])
yd = np.array([[0, 1, 0]])
zd = np.array([[0, 0, 1]])

def compute_direction(points):
    mean = np.mean(points, axis=0)

    subs = points - mean
    _, _, v = np.linalg.svd(subs)

    return v[[0], :]

def compute_line(points):
    d = compute_direction(points)
    return points[[0], :].T, d.reshape(3, 1)

def compute_projected_point(p0, d):
    t_projected = (-(p0.T)@d/(np.linalg.norm(d, ord=2) ** 2))
    return (p0 + t_projected*d).T

def compute_measurements(projected_point, d):

    distance = np.linalg.norm(projected_point)
    # modify accroding to how the camera is positiooned
    # The one used for the measurement model considers the ZX plane for
    # the given measurements
    angle =  np.arctan2(-projected_point[0, 0], projected_point[0, 2]) 

    # Line deformation that can be used to discard noisy measurements
    angle_x = np.arccos(xd@d/np.linalg.norm(d)*np.linalg.norm(xd))
    angle_y = np.arccos(yd@d/np.linalg.norm(d)*np.linalg.norm(yd))
    angle_z = np.arccos(zd@d/np.linalg.norm(d)*np.linalg.norm(zd))

    angle_x_2 = np.pi - np.abs(angle_x) 

    angle_y_2 =  np.pi - np.abs(angle_y)

    angle_z_2 = np.pi - np.abs(angle_z)

    if angle_x > angle_x_2:
        angle_x = angle_x_2
    if angle_y > angle_y_2:
        angle_y = angle_y_2
    if angle_z > angle_z_2:
        angle_z = angle_z_2

    return distance, angle, angle_x, angle_y, angle_z
