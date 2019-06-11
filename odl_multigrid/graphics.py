# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL-multigrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tools for graphical display."""

import numpy as np


__all__ = ('show_extent', 'show_both')


# TODO: better names


def extent(angle, corners, detector_pos):
    """
    Compute the detector extent of a masking region for
    a given angle and detector position, for parallel 2d
    """
    angle = np.pi - angle

    regime = (int)(angle / (0.5 * np.pi))

    left_corner = corners[regime]
    right_corner = corners[(regime + 2) % 4]

    def proj_location(x, d, theta):
        return np.dot([np.sin(theta), -np.cos(theta)], x - d(theta))

    return [proj_location(left_corner, detector_pos, angle),
            proj_location(right_corner, detector_pos, angle)]


def show_extent(data, min_pt, max_pt, detector_pos):
    """Show the sinogram data along with the mask extent."""
    # Lazy import, can be slow
    import matplotlib.pyplot as plt

    corners = [[min_pt[0], max_pt[1]],
               [min_pt[0], min_pt[1]],
               [max_pt[0], min_pt[1]],
               [max_pt[0], max_pt[1]]]

    fig, ax = plt.subplots()

    xrange = [data.space.min_pt[0], data.space.max_pt[0]]
    yrange = [data.space.min_pt[1], data.space.max_pt[1]]

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.imshow(np.rot90(data), extent=[*xrange, *yrange], cmap='bone')
    ax.set_aspect('auto', 'box')

    # TODO: generalize
    thetas = np.linspace(data.space.min_pt[0], data.space.max_pt[0],
                         data.shape[0], endpoint=False)

    alpha = [extent(theta, corners, detector_pos) for theta in thetas]
    ax.plot(thetas, alpha, linewidth=2.0)


def show_both(coarse_data, fine_data):
    """Show the coarse and fine reconstruction in a single image."""
    # Lazy import, can be slow
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.image import BboxImage
    from matplotlib.transforms import Bbox, TransformedBbox

    fig, ax = plt.subplots()

    low = min([np.min(coarse_data), np.min(fine_data)])
    high = max([np.max(coarse_data), np.max(fine_data)])

    normalization = mpl.colors.Normalize(vmin=low, vmax=high)

    ax.set_xlim(coarse_data.space.min_pt[0], coarse_data.space.max_pt[0])
    ax.set_ylim(coarse_data.space.min_pt[1], coarse_data.space.max_pt[1])

    def show(data, eps=0.0):
        # Make box slightly larger
        box_extent = data.space.partition.extent * (1.0 + eps)
        box_min = data.space.min_pt - data.space.partition.extent * eps / 2.0

        bbox0 = Bbox.from_bounds(*box_min, *box_extent)
        bbox = TransformedBbox(bbox0, ax.transData)
        # TODO: adapt interpolation
        bbox_image = BboxImage(bbox, norm=normalization, cmap='bone',
                               interpolation='nearest', origin=False)
        bbox_image.set_data(np.asarray(data).T)
        ax.add_artist(bbox_image)

    show(coarse_data)
    show(fine_data, eps=0.01)

    # TODO: set aspect from physical sizes
    ax.set_aspect('auto', 'box')

def show_all(data, low=None, high=None):
    """Show all reconstructed resolutions in a single image. For multiple ROIs
    
    	data: list of reconstructions, with background given as data[0] and
              subsequent ROI reconstructions given as data[1], data[2], ...
 
	low: lower bound of dynamic range. If 'None' set to min(data)

	high: upper bound of dynamic range. If 'None' set to max(data)
    """
    # Lazy import, can be slow
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.image import BboxImage
    from matplotlib.transforms import Bbox, TransformedBbox

    fig, ax = plt.subplots()
    
    ax.set_xlim(data[0].space.min_pt[0], data[0].space.max_pt[0])
    ax.set_ylim(data[0].space.min_pt[1], data[0].space.max_pt[1])

    if low is None:
        low = min([np.min(data[0]), np.min(data[1])])
    if high is None:
        high = max([np.max(data[0]), np.max(data[1])])

    normalization = mpl.colors.Normalize(vmin=low, vmax=high)

    def show(data, eps=0.0):
        # Make box slightly larger
        box_extent = data.space.partition.extent * (1.0 + eps)
        box_min = data.space.min_pt - data.space.partition.extent * eps / 2.0

        bbox0 = Bbox.from_bounds(*box_min, *box_extent)
        bbox = TransformedBbox(bbox0, ax.transData)
        # TODO: adapt interpolation
        bbox_image = BboxImage(bbox, norm=normalization, cmap='bone',
                               interpolation='nearest', origin=False)
        bbox_image.set_data(np.asarray(data).T)
        ax.add_artist(bbox_image)

    for i in range(data.shape[0]):
        if i == 0:
            show(data[0])
        else:
            show(data[i], eps=0.01)

    # TODO: set aspect from physical sizes
    ax.set_aspect('auto', 'box')

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
