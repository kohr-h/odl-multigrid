# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL-multigrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Numerical helpers related to multigrid applications."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from itertools import product
import numpy as np

from odl.discr.lp_discr import DiscreteLpElement
from odl.discr.partition import RectPartition
from odl.operator import Operator
from odl.util import writable_array, dtype_repr
from odl.util.numerics import apply_on_boundary


__all__ = ('reduce_over_partition',)


def _apply_reduction(arr, out, reduction, axes):
    try:
        reduction(arr, axis=axes, out=out)
    except TypeError:
        out[:] = reduction(arr, axis=axes)


def reduce_over_partition(discr_func, partition, reduction, pad_const=0,
                          out=None):
    """Reduce a discrete function blockwise over a coarser partition.

    This helper function is intended as a helper for multi-grid
    computations where a finely discretized function needs to undergo
    a blockwise reduction operation over a coarser partition of a
    containing spatial region. An example is to average the given
    function over larger blocks as defined by the partition.

    Parameters
    ----------
    discr_func : `DiscreteLpElement`
        Element in a uniformly discretized function space that is to be
        reduced over blocks defined by ``partition``.
    partition : uniform `RectPartition`
        Coarser partition than ``discr_func.space.partition`` that defines
        the large cells (blocks) over which ``discr_func`` is reduced.
        Its ``cell_sides`` must be an integer multiple of
        ``discr_func.space.cell_sides``.
    reduction : callable
        Reduction function defining the operation on each block of values
        in ``discr_func``. It needs to be callable as
        ``reduction(array, axes=my_axes)`` or
        ``reduction(array, axes=my_axes, out=out_array)``, where
        ``array, out_array`` are `numpy.ndarray`'s, and ``my_axes`` are
        sequence of ints specifying over which axes is being reduced.
        The typical examples are NumPy reductions like `np.sum` or `np.mean`,
        but custom functions are also possible.
    pad_const : scalar, optional
        This value is filled into the parts that are not covered by the
        function.
    out : `numpy.ndarray`, optional
        Array to which the output is written. It needs to have the same
        ``shape`` as ``partition`` and a ``dtype`` to which
        ``discr_func.dtype`` can be cast.

    Returns
    -------
    out : `numpy.ndarray`
        Array holding the result of the reduction operation. If ``out``
        was given, the returned object is a reference to it.

    Examples
    --------
    Consider a simple 1D example with 4 small cells per large cell,
    and summing a constant function over the large cells:

    >>> partition = odl.uniform_partition(0, 1, 5)
    >>> partition.cell_boundary_vecs
    (array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ]),)
    >>> space = odl.uniform_discr(0, 0.5, 10)  # 0.5 falls between
    >>> func = space.one()
    >>> reduce_over_partition(func, partition, reduction=np.sum)
    array([ 4.,  4.,  2.,  0.,  0.])
    >>> # The value 4 is due to summing 4 ones from 4 small cells per
    >>> # large cell.
    >>> # The "2" in the third cell is expected since it only counts half --
    >>> # the overlap of func.domain is only half a cell ([0.4, 0.5]).

    In 2D, everything (including partial overlap weighting) works per
    axis:

    >>> partition = odl.uniform_partition([0, 0], [1, 1], [5, 5])
    >>> space = odl.uniform_discr([0, 0], [0.5, 0.7], [10, 14])
    >>> func = space.one()
    >>> reduce_over_partition(func, partition, reduction=np.sum)
    array([[ 16.,  16.,  16.,   8.,   0.],
           [ 16.,  16.,  16.,   8.,   0.],
           [  8.,   8.,   8.,   4.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.]])
    >>> # 16 = sum of 16 ones from 4 x 4 small cells per large cell
    >>> # 8: cells have half weight due to half overlap
    >>> # 4: the corner cell overlaps half in both axes, i.e. 1/4 in
    >>> # total
    """
    if not isinstance(discr_func, DiscreteLpElement):
        raise TypeError('`discr_func` must be a `DiscreteLpElement` instance, '
                        'got {!r}'.format(discr_func))
    if not discr_func.space.is_uniform:
        raise ValueError('`discr_func.space` is not uniformly discretized')
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` must be a `RectPartition` instance, '
                        'got {!r}'.format(partition))
    if not partition.is_uniform:
        raise ValueError('`partition` is not uniform')

    # TODO: use different eps in each axis?
    dom_eps = 1e-8 * max(discr_func.space.partition.extent)
    if not partition.set.contains_set(discr_func.space.domain, atol=dom_eps):
        raise ValueError('`partition.set` {} does not contain '
                         '`discr_func.space.domain` {}'
                         ''.format(partition.set, discr_func.space.domain))

    if out is None:
        out = np.empty(partition.shape, dtype=discr_func.dtype,
                       order=discr_func.dtype)
    if not isinstance(out, np.ndarray):
        raise TypeError('`out` must be a `numpy.ndarray` instance, got '
                        '{!r}'.format(out))
    if not np.can_cast(discr_func.dtype, out.dtype):
        raise ValueError('cannot safely cast from `discr_func.dtype` {} '
                         'to `out.dtype` {}'
                         ''.format(dtype_repr(discr_func.dtype),
                                   dtype_repr(out.dtype)))
    if not np.array_equal(out.shape, partition.shape):
        raise ValueError('`out.shape` differs from `partition.shape` '
                         '({} != {})'.format(out.shape, partition.shape))
    if not np.can_cast(pad_const, out.dtype):
        raise ValueError('cannot safely cast `pad_const` {} '
                         'to `out.dtype` {}'
                         ''.format(pad_const, dtype_repr(out.dtype)))
    out.fill(pad_const)

    # Some abbreviations for easier notation
    # All variables starting with "s" refer to properties of
    # `discr_func.space`, whereas "p" quantities refer to the (coarse)
    # `partition`.
    spc = discr_func.space
    smin, smax = spc.min_pt, spc.max_pt
    scsides = spc.cell_sides
    part = partition
    pmin = part.min_pt, part.max_pt
    func_arr = discr_func.asarray()
    ndim = spc.ndim

    # Partition cell sides must be an integer multiple of space cell sides
    csides_ratio_f = part.cell_sides / spc.cell_sides
    csides_ratio = np.around(csides_ratio_f).astype(int)
    if not np.allclose(csides_ratio_f, csides_ratio):
        raise ValueError('`partition.cell_sides` is a non-integer multiple '
                         '({}) of `discr_func.space.cell_sides'
                         ''.format(csides_ratio_f))

    # Shift must be an integer multiple of space cell sides
    rel_shift_f = (smin - pmin) / scsides
    if not np.allclose(np.round(rel_shift_f), rel_shift_f):
        raise ValueError('shift between `partition` and `discr_func.space` '
                         'is a non-integer multiple ({}) of '
                         '`discr_func.space.cell_sides'
                         ''.format(rel_shift_f))

    # Calculate relative position of a number of interesting points

    # Positions of the space domain min and max vectors relative to the
    # partition
    cvecs = part.cell_boundary_vecs
    smin_idx = np.array(part.index(smin), ndmin=1)
    smin_partpt = np.array([cvec[si + 1] for si, cvec in zip(smin_idx, cvecs)])
    smax_idx = np.array(part.index(smax), ndmin=1)
    smax_partpt = np.array([cvec[si] for si, cvec in zip(smax_idx, cvecs)])

    # Inner part of the partition in the space domain, i.e. partition cells
    # that are completely contained in the spatial domain and do not touch
    # its boundary
    p_inner_slc = [slice(li + 1, ri) for li, ri in zip(smin_idx, smax_idx)]

    # Positions of the first and last partition points that still lie in
    # the spatial domain, relative to the space partition
    pl_idx = np.array(
        np.round(spc.index(smin_partpt, floating=True)).astype(int),
        ndmin=1)
    pr_idx = np.array(
        np.round(spc.index(smax_partpt, floating=True)).astype(int),
        ndmin=1)
    s_inner_slc = [slice(li, ri) for li, ri in zip(pl_idx, pr_idx)]

    # Slices to constrain to left and right boundary in each axis
    pl_slc = [slice(li, li + 1) for li in smin_idx]
    pr_slc = [slice(ri, ri + 1) for ri in smax_idx]

    # Slices for the overlapping space cells to the left and the right
    # (up to left index excl. / from right index incl.)
    sl_slc = [slice(None, li) for li in pl_idx]
    sr_slc = [slice(ri, None) for ri in pr_idx]

    # Shapes for reduction of the inner part by summing over axes.
    reduce_inner_shape = []
    reduce_axes = tuple(2 * i + 1 for i in range(ndim))
    inner_shape = func_arr[s_inner_slc].shape
    for n, k in zip(inner_shape, csides_ratio):
        reduce_inner_shape.extend([n // k, k])

    # Now we loop over boundary parts of all dimensions from 0 to ndim-1.
    # They are encoded as follows:
    # - We select inner (1) and outer (2) parts per axis by looping over
    #   `product([1, 2], repeat=ndim)`, using the name `parts`.
    # - Wherever there is a 2 in the sequence, 2 slices must be generated,
    #   one for left and one for right. The total number of slices is the
    #   product of the numbers in `parts`, i.e. `num_slcs = prod(parts)`.
    # - We get the indices of the 2's in the sequence and put them in
    #   `outer_indcs`.
    # - The "p" and "s" slice lists are initialized with the inner parts.
    #   We need `num_slcs` such lists for this particular sequence `parts`.
    # - Now we enumerate `outer_indcs` as `i, oi` and put into the
    #   (2*i)-th entry of the slice lists the "left" outer slice and into
    #   the (2*i+1)-th entry the "right" outer slice.
    #
    # The total number of slices to loop over is equal to
    # sum(k=0->ndim, binom(ndim, k) * 2^k) = 3^ndim.
    # This should not add too much computational overhead.
    for parts in product([1, 2], repeat=ndim):

        # Number of slices to consider
        num_slcs = np.prod(parts)

        # Indices where we need to consider the outer parts
        outer_indcs = tuple(np.where(np.equal(parts, 2))[0])

        # Initialize the "p" and "s" slice lists with the inner slices.
        # Each list contains `num_slcs` of those.
        p_slcs = [list(p_inner_slc) for _ in range(num_slcs)]
        s_slcs = [list(s_inner_slc) for _ in range(num_slcs)]
        # Put the left/right slice in the even/odd sublists at the
        # position indexed by the outer_indcs thing.
        # We also need to initialize the `reduce_shape`'s for all cases,
        # which has the value (n // k, k) for the "inner" axes and
        # (1, n) in the "outer" axes.
        reduce_shapes = [list(reduce_inner_shape) for _ in range(num_slcs)]
        for islc, bdry in enumerate(product('lr', repeat=len(outer_indcs))):
            for oi, l_or_r in zip(outer_indcs, bdry):
                if l_or_r == 'l':
                    p_slcs[islc][oi] = pl_slc[oi]
                    s_slcs[islc][oi] = sl_slc[oi]
                else:
                    p_slcs[islc][oi] = pr_slc[oi]
                    s_slcs[islc][oi] = sr_slc[oi]

            f_view = func_arr[s_slcs[islc]]
            for oi in outer_indcs:
                reduce_shapes[islc][2 * oi] = 1
                reduce_shapes[islc][2 * oi + 1] = f_view.shape[oi]

        # Compute the block reduction of all views represented by the current
        # `parts`. This is done by reshaping from the original shape to the
        # above calculated `reduce_shapes` and reducing over `reduce_axes`.
        for p_s, s_s, red_shp in zip(p_slcs, s_slcs, reduce_shapes):
            f_view = func_arr[s_s]
            out_view = out[p_s]

            if 0 not in f_view.shape:
                # View not empty, reduction makes sense
                _apply_reduction(arr=f_view.reshape(red_shp), out=out_view,
                                 axes=reduce_axes, reduction=reduction)
    return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
