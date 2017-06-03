# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL-multigrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators for multigrid applications."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.operator import Operator
from odl.util import writable_array
from odl.util.numerics import apply_on_boundary


__all__ = ('MaskingOperator',)


# TODO: this operator could be part of core ODL


class MaskingOperator(Operator):

    """An operator that masks a given spatial region.

    This operator sets the region between ``min_pt`` and ``max_pt`` to 0.
    The cut-off is "soft" in the sense that partially masked cells are
    weighted by the relative volume of the unmasked part.

    Notes
    -----
    A masking operator :math:`M` for a region-of-interest (ROI),
    applied to a function :math:`f`, returns the function :math:`M(f)`
    given by

    .. math::

        M(f)(x) =
        \\begin{cases}
            0    & \\text{if } x \in \\text{ROI} \\\\
            f(x) & \\text{otherwise.}
        \end{cases}
    """

    def __init__(self, space, min_pt, max_pt):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Domain of the operator, the space of functions to be masked.
        min_pt, max_pt:  float or sequence of floats
            Minimum/maximum corners of the masked region.

        Examples
        --------
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> space.partition.cell_boundary_vecs
        (array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ]),)

        If the masked region aligns with the cell boundaries, we make a
        "hard" cut-out:

        >>> min_pt = 0.2
        >>> max_pt = 0.6
        >>> mask_op = MaskingOperator(space, min_pt, max_pt)
        >>> masked_one = mask_op(space.one())
        >>> print(masked_one.asarray())
        [ 1.  0.  0.  1.  1.]

        Otherwise, the values linearly drop to 0:

        >>> min_pt = 0.3
        >>> max_pt = 0.75
        >>> mask_op = MaskingOperator(space, min_pt, max_pt)
        >>> masked_one = mask_op(space.one())
        >>> print(masked_one.asarray())
        [ 1.    0.5   0.    0.25  1.  ]
        """
        super().__init__(domain=space, range=space, linear=True)
        self.__min_pt = np.array(min_pt, ndmin=1)
        self.__max_pt = np.array(max_pt, ndmin=1)
        if self.min_pt.shape != (self.domain.ndim,):
            raise ValueError('`min_pt` shape not equal to `(ndim,)` '
                             '({} != ({},))'
                             ''.format(self.min_pt.shape, self.domain.ndim))
        if self.max_pt.shape != (self.domain.ndim,):
            raise ValueError('`max_pt` shape not equal to `(ndim,)` '
                             '({} != ({},))'
                             ''.format(self.max_pt.shape, self.domain.ndim))

    @property
    def min_pt(self):
        """Minimum coordinates of the masked region."""
        return self.__min_pt

    @property
    def max_pt(self):
        """Maximum coordinates of the masked region."""
        return self.__max_pt

    def _call(self, x, out):
        """Mask ``x`` and store the result in ``out`` if given."""
        # Find the indices of the mask min and max. The floating point
        # versions are also required for the linear transition.
        idx_min_flt = np.array(
            self.domain.partition.index(self.min_pt, floating=True),
            ndmin=1)
        idx_max_flt = np.array(
            self.domain.partition.index(self.max_pt, floating=True),
            ndmin=1)

        # To deal with coinciding boundaries we introduce an epsilon tolerance
        epsilon = 1e-6
        idx_min = np.floor(idx_min_flt - epsilon).astype(int)
        idx_max = np.ceil(idx_max_flt + epsilon).astype(int)

        def coeffs(d):
            return (1.0 - (idx_min_flt[d] - idx_min[d]),
                    1.0 - (idx_max[d] - idx_max_flt[d]))

        # Need an extra level of indirection in order to capture `d` inside
        # the lambda expressions
        def fn_pair(d):
            return (lambda x: x * coeffs(d)[0], lambda x: x * coeffs(d)[1])

        boundary_scale_fns = [fn_pair(d) for d in range(x.ndim)]

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        slc_inner = tuple(slice(imin + 1, imax - 1) for imin, imax in
                          zip(idx_min, idx_max))

        # Make a mask that is 1 outside the masking region, 0 inside
        # and has a linear transition where the region boundary does not
        # coincide with a cell boundary
        mask = np.ones_like(x)
        mask[slc_inner] = 0
        apply_on_boundary(mask[slc],
                          boundary_scale_fns,
                          only_once=False,
                          out=mask[slc])
        apply_on_boundary(mask[slc],
                          lambda x: 1.0 - x,
                          only_once=True,
                          out=mask[slc])

        # out = masked version of x
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc] = mask[slc] * out_arr[slc]

    @property
    def adjoint(self):
        """The (self-adjoint) masking operator."""
        return self


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
