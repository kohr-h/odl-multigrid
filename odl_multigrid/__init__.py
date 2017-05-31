# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL-multigrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""ODL multigrid plugin.

This plugin provides functionality to set up and solve problems with
multiple resolutions, using regular grids of varying sizes.
"""

from __future__ import absolute_import

__version__ = '0.1.0.dev0'
__all__ = (
    'space',
    'operators',
    'graphics',
    'phantom',
    'numerics',
)


# TODO:
# - Add a 'multi-grid' space, which can be used for reconstruction and so on
# - Add support multi-resolution phantoms
# - Define definitive API for multi-grid reconstruction


# The following subpackages are considered part of the "core", and the
# names defined their `__all__` variables are added to the top-level
# namespace
from .space import *
__all__ += space.__all__

from .operators import *
__all__ += operators.__all__

from . import graphics
from . import phantom
from . import numerics