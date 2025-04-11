# This file is part of meas_extensions_piff.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ("LSSTPixelGrid",)

from piff import PixelGrid
import galsim
import numpy as np


# This is a specialized lsst subclass for performance reasons
# First, it allows caching and setting a value for maxk that
# is used when making galsim InterpolatedImages. Calculating
# this each time a psf is rendered is very expensive, and the
# value should not change much for a given psf solution.
#
# Secondly, the normalize function modifies a star fit paramters
# in place instead of creating a new copy with modified paramers.
# This is safe because of the way we use it, we expliclitly create
# a new star object to generate an image, and so we know no other
# code has a reference to this star.
class LSSTPixelGrid(PixelGrid):
    _type_name = "LSSTPixelGrid"

    def __init__(self, scale, size, interp=None, centered=True, init=None, fit_flux=False, logger=None):
        super().__init__(scale, size, interp, centered, init, fit_flux, logger)
        self._force_maxk = 0.0

    def setMaxK(self, maxk):
        """Sets the max spatial K used when generating profiles

        parameters
        ----------
        maxK : `float`
            The maximum K used when making profiles
        """
        self._force_maxk = maxk

    def getProfile(self, params):
        im = galsim.Image(params.reshape(self.size, self.size), scale=self.scale)
        flux = None if self._fit_flux else 1.0
        return galsim.InterpolatedImage(
            im, x_interpolant=self.interp, use_true_center=False, flux=flux, _force_maxk=self._force_maxk
        )

    def normalize(self, star):
        """Make sure star.fit.params are normalized properly.

        Note: This modifies the input star in place.
        """
        # Backwards compatibility check.
        # We used to only keep nparams - 1 or nparams - 3 values in fit.params.
        # If this is the case, fix it up to match up with our new convention.
        params = star.fit.get_params(self._num)
        nparams1 = len(params)
        nparams2 = self.size**2
        if nparams1 < nparams2:
            # Difference is either 1 or 3.  If not, something weird happened.
            assert nparams2 - nparams1 in [1, 3]

            # First copy over the parameters into the full array
            temp = np.zeros((self.size, self.size))
            mask = np.ones((self.size, self.size), dtype=bool)
            origin = (self.size // 2, self.size // 2)
            mask[origin] = False
            if nparams2 == nparams1 + 3:  # pragma: no branch
                # Note: the only regression we test is with centroids free,
                # so we always hit this branch.
                mask[origin[0] + 1, origin[1]] = False
                mask[origin[0], origin[1] + 1] = False
            temp[mask] = params

            # Now populate the masked pixels
            delta_u = np.arange(-origin[0], self.size - origin[0])
            delta_v = np.arange(-origin[1], self.size - origin[1])
            u, v = np.meshgrid(delta_u, delta_v)
            if nparams2 == nparams1 + 3:  # pragma: no branch
                # Do off-origin pixels first so that the centroid is 0,0.
                temp[origin[0] + 1, origin[1]] = -np.sum(v * temp)
                temp[origin[0], origin[1] + 1] = -np.sum(u * temp)

            # Now the center from the total flux == 1
            # Note: This uses the old scheme of sb normalization, not flux
            # normalization.
            temp[origin] = 1.0 / self.pixel_area - np.sum(temp)

            params = temp.flatten()
            star.fit.params = None  # Remove the old one with the wrong size,
            # so newParams doesn't complain about the size changing.

        # Normally this is all that is required.
        if not self._fit_flux:
            params /= np.sum(params)

            if self._num is not None:
                new_params = star.fit.params
                new_params[self._num] = params
            else:
                star.fit.params = params
