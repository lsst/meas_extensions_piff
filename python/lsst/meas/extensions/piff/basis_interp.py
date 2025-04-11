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

__all__ = ("LSSTBasisPolynomial",)

from piff import BasisPolynomial
import numpy as np


# This class is a subclass from PIFF that optimizes the runtime of the
# basis calculation by using numpy's outer product function when appropriate.
class LSSTBasisPolynomial(BasisPolynomial):
    _type_name = "LSSTBasisPolynomial"

    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        # Get the interpolation key values
        vals = self.getProperties(star)
        # Make 1d arrays of all needed powers of keys
        pows1d = []
        for i, o in enumerate(self._orders):
            p = np.ones(o + 1, dtype=float)
            p[1:] = vals[i]
            pows1d.append(np.cumprod(p))
        # Use trick to produce outer product of all these powers
        # pows2d = np.prod(np.ix_(*pows1d))

        # This check and using np.outer is what is different vs piff
        if len(pows1d) == 2:
            pows2d = np.outer(*pows1d)
        else:
            pows2d = np.prod(np.meshgrid(*pows1d, indexing="ij"), axis=0)
        # Return linear array of terms making total power constraint
        return pows2d[self._mask]
