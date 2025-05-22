# This file is part of meas_extensions_piff.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["PiffPsf"]

import pickle
import piff
from packaging.version import Version
import numpy as np
from lsst.afw.typehandling import StorableHelperFactory
from lsst.meas.algorithms import ImagePsf
from lsst.afw.image import Image, Color
from lsst.geom import Box2I, Point2I, Extent2I, Point2D
import logging


class PiffPsf(ImagePsf):
    _factory = StorableHelperFactory(
        "lsst.meas.extensions.piff.piffPsf",
        "PiffPsf"
    )

    def __init__(self, width, height, piffResult, log=None):
        assert width == height
        ImagePsf.__init__(self)
        self.width = width
        self.height = height
        self.dimensions = Extent2I(width, height)
        self._piffResult = piffResult
        self._averagePosition = None
        self._averageColor = None
        self.log = log or logging.getLogger(__name__)

        try:
            color_type = [s.data.properties['colorType'] for s in self._piffResult.stars
                          if not s.is_flagged and not s.is_reserve]
            color_type = list(set(color_type))
            if len(color_type) > 1:
                self.log.warning(f"More than one color type was given during training: {color_type}")
                self._color_type = None
            else:
                self._color_type = color_type[0]
        except Exception:
            self._color_type = None

    @property
    def piffResult(self):
        return self._piffResult

    # Storable overrides

    def isPersistable(self):
        return True

    def _getPersistenceName(self):
        return "PiffPsf"

    def _getPythonModule(self):
        return "lsst.meas.extensions.piff.piffPsf"

    def _write(self):
        return pickle.dumps((self.width, self.height, self._piffResult))

    @staticmethod
    def _read(pkl):
        width, height, piffResult = pickle.loads(pkl)
        # We need to do surgery on pickles created with earlier versions of
        # piff when using piff version 1.4 or later.
        if Version(piff.version) >= Version("1.4"):
            if not hasattr(piffResult, "_num"):
                piffResult._num = None
                piffResult.model._num = None
                piffResult.model._fit_flux = None
                piffResult.interp._num = None
        return PiffPsf(width, height, piffResult)

    # ImagePsf overrides

    def __deepcopy__(self, meta=None):
        return PiffPsf(self.width, self.height, self._piffResult)

    def resized(self, width, height):
        assert width == height
        return PiffPsf(width, height, self._piffResult)

    def _doComputeImage(self, position, color):
        return self._doImage(position, center=None, color=color)

    def _doComputeKernelImage(self, position, color):
        return self._doImage(position, center=True, color=color)

    def _doComputeBBox(self, position, color):
        return self._doBBox(Point2I(0, 0), center=True)

    def getAveragePosition(self):
        if self._averagePosition is None:
            x = np.mean([s.image_pos.x for s in self._piffResult.stars
                         if not s.is_flagged and not s.is_reserve])
            y = np.mean([s.image_pos.y for s in self._piffResult.stars
                         if not s.is_flagged and not s.is_reserve])
            self._averagePosition = Point2D(x, y)
        return self._averagePosition

    def getAverageColor(self):
        if self._averageColor is None:
            if 'colorValue' in self._piffResult.interp_property_names:
                c = np.nanmean([s.data.properties['colorValue'] for s in self._piffResult.stars
                                if not s.is_flagged and not s.is_reserve])
                self._averageColor = Color(colorValue=c, colorType=self._color_type)
            else:
                self._averageColor = Color()  # set value to nan.
        return self._averageColor

    # Internal private methods

    def _doImage(self, position, center, color=None):
        # Follow Piff conventions for center.
        # None => draw as if star at position
        # True => draw in center of image
        if 'colorValue' in self._piffResult.interp_property_names:
            if color is None or color.isIndeterminate():
                meanColor = np.nan
                if self._averageColor is None:
                    meanColor = self.getAverageColor().getColorValue()
                else:
                    meanColor = self._averageColor.getColorValue()
                kwargs = {'colorValue': meanColor}
                self.log.warning("PSF model need a color information."
                                 "Set to mean Color from PSF fit right now.")
            else:
                ctype = color.getColorType()
                if self._color_type != color.getColorType():
                    raise ValueError(f"Invalid Color type. Need {self._color_type} but {ctype} is provided")
                kwargs = {'colorValue': color.getColorValue()}
        else:
            kwargs = {}
        gsimg = self._piffResult.draw(
            position.x, position.y, stamp_size=self.width, center=center, **kwargs,
        )
        bbox = self._doBBox(position, center)
        img = Image(bbox, dtype=np.float64)
        img.array[:] = gsimg.array
        img.array /= np.sum(img.array)
        return img

    def _doBBox(self, position, center):
        origin = -(self.dimensions//2)
        if center is None:
            origin = Point2I(position) + origin
        return Box2I(Point2I(origin), self.dimensions)
