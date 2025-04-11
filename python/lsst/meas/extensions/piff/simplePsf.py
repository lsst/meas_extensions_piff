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

__all__ = ("LSSTSimplePSF",)

from piff import SimplePSF, Star, BasisInterp, StarData, StarFit
import galsim
import numpy as np


def makeTarget(
    x=None,
    y=None,
    u=None,
    v=None,
    properties={},
    wcs=None,
    scale=None,
    stamp_size=48,
    image=None,
    weight=None,
    pointing=None,
    flux=1.0,
    **kwargs,
):
    # This is a copy of the makeTarge class method on Star, but re-implemented
    # to avoid using eval as it is a costly function, as well calling
    #  image._shift instead of setCenter, as it executes an expensive call we
    # dont need and then itself calls shift.

    # Check that input parameters are valid
    local_vars = vars()
    for param in ["x", "y", "u", "v"]:
        if param in properties and local_vars[param] is not None:
            raise TypeError("%s may not be given both as a kwarg and in properties" % param)
    properties = properties.copy()  # So we can modify it and not mess up the caller.
    x = properties.pop("x", x)
    y = properties.pop("y", y)
    u = properties.pop("u", u)
    v = properties.pop("v", v)
    properties.update(kwargs)  # Add any extra kwargs into properties
    if (x is None) != (y is None):
        raise TypeError("Either x and y must both be given, or neither.")
    if (u is None) != (v is None):
        raise TypeError("Either u and v must both be given, or neither.")
    if x is None and u is None:
        raise TypeError("Some kind of position must be given.")
    if wcs is not None and scale is not None:
        raise TypeError("Scale is invalid when also providing wcs.")

    # Figure out what the wcs should be if not provided
    if wcs is None:
        if scale is None:
            scale = 1.0
        wcs = galsim.PixelScale(scale)

    # Make field_pos if we have u,v
    if u is not None:
        field_pos = galsim.PositionD(float(u), float(v))
    else:
        field_pos = None

    # Figure out the image_pos
    if x is None:
        image_pos = wcs.toImage(field_pos)
        x = image_pos.x
        y = image_pos.y
    else:
        image_pos = galsim.PositionD(float(x), float(y))

    # Make the blank image
    if image is None:
        image = galsim.Image(stamp_size, stamp_size, dtype=float)
        # Make the center of the image (close to) the image_pos
        xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
        ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))
        image._shift(galsim.position._PositionI(xcen, ycen) - image.center)
        # image.setCenter(xcen, ycen)
    if image.wcs is None:
        image.wcs = wcs
    if weight is not None:
        weight = galsim.Image(weight.array, wcs=image.wcs, copy=True, bounds=image.bounds)

    # Build the StarData instance
    data = StarData(
        image, image_pos, field_pos=field_pos, properties=properties, pointing=pointing, weight=weight
    )
    fit = StarFit(None, flux=flux, center=(0.0, 0.0))
    return Star(data, fit)


# This is a specialized subclass which modifies behaviors of the parent class
# to optimize runtime performance.
#
# get_profile has been modified to operate
# on a starFit in place instead of allocating a new object. This is safe in
# this case because we know the star being passed in was freshly created,
# and no one else has a reference to it.
#
# draw has been modified to call the _shift method on an image, instead of
# setCenter. setting the center does one additional operation that has a
# high overhead, and is not needed, as we are already calculating exactly
# what shift should be done ahead of time.
class LSSTSimplePSF(SimplePSF):
    _type_name = "LSSTSimple"

    def get_profile(self, x, y, chipnum=None, flux=1.0, logger=None, **kwargs):
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        properties = {"chipnum": chipnum}
        for key in self.interp_property_names:
            if key in ["x", "y", "u", "v"]:
                continue
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required" % key)
            properties[key] = kwargs.pop(key)
        if len(kwargs) != 0:
            raise TypeError("Unexpected keyword argument(s) %r" % list(kwargs.keys())[0])

        image_pos = galsim.PositionD(x, y)
        wcs = self.wcs[chipnum]
        field_pos = StarData.calculateFieldPos(image_pos, wcs, self.pointing, properties)
        u, v = field_pos.x, field_pos.y

        star = makeTarget(x=x, y=y, u=u, v=v, wcs=wcs, properties=properties, pointing=self.pointing)
        logger.debug("Getting PSF profile at (%s,%s) on chip %s", x, y, chipnum)

        # Interpolate and adjust the flux of the star.
        if isinstance(self.interp, BasisInterp):
            K = self.interp.basis(star)
            p = np.dot(self.interp.q, K)

            # in basis interp
            if self.interp._num is not None:
                star.fit.params[self.interp._num] = p
            else:
                star.fit.params = p

        else:
            star = self.interp.interpolate(star)
        self.model.normalize(star)
        star.fit.flux = flux
        # star = self.interpolateStar(star).withFlux(flux)

        # The last step is implementd in the derived classes.
        prof, method = self._getProfile(star)
        return prof, method

    def draw(
        self,
        x,
        y,
        chipnum=None,
        flux=1.0,
        center=None,
        offset=None,
        stamp_size=48,
        image=None,
        logger=None,
        **kwargs,
    ):
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        prof, method = self.get_profile(x, y, chipnum=chipnum, flux=flux, logger=logger, **kwargs)

        logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        # Make the image if necessary
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)
            # Make the center of the image (close to) the image_pos
            xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
            ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))
            image._shift(galsim.position._PositionI(xcen, ycen) - image.center)
            # image.setCenter(xcen, ycen)

        # If no wcs is given, use the original wcs
        if image.wcs is None:
            image.wcs = self.wcs[chipnum]

        # Handle the input center
        if center is None:
            center = (x, y)
        elif center is True:
            center = image.true_center
            center = (center.x, center.y)
        elif not isinstance(center, tuple):
            raise ValueError("Invalid center parameter: %r. Must be tuple or None or True" % (center))

        # Handle offset if given
        if offset is not None:
            center = (center[0] + offset[0], center[1] + offset[1])

        prof.drawImage(image, method=method, center=center)

        return image
