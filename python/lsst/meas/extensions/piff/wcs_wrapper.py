from lsst.geom import Point2D

import galsim
import numpy as np


class CelestialWcsWrapper(galsim.wcs.CelestialWCS):
    """Wrap a `lsst.afw.geom.SkyWcs` in a GalSim WCS.

    Parameters
    ----------
    pix_to_sky : `lsst.afw.geom.SkyWcs`
        WCS to wrap.
    origin : `galsim.PositionD`, optional
        Origin in image coordinates.
    """
    def __init__(self, pix_to_sky, origin=None):
        if origin is None:
            # Use galsim._PositionD as it's faster than galsim.PositionD
            origin = galsim._PositionD(0.0, 0.0)
        self._pix_to_sky = pix_to_sky
        self._origin = origin
        self._color = None

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    def _radec(self, x, y, color=None):
        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)

        ra, dec = self._pix_to_sky.pixelToSkyArray(x1, y1)

        if np.ndim(x) == np.ndim(y) == 0:
            return ra[0], dec[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
            return ra, dec

    def _xy(self, ra, dec, color=None):
        r1 = np.atleast_1d(ra)
        d1 = np.atleast_1d(dec)

        x, y = self._pix_to_sky.skyToPixelArray(r1, d1)

        if np.ndim(ra) == np.ndim(dec) == 0:
            return x[0], y[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(ra) == np.ndim(dec)
            assert ra.shape == dec.shape
            return x, y

    def _newOrigin(self, origin):
        """Return a new CelestialWcsWrapper with new origin.

        Parameters
        ----------
        origin : `galsim.PositionD`, optional
            Origin in image coordinates.

        Returns
        -------
        ret : `CelestialWcsWrapper`
            Transformed WCS.
        """
        return CelestialWcsWrapper(self._pix_to_sky, origin=origin)


class UVWcsWrapper(galsim.wcs.EuclideanWCS):
    """Wrap a `lsst.afw.geom.TransformPoint2ToPoint2[2->2]` in a GalSim WCS.

    Parameters
    ----------
    pix_to_field : `lsst.afw.geom.TransformPoint2ToPoint2[2->2]`
        Transform to wrap.  Most likely PIXELS -> FIELD_ANGLE, but other 2D ->
        2D transforms should be possible.
    origin : `galsim.PositionD`, optional
        Origin in image coordinates.
    world_origin : `galsim.PositionD`, optional
        Origin in world coordinates.

    Notes
    -----

    GalSim EuclideanWCS assumes
        u = ufunc(x-x0, y-y0) + u0
        v = vfunc(x-x0, y-y0) + v0
    where u,v are world (likely field angles), and (x, y) are pixels.
    GalSim also assumes that origin = x0, y0 and world_origin = u0, v0.
    I might have naively thought that uv(origin) == world_origin, but
    that appears to not be required. So we're just going to leave both
    free.
    """
    _rad_to_arcsec = 206264.80624709636

    def __init__(self, pix_to_field, origin=None, world_origin=None):
        if origin is None:
            # Use galsim._PositionD as it's faster than galsim.PositionD
            origin = galsim._PositionD(0.0, 0.0)
        if world_origin is None:
            world_origin = galsim._PositionD(0.0, 0.0)
        self._pix_to_field = pix_to_field
        self._origin = origin
        self._world_origin = world_origin
        self._mapping = self._pix_to_field.getMapping()
        self._color = None

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return self._world_origin

    def _xyTouv(self, x, y, color=None):
        """Convert image coordinates to world coordinates.

        Parameters
        ----------
        x, y : ndarray
            Image coordinates.
        color : ndarray
            Color to use in transformation.  Unused currently.

        Returns
        -------
        u, v : ndarray
            World coordinates.
        """
        x = x - self.x0
        y = y - self.y0
        u, v = self._mapping.applyForward(np.vstack((x, y)))
        u *= self._rad_to_arcsec
        v *= self._rad_to_arcsec
        return u + self.u0, v + self.v0

    def _uvToxy(self, u, v, color):
        """Convert world coordinates to image coordinates.

        Parameters
        ----------
        u, v : ndarray
            World coordinates.
        color : ndarray
            Color to use in transformation.  Unused currently.

        Returns
        -------
        x, y : ndarray
            Image coordinates.
        """
        u = (u - self.u0)/self._rad_to_arcsec
        v = (v - self.v0)/self._rad_to_arcsec
        x, y = self._mapping.applyInverse(np.vstack((u, v)))
        return x + self.x0, y + self.y0

    def _posToWorld(self, image_pos, color=None):
        """Convert image coordinate to world coordinate.

        Parameters
        ----------
        image_pos : galsim.PositionD
            Image coordinate.
        color : ndarray
            Color to use in transformation.  Unused currently.

        Returns
        -------
        world_pos : galsim.PositionD
            World coordinate.
        """
        # Use galsim._PositionD as it's faster than galsim.PositionD
        return galsim._PositionD(*self._xyTouv(image_pos.x, image_pos.y, color))

    def _local(self, image_pos, color=None):
        """Compute local Jacobian WCS.

        Parameters
        ----------
        image_pos : galsim.PositionD
            Image position at which to compute local WCS.
        color : ndarray
            Color to use in transformation.  Unused currently.

        Returns
        -------
        local : galsim.JacobianWCS
            Local linear approximation to WCS.
        """
        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0

        duvdxy = self._pix_to_field.getJacobian(Point2D(x0, y0))
        return galsim.JacobianWCS(
            *(duvdxy.ravel()*self._rad_to_arcsec)
        )

    def _newOrigin(self, origin, world_origin):
        """Return a new UVWcsWrapper with new origins.

        Parameters
        ----------
        origin : `galsim.PositionD`, optional
            Origin in image coordinates.
        world_origin : `galsim.PositionD`, optional
            Origin in world coordinates.

        Returns
        -------
        ret : `UVWcsWrapper`
            Transformed WCS.
        """
        return UVWcsWrapper(self._pix_to_field, origin, world_origin)
