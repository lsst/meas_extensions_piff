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

__all__ = ["PiffPsfDeterminerConfig", "PiffPsfDeterminerTask"]

import numpy as np
import piff
import galsim
import re
import logging
import yaml

import lsst.utils.logging
from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
import lsst.pex.config as pexConfig
import lsst.meas.algorithms as measAlg
from lsst.meas.algorithms.psfDeterminer import BasePsfDeterminerTask
from lsst.pipe.base import AlgorithmError
from .piffPsf import PiffPsf
from .wcs_wrapper import CelestialWcsWrapper, UVWcsWrapper


def _validateGalsimInterpolant(name: str) -> bool:
    """A helper function to validate the GalSim interpolant at config time.

    Parameters
    ----------
    name : str
        The name of the interpolant to use from GalSim.  Valid options are:
            galsim.Lanczos(N) or Lancsos(N), where N is a positive integer
            galsim.Linear
            galsim.Cubic
            galsim.Quintic
            galsim.Delta
            galsim.Nearest
            galsim.SincInterpolant

    Returns
    -------
    is_valid : bool
        Whether the provided interpolant name is valid.
    """
    # First, check if ``name`` is a valid Lanczos interpolant.
    for pattern in (re.compile(r"Lanczos\(\d+\)"), re.compile(r"galsim.Lanczos\(\d+\)"),):
        match = re.match(pattern, name)  # Search from the start of the string.
        if match is not None:
            # Check that the pattern is also the end of the string.
            return match.end() == len(name)

    # If not, check if ``name`` is any other valid GalSim interpolant.
    names = {f"galsim.{interp}" for interp in
             ("Cubic", "Delta", "Linear", "Nearest", "Quintic", "SincInterpolant")
             }
    return name in names


class PiffTooFewGoodStarsError(AlgorithmError):
    """Raised if too few good stars are available for PSF determination.

    Parameters
    ----------
    num_good_stars : `int`
        Number of good stars used for PSF determination.
    poly_ndim : `int`
        Number of dependency parameters (dimensions) used in
        polynomial fitting.
    minimum_dof : `int`
        Minimum number of degree of freedom to do the fit.
    """

    def __init__(
        self,
        num_good_stars,
        minimum_dof,
        poly_ndim,
    ):
        self._num_good_stars = num_good_stars
        self._poly_ndim = poly_ndim
        self._minimum_dof = minimum_dof
        super().__init__(
            f"Failed to determine piff psf: too few good stars ({num_good_stars}) and minimum dof to fit "
            f"a {poly_ndim} order polynomial is {minimum_dof}."
        )

    @property
    def metadata(self):
        return {
            "num_good_stars": self._num_good_stars,
            "poly_ndim": self._poly_ndim,
            "minimum_dof": self._minimum_dof,
        }


class PiffPsfDeterminerConfig(BasePsfDeterminerTask.ConfigClass):
    spatialOrderPerBand = pexConfig.DictField(
        doc="Per-band spatial order for PSF kernel creation. "
        "Ignored if piffPsfConfigYaml is set.",
        keytype=str,
        itemtype=int,
        default={},
    )
    spatialOrder = pexConfig.Field[int](
        doc="Spatial order for PSF kernel creation. "
        "Ignored if piffPsfConfigYaml is set or if the current "
        "band is in the spatialOrderPerBand dict config.",
        default=2,
    )
    piffBasisPolynomialSolver = pexConfig.ChoiceField[str](
        doc="Solver to solve linear algebra for "
        "the spatial interpolation of the PSF. "
        "Ignore if piffPsfConfigYaml is not None.",
        allowed=dict(
            scipy="Default solver using scipy.",
            cpp="C++ solver using eigen.",
            jax="JAX solver.",
            qr="QR decomposition using scipy/numpy.",
        ),
        default="scipy",
    )
    piffPixelGridFitCenter = pexConfig.Field[bool](
        doc="PixelGrid can re-estimate center if this option is True. "
        "Will use provided centroid if set to False. Default in Piff "
        "is set to True. Depends on how input centroids can be trust. "
        "Ignore if piffPsfConfigYaml is not None.",
        default=True
    )
    samplingSize = pexConfig.Field[float](
        doc="Resolution of the internal PSF model relative to the pixel size; "
        "e.g. 0.5 is equal to 2x oversampling. This affects only the size of "
        "the PSF model stamp if piffPsfConfigYaml is set.",
        default=1,
    )
    modelSize = pexConfig.Field[int](
        doc="Internal model size for PIFF (typically odd, but not enforced). "
        "Partially ignored if piffPsfConfigYaml is set.",
        default=25,
    )
    outlierNSigma = pexConfig.Field[float](
        doc="n sigma for chisq outlier rejection. "
        "Ignored if piffPsfConfigYaml is set.",
        default=4.0
    )
    outlierMaxRemove = pexConfig.Field[float](
        doc="Max fraction of stars to remove as outliers each iteration. "
        "Ignored if piffPsfConfigYaml is set.",
        default=0.05
    )
    maxSNR = pexConfig.Field[float](
        doc="Rescale the weight of bright stars such that their SNR is less "
            "than this value.",
        default=200.0
    )
    zeroWeightMaskBits = pexConfig.ListField[str](
        doc="List of mask bits for which to set pixel weights to zero.",
        default=['BAD', 'CR', 'INTRP', 'SAT', 'SUSPECT', 'NO_DATA']
    )
    minimumUnmaskedFraction = pexConfig.Field[float](
        doc="Minimum fraction of unmasked pixels required to use star.",
        default=0.5
    )
    useColor = pexConfig.Field[bool](
        doc="Use color information to correct for amtospheric turbulences and "
            "differential chromatic refraction."
            "Ignored if piffPsfConfigYaml is set.",
        default=False,
    )
    color = pexConfig.DictField(
        doc="The bands to use for calculating color."
        "Ignored if piffPsfConfigYaml is set.",
        default={},
        keytype=str,
        itemtype=str,
    )
    colorOrder = pexConfig.Field[int](
        doc="Color order for PSF kernel creation. "
        "Ignored if piffPsfConfigYaml is set.",
        default=1,
    )
    interpolant = pexConfig.Field[str](
        doc="GalSim interpolant name for Piff to use. "
            "Options include 'Lanczos(N)', where N is an integer, along with "
            "galsim.Cubic, galsim.Delta, galsim.Linear, galsim.Nearest, "
            "galsim.Quintic, and galsim.SincInterpolant. Ignored if "
            "piffPsfConfigYaml is set.",
        check=_validateGalsimInterpolant,
        default="Lanczos(11)",
    )
    zerothOrderInterpNotEnoughStars = pexConfig.Field[bool](
        doc="If True, use zeroth order interpolation if not enough stars are found.",
        default=False
    )
    debugStarData = pexConfig.Field[bool](
        doc="Include star images used for fitting in PSF model object.",
        default=False
    )
    useCoordinates = pexConfig.ChoiceField[str](
        doc="Which spatial coordinates to regress against in PSF modeling.",
        allowed=dict(
            pixel='Regress against pixel coordinates',
            field='Regress against field angles',
            sky='Regress against RA/Dec'
        ),
        default='pixel'
    )
    piffMaxIter = pexConfig.Field[int](
        doc="Maximum iteration while doing outlier rejection."
        "Might be overwrite if zerothOrderInterpNotEnoughStars is True and "
        "ignore if piffPsfConfigYaml is not None.",
        default=15
    )
    piffLoggingLevel = pexConfig.RangeField[int](
        doc="PIFF-specific logging level, from 0 (least chatty) to 3 (very verbose); "
            "note that logs will come out with unrelated notations (e.g. WARNING does "
            "not imply a warning.)",
        default=0,
        min=0,
        max=3,
    )
    piffPsfConfigYaml = pexConfig.Field[str](
        doc="Configuration file for PIFF in YAML format. This overrides many "
        "other settings in this config and is not validated ahead of runtime.",
        default=None,
        optional=True,
    )
    writeTrainingSet = pexConfig.Field[bool](
        doc="write training set for a learned PSF model.",
        default=False,
    )
    trainingSetLocation = pexConfig.Field[str](
        doc="Where to write training set for a learned PSF model.",
        default="",
    )
    cameraModelTrainingSet = pexConfig.Field[str](
        doc="which camera is used for the learned PSF model.",
        default="LSSTComCam",
    )

    def setDefaults(self):
        super().setDefaults()
        # stampSize should be at least 25 so that
        # i) aperture flux with 12 pixel radius can be compared to PSF flux.
        # ii) fake sources injected to match the 12 pixel aperture flux get
        #     measured correctly
        # stampSize should also be at least sqrt(2)*modelSize/samplingSize so
        # that rotated images have support in the model

        self.stampSize = 25
        # Resize the stamp to accommodate the model, but only if necessary.
        if self.useCoordinates == "sky":
            self.stampSize = max(25, 2*int(0.5*self.modelSize*np.sqrt(2)/self.samplingSize) + 1)

    def validate(self):
        super().validate()

        if (stamp_size := self.stampSize) is not None:
            model_size = self.modelSize
            sampling_size = self.samplingSize
            if self.useCoordinates == 'sky':
                min_stamp_size = int(np.sqrt(2) * model_size / sampling_size)
            else:
                min_stamp_size = int(model_size / sampling_size)

            if stamp_size < min_stamp_size:
                msg = (f"PIFF model size of {model_size} is too large for stamp size {stamp_size}. "
                       f"Set stampSize >= {min_stamp_size}"
                       )
                raise pexConfig.FieldValidationError(self.__class__.modelSize, self, msg)


def getGoodPixels(maskedImage, zeroWeightMaskBits):
    """Compute an index array indicating good pixels to use.

    Parameters
    ----------
    maskedImage : `afw.image.MaskedImage`
        PSF candidate postage stamp
    zeroWeightMaskBits : `List[str]`
        List of mask bits for which to set pixel weights to zero.

    Returns
    -------
    good : `ndarray`
        Index array indicating good pixels.
    """
    imArr = maskedImage.image.array
    varArr = maskedImage.variance.array
    bitmask = maskedImage.mask.getPlaneBitMask(zeroWeightMaskBits)
    good = (
        (varArr != 0)
        & (np.isfinite(varArr))
        & (np.isfinite(imArr))
        & ((maskedImage.mask.array & bitmask) == 0)
    )
    return good


def computeWeight(maskedImage, maxSNR, good):
    """Derive a weight map without Poisson variance component due to signal.

    Parameters
    ----------
    maskedImage : `afw.image.MaskedImage`
        PSF candidate postage stamp
    maxSNR : `float`
        Maximum SNR applying variance floor.
    good : `ndarray`
        Index array indicating good pixels.

    Returns
    -------
    weightArr : `ndarry`
        Array to use for weight.

    See Also
    --------
    `lsst.meas.algorithms.variance_plance.remove_signal_from_variance` :
        Remove the Poisson contribution from sources in the variance plane of
        an Exposure.
    """
    imArr = maskedImage.image.array
    varArr = maskedImage.variance.array

    # Fit a straight line to variance vs (sky-subtracted) signal.
    # The evaluate that line at zero signal to get an estimate of the
    # signal-free variance.
    fit = np.polyfit(imArr[good], varArr[good], deg=1)
    # fit is [1/gain, sky_var]
    weightArr = np.zeros_like(imArr, dtype=float)
    weightArr[good] = 1./fit[1]

    applyMaxSNR(imArr, weightArr, good, maxSNR)
    return weightArr


def applyMaxSNR(imArr, weightArr, good, maxSNR):
    """Rescale weight of bright stars to cap the computed SNR.

    Parameters
    ----------
    imArr : `ndarray`
        Signal (image) array of stamp.
    weightArr : `ndarray`
        Weight map array.  May be rescaled in place.
    good : `ndarray`
        Index array of pixels to use when computing SNR.
    maxSNR : `float`
        Threshold for adjusting variance plane implementing maximum SNR.
    """
    # We define the SNR value following Piff.  Here's the comment from that
    # code base explaining the calculation.
    #
    # The S/N value that we use will be the weighted total flux where the
    # weight function is the star's profile itself.  This is the maximum S/N
    # value that any flux measurement can possibly produce, which will be
    # closer to an in-practice S/N than using all the pixels equally.
    #
    # F = Sum_i w_i I_i^2
    # var(F) = Sum_i w_i^2 I_i^2 var(I_i)
    #        = Sum_i w_i I_i^2             <--- Assumes var(I_i) = 1/w_i
    #
    # S/N = F / sqrt(var(F))
    #
    # Note that if the image is pure noise, this will produce a "signal" of
    #
    # F_noise = Sum_i w_i 1/w_i = Npix
    #
    # So for a more accurate estimate of the S/N of the actual star itself, one
    # should subtract off Npix from the measured F.
    #
    # The final formula then is:
    #
    # F = Sum_i w_i I_i^2
    # S/N = (F-Npix) / sqrt(F)
    F = np.sum(weightArr[good]*imArr[good]**2, dtype=float)
    Npix = np.sum(good)
    SNR = 0.0 if F < Npix else (F-Npix)/np.sqrt(F)
    # rescale weight of bright stars.  Essentially makes an error floor.
    if SNR > maxSNR:
        factor = (maxSNR / SNR)**2
        weightArr[good] *= factor


def _computeWeightAlternative(maskedImage, maxSNR):
    """Alternative algorithm for creating weight map.

    This version is equivalent to that used by Piff internally.  The weight map
    it produces tends to leave a residual when removing the Poisson component
    due to the signal.  We leave it here as a reference, but without intending
    that it be used (or be maintained).
    """
    imArr = maskedImage.image.array
    varArr = maskedImage.variance.array
    good = (varArr != 0) & np.isfinite(varArr) & np.isfinite(imArr)

    fit = np.polyfit(imArr[good], varArr[good], deg=1)
    # fit is [1/gain, sky_var]
    gain = 1./fit[0]
    varArr[good] -= imArr[good] / gain
    weightArr = np.zeros_like(imArr, dtype=float)
    weightArr[good] = 1./varArr[good]

    applyMaxSNR(imArr, weightArr, good, maxSNR)
    return weightArr


class PiffPsfDeterminerTask(BasePsfDeterminerTask):
    """A measurePsfTask PSF estimator using Piff as the implementation.
    """
    ConfigClass = PiffPsfDeterminerConfig
    _DefaultName = "psfDeterminer.Piff"

    def __init__(self, config, schema=None, **kwds):
        BasePsfDeterminerTask.__init__(self, config, schema=schema, **kwds)

        piffLoggingLevels = {
            0: logging.CRITICAL,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
        }
        self.piffLogger = lsst.utils.logging.getLogger(f"{self.log.name}.piff")
        self.piffLogger.setLevel(piffLoggingLevels[self.config.piffLoggingLevel])

    def determinePsf(
        self, exposure, psfCandidateList, metadata=None, flagKey=None,
    ):
        """Determine a Piff PSF model for an exposure given a list of PSF
        candidates.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
           Exposure containing the PSF candidates.
        psfCandidateList : `list` of `lsst.meas.algorithms.PsfCandidate`
           A sequence of PSF candidates typically obtained by detecting sources
           and then running them through a star selector.
        metadata : `lsst.daf.base import PropertyList` or `None`, optional
           A home for interesting tidbits of information.
        flagKey : `str` or `None`, optional
           Schema key used to mark sources actually used in PSF determination.

        Returns
        -------
        psf : `lsst.meas.extensions.piff.PiffPsf`
           The measured PSF model.
        psfCellSet : `None`
           Unused by this PsfDeterminer.
        """
        psfCandidateList = self.downsampleCandidates(psfCandidateList)

        if self.config.stampSize:
            stampSize = self.config.stampSize
            if stampSize > psfCandidateList[0].getWidth():
                self.log.warning("stampSize is larger than the PSF candidate size.  Using candidate size.")
                stampSize = psfCandidateList[0].getWidth()
        else:  # TODO: Only the if block should stay after DM-36311
            self.log.debug("stampSize not set.  Using candidate size.")
            stampSize = psfCandidateList[0].getWidth()

        scale = exposure.getWcs().getPixelScale(exposure.getBBox().getCenter()).asArcseconds()

        match self.config.useCoordinates:
            case 'field':
                detector = exposure.getDetector()
                pix_to_field = detector.getTransform(PIXELS, FIELD_ANGLE)
                gswcs = UVWcsWrapper(pix_to_field)
                pointing = None
                keys = ['u', 'v']

            case 'sky':
                gswcs = CelestialWcsWrapper(exposure.getWcs())
                skyOrigin = exposure.getWcs().getSkyOrigin()
                ra = skyOrigin.getLongitude().asDegrees()
                dec = skyOrigin.getLatitude().asDegrees()
                pointing = galsim.CelestialCoord(
                    ra*galsim.degrees,
                    dec*galsim.degrees
                )
                keys = ['u', 'v']

            case 'pixel':
                gswcs = galsim.PixelScale(scale)
                pointing = None
                keys = ['x', 'y']

        stars = []
        for candidate in psfCandidateList:
            cmi = candidate.getMaskedImage(stampSize, stampSize)
            good = getGoodPixels(cmi, self.config.zeroWeightMaskBits)
            fracGood = np.sum(good)/good.size
            if fracGood < self.config.minimumUnmaskedFraction:
                continue
            weight = computeWeight(cmi, self.config.maxSNR, good)

            bbox = cmi.getBBox()
            bds = galsim.BoundsI(
                galsim.PositionI(*bbox.getMin()),
                galsim.PositionI(*bbox.getMax())
            )
            gsImage = galsim.Image(bds, wcs=gswcs, dtype=float)
            gsImage.array[:] = cmi.image.array
            gsWeight = galsim.Image(bds, wcs=gswcs, dtype=float)
            gsWeight.array[:] = weight

            source = candidate.getSource()
            starId = source.getId()
            image_pos = galsim.PositionD(source.getX(), source.getY())

            data = piff.StarData(
                gsImage,
                image_pos,
                weight=gsWeight,
                pointing=pointing
            )
            star = piff.Star(data, None)
            star.data.properties['starId'] = starId
            star.data.properties['colorValue'] = candidate.getPsfColorValue()
            star.data.properties['colorType'] = candidate.getPsfColorType()
            stars.append(star)

        # The following is mostly accommodating unittests that don't have
        # the filter attribute set on the mock exposure.
        if hasattr(exposure.filter, "bandLabel"):
            band = exposure.filter.bandLabel
        else:
            band = None
        spatialOrder = self.config.spatialOrderPerBand.get(band, self.config.spatialOrder)
        orders = [spatialOrder] * len(keys)

        if self.config.useColor:
            colors = [s.data.properties['colorValue'] for s in stars
                      if np.isfinite(s.data.properties['colorValue'])]
            colorTypes = [s.data.properties['colorType'] for s in stars
                          if np.isfinite(s.data.properties['colorValue'])]
            if len(colors) == 0:
                self.log.warning("No color informations for PSF candidates, Set PSF colors to 0s.")
                meanColors = 0.
            else:
                meanColors = np.mean(colors)
                colorType = list(set(colorTypes))
                if len(colorType) > 1:
                    raise ValueError(f"More than one colorType was providen:{colorType}")
                colorType = colorType[0]
            for s in stars:
                if not np.isfinite(s.data.properties['colorValue']):
                    s.data.properties['colorValue'] = meanColors
                    s.data.properties['colorType'] = colorType
            keys.append('colorValue')
            orders.append(self.config.colorOrder)

        if self.config.piffPsfConfigYaml is None:
            piffConfig = {
                'type': 'Simple',
                'model': {
                    'type': 'PixelGrid',
                    'scale': scale * self.config.samplingSize,
                    'size': self.config.modelSize,
                    'interp': self.config.interpolant,
                    'centered': self.config.piffPixelGridFitCenter,
                },
                'interp': {
                    'type': 'BasisPolynomial',
                    'order': orders,
                    'keys': keys,
                    'solver': self.config.piffBasisPolynomialSolver,
                },
                'outliers': {
                    'type': 'Chisq',
                    'nsigma': self.config.outlierNSigma,
                    'max_remove': self.config.outlierMaxRemove,
                },
                'max_iter': self.config.piffMaxIter
            }
        else:
            piffConfig = yaml.safe_load(self.config.piffPsfConfigYaml)

        def _get_threshold(nth_order):
            if isinstance(nth_order, list):
                # right now, nth_order[0] and nth_order[1] are the same.
                freeParameters = ((nth_order[0] + 1) * (nth_order[0] + 2)) // 2
                if len(nth_order) == 3:  # when color correction
                    freeParameters += nth_order[2]
            else:
                # number of free parameter in the polynomial interpolation
                freeParameters = ((nth_order + 1) * (nth_order + 2)) // 2
            return freeParameters

        if piffConfig['interp']['type'] in ['BasisPolynomial', 'Polynomial']:
            threshold = _get_threshold(piffConfig['interp']['order'])
            if len(stars) < threshold:
                if self.config.zerothOrderInterpNotEnoughStars:
                    self.log.warning(
                        "Only %d stars found, "
                        "but %d required. Using zeroth order interpolation."%((len(stars), threshold))
                    )
                    piffConfig['interp']['order'] = [0] * len(keys)
                    # No need to do any outlier rejection assume
                    # PSF to be average of few stars.
                    piffConfig['max_iter'] = 1
                else:
                    raise PiffTooFewGoodStarsError(
                        num_good_stars=len(stars),
                        minimum_dof=threshold,
                        poly_ndim=piffConfig['interp']['order'],
                    )
        self._piffConfig = piffConfig
        piffResult = piff.PSF.process(piffConfig)
        wcs = {0: gswcs}

        piffResult.fit(stars, wcs, pointing, logger=self.piffLogger)

        nUsedStars = len([s for s in piffResult.stars if not s.is_flagged and not s.is_reserve])

        if piffConfig['interp']['type'] in ['BasisPolynomial', 'Polynomial']:
            threshold = _get_threshold(piffConfig['interp']['order'])
            if nUsedStars < threshold:
                if self.config.zerothOrderInterpNotEnoughStars:
                    self.log.warning(
                        "Only %d after outlier rejection, "
                        "but %d required. Using zeroth order interpolation."%((nUsedStars, threshold))
                    )
                    piffConfig['interp']['order'] = [0] * len(keys)
                    # No need to do any outlier rejection assume
                    # PSF to be average of few stars.
                    piffConfig['max_iter'] = 1
                    piffResult.fit(stars, wcs, pointing, logger=self.piffLogger)
                    nUsedStars = len(stars)
                else:
                    raise PiffTooFewGoodStarsError(
                        num_good_stars=nUsedStars,
                        minimum_dof=threshold,
                        poly_ndim=piffConfig['interp']['order'],
                    )

        drawSize = 2*np.floor(0.5*stampSize/self.config.samplingSize) + 1

        used_image_starId = {s.data.properties['starId'] for s in piffResult.stars
                             if not s.is_flagged and not s.is_reserve}

        assert len(used_image_starId) == nUsedStars, "Star IDs are not unique"

        if flagKey:
            for candidate in psfCandidateList:
                source = candidate.getSource()
                starId = source.getId()
                if starId in used_image_starId:
                    source.set(flagKey, True)

        if self.config.writeTrainingSet:

            dic = {}

            import lsst.afw.cameraGeom as cameraGeom
            from lsst.obs.lsst import LsstComCam, LsstCam
            from lsst.obs.subaru import HyperSuprimeCam
            from lsst.geom import Point2D
            import pickle
            import os

            if self.config.cameraModelTrainingSet not in ["LSSTComCam", "LSSTCam", "HyperSuprimeCam"]:
                raise ValueError('work only for LSST cameras')
            if self.config.cameraModelTrainingSet == "LSSTComCam":
                camera = LsstComCam.getCamera()
            if self.config.cameraModelTrainingSet == "LSSTCam":
                camera = LsstCam.getCamera()
            if self.config.cameraModelTrainingSet ==  "HyperSuprimeCam":
                hsc = HyperSuprimeCam()
                camera = hsc.getCamera()

            def _pixel_to_focal(x, y, det):
                tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
                fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
                if camera in ["LSSTComCam", "LSSTCam"]:
                    return fpx.ravel()[0], fpy.ravel()[0]
                if camera in ["HyperSuprimeCam"]:
                    return fpx, fpy
        
            def _get_second_moment(image, guessCentroid, sigma):

                shape = galsim.hsm.FindAdaptiveMom(
                    image,
                    weight=None,
                    badpix=None,  # Already incorporated into `weight_image`.
                    guess_sig=sigma,
                    precision=1.0e-6,
                    guess_centroid=guessCentroid,
                    strict=True,  # Raises GalSimHSMError if estimation fails.
                    check=False,  # This speeds up the code!
                    round_moments=False,
                    hsmparams=None,
                )
                return shape.moments_sigma, shape.observed_shape.e1, shape.observed_shape.e2

            psfModel = PiffPsf(drawSize, drawSize, piffResult)

            detectorId = exposure.getDetector().getId()
            visitId = exposure.getInfo().getVisitInfo().id
            bandId = exposure.getInfo().getFilter().bandLabel

            for s in piffResult.stars:
                if not s.is_flagged and not s.is_reserve:
                    starId = f"{visitId}_{detectorId}_{bandId}_{s.data.properties['starId']}"
                    starPiff = piffResult.draw(s.x, s.y, stamp_size=drawSize, center=None)
                    xFoV, yFoV = _pixel_to_focal(np.array([s.x]), np.array([s.y]), camera[detectorId])
                    sumStar = np.sum(s.data.image.array)

                    centroid = galsim._PositionD(s.x, s.y)
                    psfSigma = psfModel.computeShape(Point2D(s.x, s.y)).getTraceRadius()
                    try:
                        sigmaStar, e1Star, e2Star = _get_second_moment(s.data.image, centroid, psfSigma)
                        sigmaPiff, e1Piff, e2Piff = _get_second_moment(starPiff, centroid, psfSigma)
                    except Exception:
                        sigmaStar, e1Star, e2Star = -999, -999, -999
                        sigmaPiff, e1Piff, e2Piff = -999, -999, -999

                    starToSave = s.data.image.array / sumStar
                    starPiffToSave = starPiff.array

                    dic[starId] = {"star": starToSave.astype(np.float32),
                                   "weight": None, # s.data.weight.array * sumStar**2,
                                   "starPiff": starPiffToSave.astype(np.float32),
                                   "sigmaStar": sigmaStar,
                                   "e1Star": e1Star,
                                   "e2Star": e2Star,
                                   "sigmaPiff": sigmaPiff,
                                   "e1Piff": e1Piff,
                                   "e2Piff": e2Piff,
                                   "xCCD": s.x,
                                   "yCCD": s.y,
                                   "xFoV": xFoV,
                                   "yFoV": yFoV,
                                   "sumStar": sumStar,
                                   "detector": detectorId,
                                   "visit": visitId,
                                   "band": bandId}

            pklName = os.path.join(self.config.trainingSetLocation, f"{visitId}_{detectorId}_{bandId}.pkl")
            pklFile = open(pklName, 'wb')
            pickle.dump(dic, pklFile)
            pklFile.close()

        if metadata is not None:
            metadata["spatialFitChi2"] = piffResult.chisq
            metadata["numAvailStars"] = len(stars)
            metadata["numGoodStars"] = nUsedStars
            metadata["avgX"] = np.mean([s.x for s in piffResult.stars
                                        if not s.is_flagged and not s.is_reserve])
            metadata["avgY"] = np.mean([s.y for s in piffResult.stars
                                        if not s.is_flagged and not s.is_reserve])

        if not self.config.debugStarData:
            for star in piffResult.stars:
                # Remove large data objects from the stars
                del star.fit.params
                del star.fit.params_var
                del star.fit.A
                del star.fit.b
                del star.data.image
                del star.data.weight
                del star.data.orig_weight

        return PiffPsf(drawSize, drawSize, piffResult), None


measAlg.psfDeterminerRegistry.register("piff", PiffPsfDeterminerTask)
