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

import lsst.utils.logging
from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
import lsst.pex.config as pexConfig
import lsst.meas.algorithms as measAlg
from lsst.meas.algorithms.psfDeterminer import BasePsfDeterminerTask
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


class PiffPsfDeterminerConfig(BasePsfDeterminerTask.ConfigClass):
    spatialOrder = pexConfig.Field[int](
        doc="specify spatial order for PSF kernel creation",
        default=2,
    )
    samplingSize = pexConfig.Field[float](
        doc="Resolution of the internal PSF model relative to the pixel size; "
        "e.g. 0.5 is equal to 2x oversampling",
        default=1,
    )
    modelSize = pexConfig.Field[int](
        doc="Internal model size for PIFF (typically odd, but not enforced)",
        default=25,
    )
    outlierNSigma = pexConfig.Field[float](
        doc="n sigma for chisq outlier rejection",
        default=4.0
    )
    outlierMaxRemove = pexConfig.Field[float](
        doc="Max fraction of stars to remove as outliers each iteration",
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
    interpolant = pexConfig.Field[str](
        doc="GalSim interpolant name for Piff to use. "
            "Options include 'Lanczos(N)', where N is an integer, along with "
            "galsim.Cubic, galsim.Delta, galsim.Linear, galsim.Nearest, "
            "galsim.Quintic, and galsim.SincInterpolant.",
        check=_validateGalsimInterpolant,
        default="Lanczos(11)",
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
    piffLoggingLevel = pexConfig.RangeField[int](
        doc="PIFF-specific logging level, from 0 (least chatty) to 3 (very verbose); "
            "note that logs will come out with unrelated notations (e.g. WARNING does "
            "not imply a warning.)",
        default=0,
        min=0,
        max=3,
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
        self, exposure, psfCandidateList, metadata=None, flagKey=None, color=None):
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
        color : `ndarray` of `float`, `list`, or `None`, optional
              Color information for each PSF candidate.
        Returns
        -------
        psf : `lsst.meas.extensions.piff.PiffPsf`
           The measured PSF model.
        psfCellSet : `None`
           Unused by this PsfDeterminer.
        """

        # TO DO: Dirty, need to make that better.
        if color is None:
            psfCandidateList = self.downsampleCandidates(psfCandidateList)
        else:
            psfCandidateList, color = self.downsampleCandidates(psfCandidateList, paramsCandidateList=color)

        # TO DO: dummy color, need to remove, here for testing.
        # if color is None:
        #    np.random.seed(42)
        #    color = np.random.normal(0, 1, len(psfCandidateList))

        if color is None:
            use_color = False
        else:
            use_color = True

        if color is None:
            color = np.zeros(len(psfCandidateList))

        if self.config.stampSize:
            stampSize = self.config.stampSize
            if stampSize > psfCandidateList[0].getWidth():
                self.log.warning("stampSize is larger than the PSF candidate size.  Using candidate size.")
                stampSize = psfCandidateList[0].getWidth()
        else:  # TODO: Only the if block should stay after DM-36311
            self.log.debug("stampSize not set.  Using candidate size.")
            stampSize = psfCandidateList[0].getWidth()

        scale = exposure.getWcs().getPixelScale().asArcseconds()

        match self.config.useCoordinates:
            case 'field':
                detector = exposure.getDetector()
                pix_to_field = detector.getTransform(PIXELS, FIELD_ANGLE)
                gswcs = UVWcsWrapper(pix_to_field)
                pointing = None
                # TO DO: Not sure if this working like that right now. Need to check.
                # write unit tests for this.
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
                # TO DO: Not sure if this working like that right now. Need to check.
                # write unit tests for this.
                # need to be ['ra', 'dec'], need to investigate.
                # sure that will work at least with the color key.
                keys = ['u', 'v']

            case 'pixel':
                gswcs = galsim.PixelScale(scale)
                pointing = None
                keys = ['x', 'y']

        if use_color:
            # TO DO: in DES they are using different polynomial order for different keys.
            # Need to implement that.
            keys.append('color')

        orders = [self.config.spatialOrder] * len(keys)

        stars = []
        for candidate, col in zip(psfCandidateList, color):
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
            image_pos = galsim.PositionD(source.getX(), source.getY())

            properties = {}

            if use_color:
                properties['color'] = col

            data = piff.StarData(
                gsImage,
                image_pos,
                weight=gsWeight,
                pointing=pointing,
                properties=properties,
            )
            stars.append(piff.Star(data, None))

        piffConfig = {
            'type': "Simple",
            'model': {
                'type': 'PixelGrid',
                'scale': scale * self.config.samplingSize,
                'size': self.config.modelSize,
                'interp': self.config.interpolant
            },
            'interp': {
                'type': 'BasisPolynomial',
                'order': orders,
                'keys': keys,
            },
            'outliers': {
                'type': 'Chisq',
                'nsigma': self.config.outlierNSigma,
                'max_remove': self.config.outlierMaxRemove
            }
        }

        piffResult = piff.PSF.process(piffConfig)
        wcs = {0: gswcs}

        piffResult.fit(stars, wcs, pointing, logger=self.piffLogger)
        drawSize = 2*np.floor(0.5*stampSize/self.config.samplingSize) + 1

        used_image_pos = [s.image_pos for s in piffResult.stars]
        if flagKey:
            for candidate in psfCandidateList:
                source = candidate.getSource()
                posd = galsim.PositionD(source.getX(), source.getY())
                if posd in used_image_pos:
                    source.set(flagKey, True)

        if metadata is not None:
            metadata["spatialFitChi2"] = piffResult.chisq
            metadata["numAvailStars"] = len(stars)
            metadata["numGoodStars"] = len(piffResult.stars)
            metadata["avgX"] = np.mean([p.x for p in piffResult.stars])
            metadata["avgY"] = np.mean([p.y for p in piffResult.stars])

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
