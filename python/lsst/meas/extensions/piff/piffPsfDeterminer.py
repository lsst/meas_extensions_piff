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

import lsst.pex.config as pexConfig
import lsst.meas.algorithms as measAlg
from lsst.meas.algorithms.psfDeterminer import BasePsfDeterminerTask
from .piffPsf import PiffPsf


class PiffPsfDeterminerConfig(BasePsfDeterminerTask.ConfigClass):
    spatialOrder = pexConfig.Field(
        doc="specify spatial order for PSF kernel creation",
        dtype=int,
        default=2,
    )
    samplingSize = pexConfig.Field(
        doc="Resolution of the internal PSF model relative to the pixel size; "
        "e.g. 0.5 is equal to 2x oversampling",
        dtype=float,
        default=1,
    )
    outlierNSigma = pexConfig.Field(
        doc="n sigma for chisq outlier rejection",
        dtype=float,
        default=4.0
    )
    outlierMaxRemove = pexConfig.Field(
        doc="Max fraction of stars to remove as outliers each iteration",
        dtype=float,
        default=0.05
    )
    maxSNR = pexConfig.Field(
        doc="Rescale the weight of bright stars such that their SNR is less "
            "than this value.",
        dtype=float,
        default=200.0
    )
    zeroWeightMaskBits = pexConfig.ListField(
        doc="List of mask bits for which to set pixel weights to zero.",
        dtype=str,
        default=['BAD', 'CR', 'INTRP', 'SAT', 'SUSPECT', 'NO_DATA']
    )
    minimumUnmaskedFraction = pexConfig.Field(
        doc="Minimum fraction of unmasked pixels required to use star.",
        dtype=float,
        default=0.5
    )

    def setDefaults(self):
        self.kernelSize = 21
        self.kernelSizeMin = 11
        self.kernelSizeMax = 35


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

    def determinePsf(
        self, exposure, psfCandidateList, metadata=None, flagKey=None
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
        stars = []
        for candidate in psfCandidateList:
            cmi = candidate.getMaskedImage()
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
            gsImage = galsim.Image(bds, scale=1.0, dtype=float)
            gsImage.array[:] = cmi.image.array
            gsWeight = galsim.Image(bds, scale=1.0, dtype=float)
            gsWeight.array[:] = weight

            source = candidate.getSource()
            image_pos = galsim.PositionD(source.getX(), source.getY())

            data = piff.StarData(
                gsImage,
                image_pos,
                weight=gsWeight
            )
            stars.append(piff.Star(data, None))

        kernelSize = int(np.clip(
            self.config.kernelSize,
            self.config.kernelSizeMin,
            self.config.kernelSizeMax
        ))

        piffConfig = {
            'type': "Simple",
            'model': {
                'type': 'PixelGrid',
                'scale': self.config.samplingSize,
                'size': kernelSize
            },
            'interp': {
                'type': 'BasisPolynomial',
                'order': self.config.spatialOrder
            },
            'outliers': {
                'type': 'Chisq',
                'nsigma': self.config.outlierNSigma,
                'max_remove': self.config.outlierMaxRemove
            }
        }

        piffResult = piff.PSF.process(piffConfig)
        # Run on a single CCD, and in image coords rather than sky coords.
        wcs = {0: galsim.PixelScale(1.0)}
        pointing = None

        piffResult.fit(stars, wcs, pointing, logger=self.log)
        psf = PiffPsf(kernelSize, kernelSize, piffResult)

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

        return psf, None


measAlg.psfDeterminerRegistry.register("piff", PiffPsfDeterminerTask)
