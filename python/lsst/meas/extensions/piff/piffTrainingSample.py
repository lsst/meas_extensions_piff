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

__all__ = ["PiffTrainingSampleConfig", "PiffTrainingSampleTask"]

import os
import pickle

import numpy as np

from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from lsst.geom import Point2D
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class PiffTrainingSampleConfig(pexConfig.Config):
    trainingSetLocation = pexConfig.Field[str](
        doc="Directory to which the per-(visit, detector, band) training sample "
        "pickle files are written.",
        default=".",
    )


class PiffTrainingSampleTask(pipeBase.Task):
    """Write a PSF training sample from a fitted Piff result.

    This task is meant to be run as a subtask of
    `~lsst.meas.extensions.piff.PiffPsfDeterminerTask` (enabled with its
    ``writeTrainingSet`` config option).  For each star used in the PSF fit,
    it saves the flux-normalized postage stamp together with the fitted PSF
    model prediction at the star position and the star coordinates in both
    pixel and focal-plane coordinates.  The collection is written as one
    pickle file per (visit, detector, band):
    ``{trainingSetLocation}/{visit}_{detector}_{band}.pkl``.

    These pickle files are the training inputs of the Piff ``trainify``
    executable, which trains the autoencoder used by the Piff AIPSF model.

    The pickle file contains a `dict` indexed by
    ``"{visit}_{detector}_{band}_{starId}"``, with one record per star::

        {
            "star":     numpy.float32 (N, N), stamp normalized to sum to 1,
            "weight":   None (reserved for future use),
            "starPiff": numpy.float32 (N, N), fitted PSF model drawn at the
                        star position,
            "xCCD", "yCCD": star position in pixel coordinates,
            "xFoV", "yFoV": star position in focal-plane coordinates (mm),
            "sumStar":  original stamp flux (the normalization factor),
            "detector": detector id,
            "visit":    visit id,
            "band":     band label,
        }
    """
    ConfigClass = PiffTrainingSampleConfig
    _DefaultName = "piffTrainingSample"

    def run(self, piffResult, exposure, drawSize):
        """Extract the training sample from a Piff fit and write it to disk.

        Parameters
        ----------
        piffResult : `piff.PSF`
            The fitted Piff PSF, with its ``stars`` attribute still holding
            the star data (i.e. before any cleanup of the star images).
        exposure : `lsst.afw.image.Exposure`
            The exposure on which the PSF was fit; used for the detector
            transform to focal-plane coordinates and the visit/detector/band
            identifiers.
        drawSize : `int`
            Stamp size (pixels) with which to draw the PSF model predictions.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Struct with components:

            ``numStars`` : `int`
                Number of stars written to the training sample.
            ``fileName`` : `str`
                Name of the pickle file written.
        """
        detector = exposure.getDetector()
        pixelsToFocal = detector.getTransform(PIXELS, FOCAL_PLANE)

        detectorId = detector.getId()
        visitId = exposure.getInfo().getVisitInfo().id
        bandId = exposure.getInfo().getFilter().bandLabel

        trainingSample = {}
        for star in piffResult.stars:
            if star.is_flagged or star.is_reserve:
                continue

            sumStar = np.sum(star.data.image.array)
            if not np.isfinite(sumStar) or sumStar <= 0:
                self.log.warning(
                    "Skipping star %s at (%.1f, %.1f): non-finite or non-positive "
                    "stamp flux (%s).",
                    star.data.properties['starId'], star.x, star.y, sumStar,
                )
                continue

            starId = f"{visitId}_{detectorId}_{bandId}_{star.data.properties['starId']}"
            starPiff = piffResult.draw(star.x, star.y, stamp_size=drawSize, center=None)
            focalPoint = pixelsToFocal.applyForward(Point2D(star.x, star.y))

            trainingSample[starId] = {
                "star": (star.data.image.array / sumStar).astype(np.float32),
                "weight": None,
                "starPiff": starPiff.array.astype(np.float32),
                "xCCD": star.x,
                "yCCD": star.y,
                "xFoV": focalPoint.getX(),
                "yFoV": focalPoint.getY(),
                "sumStar": sumStar,
                "detector": detectorId,
                "visit": visitId,
                "band": bandId,
            }

        fileName = os.path.join(
            self.config.trainingSetLocation, f"{visitId}_{detectorId}_{bandId}.pkl"
        )
        with open(fileName, "wb") as f:
            pickle.dump(trainingSample, f)
        self.log.info("Wrote %d PSF training stars to %s.", len(trainingSample), fileName)

        return pipeBase.Struct(numStars=len(trainingSample), fileName=fileName)
