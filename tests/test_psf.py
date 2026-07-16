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

from pathlib import Path

import galsim  # noqa: F401
import piff
import unittest
import numpy as np
import copy
from galsim import Lanczos  # noqa: F401
import logging

import lsst.utils.tests
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
from lsst.pipe.base import AlgorithmError
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.extensions.piff.piffPsfDeterminer import PiffPsfDeterminerConfig, PiffPsfDeterminerTask
from lsst.meas.extensions.piff.piffPsfDeterminer import _validateGalsimInterpolant
from packaging.version import Version

# PyTorch is part of the Rubin conda environment, so it is always available here.
import torch

# The AIPSF model needs a piff version that provides it.
HAVE_AIPSF = hasattr(piff, "AIPSF")


def psfVal(ix, iy, x, y, sigma1, sigma2, b):
    """Return the value at (ix, iy) of a double Gaussian
       (N(0, sigma1^2) + b*N(0, sigma2^2))/(1 + b)
       centered at (x, y)
    """
    dx, dy = x - ix, y - iy
    theta = np.radians(30)
    ab = 1.0/0.75  # axis ratio
    c, s = np.cos(theta), np.sin(theta)
    u, v = c*dx - s*dy, s*dx + c*dy

    return (np.exp(-0.5*(u**2 + (v*ab)**2)/sigma1**2)
            + b*np.exp(-0.5*(u**2 + (v*ab)**2)/sigma2**2))/(1 + b)


def make_wcs(angle_degrees=None):
    """Make a simple SkyWcs that is rotated around the origin.

    Parameters
    ----------
    angle_degrees : `float`, optional
        The angle to rotate the WCS by, in degrees.

    Returns
    -------
    wcs : `~lsst.afw.geom.SkyWcs`
        The WCS object.
    """
    cdMatrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]) * 0.2 / 3600

    if angle_degrees is not None:
        angle_radians = np.radians(angle_degrees)
        cosang = np.cos(angle_radians)
        sinang = np.sin(angle_radians)
        rot = np.array([
            [cosang, -sinang],
            [sinang, cosang]
        ])
        cdMatrix = np.dot(cdMatrix, rot)

    return afwGeom.makeSkyWcs(
        crpix=geom.PointD(0, 0),
        crval=geom.SpherePoint(0.0, 0.0, geom.degrees),
        cdMatrix=cdMatrix,
    )


class SpatialModelPsfTestCase(lsst.utils.tests.TestCase):
    """A test case for SpatialModelPsf"""

    def measure(self, footprintSet, exposure):
        """Measure a set of Footprints, returning a SourceCatalog"""
        catalog = afwTable.SourceCatalog(self.schema)

        footprintSet.makeSources(catalog)

        self.measureSources.run(catalog, exposure)
        return catalog

    def setUp(self):
        config = SingleFrameMeasurementTask.ConfigClass()
        config.plugins.names = [
            "base_PsfFlux",
            "base_GaussianFlux",
            "base_SdssCentroid",
            "base_SdssShape",
            "base_PixelFlags",
            "base_CircularApertureFlux",
        ]
        config.slots.apFlux = 'base_CircularApertureFlux_12_0'
        self.schema = afwTable.SourceTable.makeMinimalSchema()

        self.measureSources = SingleFrameMeasurementTask(
            self.schema, config=config
        )
        self.usePsfFlag = self.schema.addField("use_psf", type="Flag")

        width, height = 110, 301

        self.mi = afwImage.MaskedImageF(geom.ExtentI(width, height))
        self.mi.set(0)
        sd = 3  # standard deviation of image
        self.mi.getVariance().set(sd*sd)
        self.mi.getMask().addMaskPlane("DETECTED")

        self.ksize = 31  # size of desired kernel

        sigma1 = 1.75
        sigma2 = 2*sigma1

        self.exposure = afwImage.makeExposure(self.mi)
        self.exposure.setPsf(measAlg.DoubleGaussianPsf(self.ksize, self.ksize,
                                                       1.5*sigma1, 1, 0.1))
        wcs = make_wcs()
        self.exposure.setWcs(wcs)

        #
        # Make a kernel with the exactly correct basis functions.
        # Useful for debugging
        #
        basisKernelList = []
        for sigma in (sigma1, sigma2):
            basisKernel = afwMath.AnalyticKernel(
                self.ksize, self.ksize, afwMath.GaussianFunction2D(sigma, sigma)
            )
            basisImage = afwImage.ImageD(basisKernel.getDimensions())
            basisKernel.computeImage(basisImage, True)
            basisImage /= np.sum(basisImage.getArray())

            if sigma == sigma1:
                basisImage0 = basisImage
            else:
                basisImage -= basisImage0

            basisKernelList.append(afwMath.FixedKernel(basisImage))

        order = 1  # 1 => up to linear
        spFunc = afwMath.PolynomialFunction2D(order)

        exactKernel = afwMath.LinearCombinationKernel(basisKernelList, spFunc)
        exactKernel.setSpatialParameters(
            [[1.0, 0, 0],
             [0.0, 0.5*1e-2, 0.2e-2]]
        )

        rand = afwMath.Random()  # make these tests repeatable by setting seed

        im = self.mi.getImage()
        afwMath.randomGaussianImage(im, rand)  # N(0, 1)
        im *= sd                               # N(0, sd^2)

        xarr, yarr = [], []

        for x, y in [(20, 20), (60, 20),
                     (30, 35),
                     (50, 50),
                     (20, 90), (70, 160), (25, 265), (75, 275), (85, 30),
                     (50, 120), (70, 80),
                     (60, 210), (20, 210),
                     ]:
            xarr.append(x)
            yarr.append(y)

        for x, y in zip(xarr, yarr):
            dx = rand.uniform() - 0.5   # random (centered) offsets
            dy = rand.uniform() - 0.5

            k = exactKernel.getSpatialFunction(1)(x, y)
            b = (k*sigma1**2/((1 - k)*sigma2**2))

            flux = 80000*(1 + 0.1*(rand.uniform() - 0.5))
            I0 = flux*(1 + b)/(2*np.pi*(sigma1**2 + b*sigma2**2))
            for iy in range(y - self.ksize//2, y + self.ksize//2 + 1):
                if iy < 0 or iy >= self.mi.getHeight():
                    continue

                for ix in range(x - self.ksize//2, x + self.ksize//2 + 1):
                    if ix < 0 or ix >= self.mi.getWidth():
                        continue

                    II = I0*psfVal(ix, iy, x + dx, y + dy, sigma1, sigma2, b)
                    Isample = rand.poisson(II)
                    self.mi.image[ix, iy, afwImage.LOCAL] += Isample
                    self.mi.variance[ix, iy, afwImage.LOCAL] += II

        bbox = geom.BoxI(geom.PointI(0, 0), geom.ExtentI(width, height))
        self.cellSet = afwMath.SpatialCellSet(bbox, 100)

        self.footprintSet = afwDetection.FootprintSet(
            self.mi, afwDetection.Threshold(100), "DETECTED"
        )

        self.catalog = self.measure(self.footprintSet, self.exposure)

        for source in self.catalog:
            cand = measAlg.makePsfCandidate(source, self.exposure)
            self.cellSet.insertCandidate(cand)

    def setupDeterminer(
        self,
        stampSize=None,
        kernelSize=None,
        modelSize=25,
        debugStarData=False,
        useCoordinates='pixel',
        spatialOrder=1,
        zerothOrderInterpNotEnoughStars=False,
        piffPsfConfigYaml=None,
        downsample=False,
        useColor=False,
        colorOrder=0,
        withlog=False,
        modelType="pixelGrid",
        aipsfModelFile=None,
        writeTrainingSet=False,
        trainingSetLocation=None,
        maxStarsPerDetector=None,
    ):
        """Setup the starSelector and psfDeterminer

        Parameters
        ----------
        stampSize : `int`, optional
            Set ``config.stampSize`` to this, if not None.
        kernelSize : `int`, optional
            Cutout size for the PSF candidates. This is unused if ``stampSize``
            if provided and its value is used for cutout size instead.
        modelSize : `int`, optional
            Internal model size for PIFF.
        debugStarData : `bool`, optional
            Include star images used for fitting in PSF model object?
        useCoordinates : `str`, optional
            Spatial coordinates to regress against for PSF modelling.
        spatialOrder : `int`, optional
            Spatial order for PSF parameter interpolation.
        zerothOrderInterpNotEnoughStars : `bool`, optional
            If True, use zeroth order interpolation if not enough star.
        piffPsfConfigYaml : `str`, optional
            Configuration file for PIFF in YAML format.
        downsample : `bool`, optional
            Whether to downsample the PSF candidates before modelling?
        withlog : `bool`, optional
            Should Piff produce chatty log messages?
        modelType : `str`, optional
            Piff model to use: "pixelGrid" or "aipsf".
        aipsfModelFile : `str`, optional
            Path to the trained AIPSF checkpoint file; only used if
            ``modelType`` is "aipsf".
        writeTrainingSet : `bool`, optional
            Write a PSF training sample after the fit?
        trainingSetLocation : `str`, optional
            Directory for the training sample pickle files.
        maxStarsPerDetector : `int`, optional
            Cap on the number of stars in the training sample.
        """
        starSelectorClass = measAlg.sourceSelectorRegistry["objectSize"]
        starSelectorConfig = starSelectorClass.ConfigClass()
        starSelectorConfig.sourceFluxField = "base_GaussianFlux_instFlux"
        starSelectorConfig.badFlags = [
            "base_PixelFlags_flag_edge",
            "base_PixelFlags_flag_interpolatedCenter",
            "base_PixelFlags_flag_saturatedCenter",
            "base_PixelFlags_flag_crCenter",
        ]
        # Set to match when the tolerance of the test was set
        starSelectorConfig.widthStdAllowed = 0.5

        self.starSelector = starSelectorClass(config=starSelectorConfig)

        makePsfCandidatesConfig = measAlg.MakePsfCandidatesTask.ConfigClass()
        if kernelSize:
            makePsfCandidatesConfig.kernelSize = kernelSize
        if stampSize is not None:
            makePsfCandidatesConfig.kernelSize = stampSize

        self.makePsfCandidates = measAlg.MakePsfCandidatesTask(config=makePsfCandidatesConfig)

        psfDeterminerConfig = PiffPsfDeterminerConfig()
        psfDeterminerConfig.spatialOrder = spatialOrder
        psfDeterminerConfig.zerothOrderInterpNotEnoughStars = zerothOrderInterpNotEnoughStars
        psfDeterminerConfig.stampSize = stampSize
        psfDeterminerConfig.modelSize = modelSize

        psfDeterminerConfig.debugStarData = debugStarData
        psfDeterminerConfig.useCoordinates = useCoordinates
        psfDeterminerConfig.piffPsfConfigYaml = piffPsfConfigYaml

        psfDeterminerConfig.colorOrder = colorOrder
        psfDeterminerConfig.useColor = useColor

        psfDeterminerConfig.modelType = modelType
        psfDeterminerConfig.aipsfModelFile = aipsfModelFile

        psfDeterminerConfig.writeTrainingSet = writeTrainingSet
        if trainingSetLocation is not None:
            psfDeterminerConfig.trainingSample.trainingSetLocation = trainingSetLocation
        psfDeterminerConfig.trainingSample.maxStarsPerDetector = maxStarsPerDetector

        if piffPsfConfigYaml is None:
            self.useYaml = False
        else:
            self.useYaml = True

        if downsample:
            psfDeterminerConfig.maxCandidates = 10
        if withlog:
            psfDeterminerConfig.piffLoggingLevel = 1

        self.psfDeterminer = PiffPsfDeterminerTask(psfDeterminerConfig)

    def subtractStars(self, exposure, catalog, chi_lim=-1.):
        """Subtract the exposure's PSF from all the sources in catalog"""
        mi, psf = exposure.getMaskedImage(), exposure.getPsf()

        subtracted = mi.Factory(mi, True)
        for s in catalog:
            xc, yc = s.getX(), s.getY()
            bbox = subtracted.getBBox(afwImage.PARENT)
            if bbox.contains(geom.PointI(int(xc), int(yc))):
                measAlg.subtractPsf(psf, subtracted, xc, yc)
        chi = subtracted.Factory(subtracted, True)
        var = subtracted.getVariance()
        np.sqrt(var.getArray(), var.getArray())  # inplace sqrt
        chi /= var

        chi_min = np.min(chi.getImage().getArray())
        chi_max = np.max(chi.getImage().getArray())

        if chi_lim > 0:
            self.assertGreater(chi_min, -chi_lim)
            self.assertLess(chi_max, chi_lim)

    def checkPiffDeterminer(self, **kwargs):
        """Configure PiffPsfDeterminerTask and run basic tests on it.

        Parameters
        ----------
        kwargs : `dict`, optional
            Additional keyword arguments to pass to setupDeterminer.
        """
        self.setupDeterminer(**kwargs)
        metadata = dafBase.PropertyList()

        stars = self.starSelector.run(self.catalog, exposure=self.exposure)
        psfCandidateList = self.makePsfCandidates.run(
            stars.sourceCat,
            exposure=self.exposure
        ).psfCandidates

        for psf in psfCandidateList:
            psf.setPsfColorValue(0.42)
            psf.setPsfColorType("g-r")

        logger = logging.getLogger("lsst.psfDeterminer.Piff")

        if Version(piff.version) >= Version("1.6"):
            log_level = logging.INFO
            log_regex = "INFO:.*:Iteration"
        else:
            log_level = logging.WARNING
            log_regex = "WARNING:.*:Iteration"

        with self.assertLogs("lsst.psfDeterminer.Piff.piff", log_level) as cm:
            if kwargs.get("zerothOrderInterpNotEnoughStars", False):
                psf, cellSet = self.psfDeterminer.determinePsf(
                    self.exposure,
                    psfCandidateList,
                    metadata,
                    flagKey=self.usePsfFlag
                )
            else:
                with self.assertNoLogs("lsst.psfDeterminer.Piff", logging.WARNING):
                    psf, cellSet = self.psfDeterminer.determinePsf(
                        self.exposure,
                        psfCandidateList,
                        metadata,
                        flagKey=self.usePsfFlag
                    )

        # Check that the iterations are being logged.
        logged = "\n".join(cm.output)
        self.assertRegex(logged, log_regex)

        # And check that the levels are set correctly for suppression.
        logger = logging.getLogger("lsst.psfDeterminer.Piff.piff")
        if kwargs.get("withlog", False):
            self.assertEqual(logger.level, logging.WARNING)
        else:
            self.assertEqual(logger.level, logging.CRITICAL)

        self.exposure.setPsf(psf)

        if kwargs.get("downsample", False):
            # When downsampling the PSF model is not quite as
            # good so the chi2 test limit needs to be modified.
            numAvail = self.psfDeterminer.config.maxCandidates
            chiLim = 7.0
        elif kwargs.get("zerothOrderInterpNotEnoughStars", False):
            numAvail = len(psfCandidateList)
            chiLim = 20.31
        else:
            numAvail = len(psfCandidateList)
            chiLim = 6.4

        self.assertEqual(metadata['numAvailStars'], numAvail)
        self.assertEqual(sum(self.catalog['use_psf']), metadata['numGoodStars'])
        self.assertLessEqual(metadata['numGoodStars'], metadata['numAvailStars'])

        self.assertEqual(
            psf.getAveragePosition(),
            geom.Point2D(
                np.mean([s.x for s in psf._piffResult.stars
                         if not s.is_flagged and not s.is_reserve]),
                np.mean([s.y for s in psf._piffResult.stars
                         if not s.is_flagged and not s.is_reserve])
            )
        )
        if self.psfDeterminer.config.debugStarData:
            self.assertIn('image', psf._piffResult.stars[0].data.__dict__)
        else:
            self.assertNotIn('image', psf._piffResult.stars[0].data.__dict__)

        # Test how well we can subtract the PSF model
        self.subtractStars(self.exposure, self.catalog, chi_lim=chiLim)

        # Test bboxes
        for point in [
            psf.getAveragePosition(),
            geom.Point2D(),
            geom.Point2D(1, 1)
        ]:
            self.assertEqual(
                psf.computeBBox(point),
                psf.computeKernelImage(point).getBBox()
            )
            self.assertEqual(
                psf.computeKernelBBox(point),
                psf.computeKernelImage(point).getBBox()
            )
            self.assertEqual(
                psf.computeImageBBox(point),
                psf.computeImage(point).getBBox()
            )

        # Some roundtrips
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self.exposure.writeFits(tmpFile)
            fitsIm = afwImage.ExposureF(tmpFile)
            copyIm = copy.deepcopy(self.exposure)

            for newIm in [fitsIm, copyIm]:
                # Piff doesn't enable __eq__ for its results, so we just check
                # that some PSF images come out the same.
                for point in [
                    geom.Point2D(0, 0),
                    geom.Point2D(10, 100),
                    geom.Point2D(-200, 30),
                    geom.Point2D(float("nan"))  # "nullPoint"
                ]:
                    self.assertImagesAlmostEqual(
                        psf.computeImage(point),
                        newIm.getPsf().computeImage(point)
                    )
                # Also check average position
                newPsf = newIm.getPsf()
                self.assertImagesAlmostEqual(
                    psf.computeImage(psf.getAveragePosition()),
                    newPsf.computeImage(newPsf.getAveragePosition())
                )

    def testReadOldPiffVersions(self):
        """Test we can read psfs serialized with older versions of Piff."""
        point_names = [
            (geom.Point2D(0, 0), "psfIm_0_0.fits"),
            (geom.Point2D(10, 100), "psfIm_10_100.fits"),
            (geom.Point2D(-200, 30), "psfIm_-200_30.fits"),
            (geom.Point2D(float("nan")), "psfIm_nan.fits"),
        ]

        if False:  # Documenting block that created the test data...
            # Specifically interested in the case where the PSF is created with
            # piff older than v1.4
            assert Version(piff.version) < Version("1.4")

            self.setupDeterminer()
            stars = self.starSelector.run(self.catalog, exposure=self.exposure)
            psfCandidateList = self.makePsfCandidates.run(
                stars.sourceCat,
                exposure=self.exposure
            ).psfCandidates
            psf, _ = self.psfDeterminer.determinePsf(
                self.exposure,
                psfCandidateList,
            )
            self.exposure.setPsf(psf)

            path = Path(__file__).parent / "data" / "exp.fits"
            self.exposure.writeFits(str(path))
            for point, name in point_names:
                psfIm = psf.computeImage(point)
                psfIm.writeFits(str(path.with_name(name)))

        path = Path(__file__).parent / "data" / "exp.fits"
        exposure = afwImage.ExposureF(str(path))
        for point, name in point_names:
            self.assertImagesAlmostEqual(
                afwImage.ImageF(str(path.with_name(name))),
                exposure.getPsf().computeImage(point)
            )

    def testPiffDeterminer_default(self):
        """Test piff with the default config."""
        self.checkPiffDeterminer()

    def testPiffDeterminer_stampSize27(self):
        """Test Piff with a psf stampSize of 27."""
        self.checkPiffDeterminer(stampSize=27)
        self.assertEqual(
            self.exposure.psf.computeKernelImage(self.exposure.getBBox().getCenter()).getDimensions(),
            geom.Extent2I(27, 27),
        )

    def testPiffDeterminer_debugStarData(self):
        """Test Piff with debugStarData=True."""
        self.checkPiffDeterminer(debugStarData=True)

    def testPiffDeterminer_downsample(self):
        """Test Piff determiner with downsampling."""
        self.checkPiffDeterminer(downsample=True)

    def testPiffDeterminer_withlog(self):
        """Test Piff determiner with chatty logs."""
        self.checkPiffDeterminer(withlog=True)

    def testPiffDeterminer_stampSize26(self):
        """Test Piff with a psf stampSize of 26."""
        with self.assertRaises(ValueError):
            self.checkPiffDeterminer(stampSize=26)

    def testPiffDeterminer_modelSize26(self):
        """Test Piff with a psf stampSize of 26."""
        with self.assertRaises(ValueError):
            self.checkPiffDeterminer(modelSize=26, stampSize=25)

    def testPiffDeterminer_skyCoords(self):
        """Test Piff sky coords."""

        self.checkPiffDeterminer(useCoordinates='sky')

    @lsst.utils.tests.methodParameters(angle_degrees=[0, 35, 45, 77, 135])
    def testPiffDeterminer_skyCoords_with_rotation(self, angle_degrees):
        """Test Piff sky coords with rotation."""

        wcs = make_wcs(angle_degrees=angle_degrees)
        self.exposure.setWcs(wcs)
        self.checkPiffDeterminer(useCoordinates='sky', kernelSize=35)

    def testPiffDeterminer_skyCoords_failure(self, angle_degrees=135):
        """Test that using small PSF candidates with sky coordinates fails."""
        wcs = make_wcs(angle_degrees=angle_degrees)
        self.exposure.setWcs(wcs)
        with self.assertRaises(ValueError):
            self.checkPiffDeterminer(useCoordinates='sky', stampSize=15)

    def testPiffZerothOrderInterpNotEnoughStars(self):
        self.checkPiffDeterminer(spatialOrder=4, zerothOrderInterpNotEnoughStars=True)
        if not self.useYaml:
            self.assertEqual(self.psfDeterminer._piffConfig['interp']['order'], [0, 0])
            self.assertEqual(self.psfDeterminer._piffConfig['max_iter'], 1)
        else:
            # Yaml will overwrite input value.
            self.assertEqual(self.psfDeterminer._piffConfig['interp']['order'], 1)
            self.assertNotIn('max_iter', self.psfDeterminer._piffConfig)

    def testPiffRaiseErrorNotEnoughStars(self):
        with self.assertRaises(AlgorithmError):
            self.checkPiffDeterminer(spatialOrder=42,
                                     zerothOrderInterpNotEnoughStars=False,
                                     piffPsfConfigYaml=None)

    def testPiffDummyColorfit(self):
        self.checkPiffDeterminer(useColor=True,
                                 colorOrder=0,
                                 piffPsfConfigYaml=None)

    def testPiffDeterminer_trainingSample(self):
        """Test that the trainingSample subtask writes the expected pickle file."""
        import os
        import pickle
        import tempfile
        import lsst.afw.cameraGeom as cameraGeom
        from lsst.afw.cameraGeom.testUtils import DetectorWrapper

        # The synthetic test exposure has no detector, visit info, or filter,
        # which the training sample needs for the focal-plane transform and
        # the output identifiers.
        detector = DetectorWrapper().detector
        self.exposure.setDetector(detector)
        self.exposure.info.setVisitInfo(afwImage.VisitInfo(id=1234))
        self.exposure.setFilter(afwImage.FilterLabel(band="r", physical="r_test"))

        stampSize = 25
        with tempfile.TemporaryDirectory() as trainingSetLocation:
            self.setupDeterminer(
                stampSize=stampSize,
                writeTrainingSet=True,
                trainingSetLocation=trainingSetLocation,
            )
            metadata = dafBase.PropertyList()

            stars = self.starSelector.run(self.catalog, exposure=self.exposure)
            psfCandidateList = self.makePsfCandidates.run(
                stars.sourceCat,
                exposure=self.exposure
            ).psfCandidates

            psf, cellSet = self.psfDeterminer.determinePsf(
                self.exposure,
                psfCandidateList,
                metadata,
                flagKey=self.usePsfFlag
            )

            fileName = os.path.join(
                trainingSetLocation, f"1234_{detector.getId()}_r.pkl"
            )
            self.assertTrue(os.path.exists(fileName))
            with open(fileName, "rb") as f:
                trainingSample = pickle.load(f)

            self.assertEqual(len(trainingSample), metadata['numGoodStars'])
            pixelsToFocal = detector.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
            for starId, record in trainingSample.items():
                self.assertTrue(starId.startswith(f"1234_{detector.getId()}_r_"))
                self.assertEqual(
                    set(record.keys()),
                    {"star", "weight", "starPiff", "xCCD", "yCCD", "xFoV", "yFoV",
                     "sumStar", "detector", "visit", "band"},
                )
                self.assertEqual(record["star"].shape, (stampSize, stampSize))
                self.assertEqual(record["starPiff"].shape, (stampSize, stampSize))
                self.assertEqual(record["weight"].shape, (stampSize, stampSize))
                self.assertEqual(record["star"].dtype, np.float32)
                self.assertEqual(record["starPiff"].dtype, np.float32)
                self.assertEqual(record["weight"].dtype, np.float32)
                self.assertFloatsAlmostEqual(record["star"].sum(), 1.0, rtol=1e-5)
                # The weight map is an inverse variance: finite, non-negative,
                # and not all zero.
                self.assertTrue(np.all(np.isfinite(record["weight"])))
                self.assertTrue(np.all(record["weight"] >= 0))
                self.assertGreater(np.sum(record["weight"] > 0), 0)
                self.assertEqual(record["detector"], detector.getId())
                self.assertEqual(record["visit"], 1234)
                self.assertEqual(record["band"], "r")
                focalPoint = pixelsToFocal.applyForward(
                    geom.Point2D(record["xCCD"], record["yCCD"])
                )
                self.assertFloatsAlmostEqual(record["xFoV"], focalPoint.getX(), rtol=1e-12)
                self.assertFloatsAlmostEqual(record["yFoV"], focalPoint.getY(), rtol=1e-12)

            # The determiner still returns a valid PSF; writing the training
            # sample is not exclusive with a science fit.
            self.exposure.setPsf(psf)
            image = psf.computeKernelImage(self.exposure.getBBox().getCenter())
            self.assertTrue(np.all(np.isfinite(image.array)))

            # With maxStarsPerDetector set, a random subset of the good stars
            # is kept.
            uncappedKeys = set(trainingSample.keys())
            cap = 3
            self.assertGreater(len(uncappedKeys), cap)
            self.setupDeterminer(
                stampSize=stampSize,
                writeTrainingSet=True,
                trainingSetLocation=trainingSetLocation,
                maxStarsPerDetector=cap,
            )
            stars = self.starSelector.run(self.catalog, exposure=self.exposure)
            psfCandidateList = self.makePsfCandidates.run(
                stars.sourceCat,
                exposure=self.exposure
            ).psfCandidates
            self.psfDeterminer.determinePsf(
                self.exposure,
                psfCandidateList,
                dafBase.PropertyList(),
                flagKey=self.usePsfFlag
            )
            with open(fileName, "rb") as f:
                cappedSample = pickle.load(f)
            self.assertEqual(len(cappedSample), cap)
            # The capped selection is a subset of the uncapped one (the same
            # stars are good in both fits).
            self.assertLessEqual(set(cappedSample.keys()), uncappedKeys)

    @unittest.skipUnless(HAVE_AIPSF, "this piff version does not provide AIPSF")
    def testPiffDeterminer_aipsf(self):
        """Test the AIPSF model path with a small random-weight network.

        The network weights are random, so this checks the mechanics of the
        fit (latent encoding, Polynomial interpolation of the latent space,
        drawing, and persistence), not the quality of the PSF model.
        """
        with lsst.utils.tests.getTempFilePath(".pth") as modelFile:
            torch.manual_seed(1234)
            net = piff.aimodels.Conv2dAutoEncoder(grid_size=25, latent_dim=4, hidden_channels=2)
            net.eval()
            piff.aimodels.save_checkpoint(net, modelFile)

            self.setupDeterminer(
                stampSize=25,
                modelSize=25,
                spatialOrder=1,
                modelType="aipsf",
                aipsfModelFile=modelFile,
            )
            metadata = dafBase.PropertyList()

            stars = self.starSelector.run(self.catalog, exposure=self.exposure)
            psfCandidateList = self.makePsfCandidates.run(
                stars.sourceCat,
                exposure=self.exposure
            ).psfCandidates

            psf, cellSet = self.psfDeterminer.determinePsf(
                self.exposure,
                psfCandidateList,
                metadata,
                flagKey=self.usePsfFlag
            )

            self.assertEqual(self.psfDeterminer._piffConfig['model']['type'], 'AIPSF')
            self.assertEqual(self.psfDeterminer._piffConfig['model']['background_fit_mode'],
                             'free')
            self.assertEqual(self.psfDeterminer._piffConfig['interp']['type'], 'Polynomial')

            # For AIPSF, the latent encodings (fit.params) are kept through the
            # star-data cleanup, and fit() stored the per-star (a, b) nuisance
            # parameters; the large data objects are still removed.
            usedStars = [s for s in psf._piffResult.stars
                         if not s.is_flagged and not s.is_reserve]
            self.assertGreater(len(usedStars), 0)
            for s in usedStars:
                self.assertEqual(len(s.fit.params), 4)
                self.assertTrue(np.all(np.isfinite(s.fit.params)))
                self.assertIn('aipsf_a', s.data.properties)
                self.assertIn('aipsf_b', s.data.properties)
            self.assertNotIn('image', psf._piffResult.stars[0].data.__dict__)

            self.exposure.setPsf(psf)

            # For the AIPSF model the drawn kernel has the stamp size.
            self.assertEqual(
                psf.computeKernelImage(self.exposure.getBBox().getCenter()).getDimensions(),
                geom.Extent2I(25, 25),
            )

            self.assertEqual(metadata['numAvailStars'], len(psfCandidateList))
            self.assertLessEqual(metadata['numGoodStars'], metadata['numAvailStars'])
            self.assertEqual(
                psf.getAveragePosition(),
                geom.Point2D(
                    np.mean([s.x for s in psf._piffResult.stars
                             if not s.is_flagged and not s.is_reserve]),
                    np.mean([s.y for s in psf._piffResult.stars
                             if not s.is_flagged and not s.is_reserve])
                )
            )

            # The PSF images are all finite and normalized-ish.
            for point in [
                psf.getAveragePosition(),
                geom.Point2D(),
                geom.Point2D(1, 1)
            ]:
                image = psf.computeKernelImage(point)
                self.assertTrue(np.all(np.isfinite(image.array)))
                self.assertEqual(
                    psf.computeBBox(point),
                    image.getBBox()
                )

            # Persistence roundtrips.
            with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
                self.exposure.writeFits(tmpFile)
                fitsIm = afwImage.ExposureF(tmpFile)
                copyIm = copy.deepcopy(self.exposure)

                for newIm in [fitsIm, copyIm]:
                    for point in [
                        geom.Point2D(0, 0),
                        geom.Point2D(10, 100),
                        geom.Point2D(-200, 30),
                        geom.Point2D(float("nan"))  # "nullPoint"
                    ]:
                        self.assertImagesAlmostEqual(
                            psf.computeImage(point),
                            newIm.getPsf().computeImage(point)
                        )


class piffPsfConfigYamlTestCase(SpatialModelPsfTestCase):
    """A test case to trigger the codepath that uses piffPsfConfigYaml."""

    def checkPiffDeterminer(self, **kwargs):
        # Docstring inherited.
        if "piffPsfConfigYaml" not in kwargs:
            piffPsfConfigYaml = """
                # A minimal Piff config corresponding to the defaults.
                type: Simple
                model:
                  type: PixelGrid
                  scale: 0.2
                  size: 25
                  interp: Lanczos(11)
                interp:
                  type: BasisPolynomial
                  order: 1
                outliers:
                  type: Chisq
                  nsigma: 4.0
                  max_remove: 0.05
                """
            kwargs["piffPsfConfigYaml"] = piffPsfConfigYaml
        return super().checkPiffDeterminer(**kwargs)


class PiffConfigTestCase(lsst.utils.tests.TestCase):
    """A test case to check for valid Piff config"""
    def testValidateGalsimInterpolant(self):
        # Check that random strings are not valid interpolants.
        self.assertFalse(_validateGalsimInterpolant("foo"))
        # Check that the Lanczos order is an integer
        self.assertFalse(_validateGalsimInterpolant("Lanczos(3.0"))
        self.assertFalse(_validateGalsimInterpolant("Lanczos(-5.0"))
        self.assertFalse(_validateGalsimInterpolant("Lanczos(N)"))
        # Check for various valid Lanczos interpolants
        for interp in ("Lanczos(4)", "galsim.Lanczos(7)"):
            self.assertTrue(_validateGalsimInterpolant(interp))
            self.assertFalse(_validateGalsimInterpolant(interp.lower()))
            # Evaluating the string should succeed. This is how Piff does it.
            self.assertTrue(eval(interp))
        # Check that interpolation methods are case sensitive.
        for interp in ("Linear", "Cubic", "Quintic", "Delta", "Nearest", "SincInterpolant"):
            self.assertFalse(_validateGalsimInterpolant(f"galsim.{interp.lower()}"))
            self.assertFalse(_validateGalsimInterpolant(interp))
            self.assertTrue(_validateGalsimInterpolant(f"galsim.{interp}"))
            self.assertTrue(eval(f"galsim.{interp}"))

    def _makeAipsfConfig(self):
        """Make a config with a valid aipsf setup."""
        config = PiffPsfDeterminerConfig()
        config.modelType = "aipsf"
        config.aipsfModelFile = "some_checkpoint.pth"
        config.stampSize = 25
        config.modelSize = 25
        return config

    def testValidateAipsfConfig(self):
        # A consistent aipsf config validates.  (The checkpoint file existence
        # is only checked at runtime, so no torch is needed here.)
        config = self._makeAipsfConfig()
        config.validate()

        # aipsfModelFile is required with modelType='aipsf'.
        config = self._makeAipsfConfig()
        config.aipsfModelFile = None
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        # The stamp size must equal the model size.
        config = self._makeAipsfConfig()
        config.stampSize = 27
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        # No internal resampling with aipsf.
        config = self._makeAipsfConfig()
        config.samplingSize = 0.5
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        # Color is not supported with aipsf (the Polynomial latent interpolation
        # cannot regress against color).
        config = self._makeAipsfConfig()
        config.useColor = True
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        # Only pixel coordinates are supported with aipsf (the autoencoder
        # operates on raw pixel stamps).
        for useCoordinates in ("field", "sky"):
            config = self._makeAipsfConfig()
            config.useCoordinates = useCoordinates
            with self.assertRaises(pexConfig.FieldValidationError):
                config.validate()

        # The aipsf constraints do not apply when a piffPsfConfigYaml overrides
        # the model configuration.
        config = self._makeAipsfConfig()
        config.aipsfModelFile = None
        config.piffPsfConfigYaml = "{type: Simple}"
        config.validate()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
