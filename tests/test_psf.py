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

import galsim  # noqa: F401
import unittest
import numpy as np
import copy
from galsim import Lanczos  # noqa: F401

import lsst.utils.tests
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.extensions.piff.piffPsfDeterminer import PiffPsfDeterminerConfig, PiffPsfDeterminerTask
from lsst.meas.extensions.piff.piffPsfDeterminer import _validateGalsimInterpolant


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
        cdMatrix = np.array([1.0, 0.0, 0.0, 1.0]) * 0.2/3600
        cdMatrix.shape = (2, 2)
        wcs = afwGeom.makeSkyWcs(crpix=geom.PointD(0, 0),
                                 crval=geom.SpherePoint(0.0, 0.0, geom.degrees),
                                 cdMatrix=cdMatrix)
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
        debugStarData=False,
        useCoordinates='pixel',
        downsample=False,
    ):
        """Setup the starSelector and psfDeterminer

        Parameters
        ----------
        stampSize : `int`, optional
            Set ``config.stampSize`` to this, if not None.
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
        if stampSize is not None:
            makePsfCandidatesConfig.kernelSize = stampSize
        self.makePsfCandidates = measAlg.MakePsfCandidatesTask(config=makePsfCandidatesConfig)

        psfDeterminerConfig = PiffPsfDeterminerConfig()
        psfDeterminerConfig.spatialOrder = 1
        if stampSize is not None:
            psfDeterminerConfig.stampSize = stampSize
        psfDeterminerConfig.debugStarData = debugStarData
        psfDeterminerConfig.useCoordinates = useCoordinates
        if downsample:
            psfDeterminerConfig.maxCandidates = 10

        self.psfDeterminer = PiffPsfDeterminerTask(psfDeterminerConfig)

    def subtractStars(self, exposure, catalog, chi_lim=-1):
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
        print(chi_min, chi_max)

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
        psf, cellSet = self.psfDeterminer.determinePsf(
            self.exposure,
            psfCandidateList,
            metadata,
            flagKey=self.usePsfFlag
        )
        self.exposure.setPsf(psf)

        if kwargs.get("downsample", False):
            # When downsampling the PSF model is not quite as
            # good so the chi2 test limit needs to be modified.
            numAvail = self.psfDeterminer.config.maxCandidates
            chiLim = 7.0
        else:
            numAvail = len(psfCandidateList)
            chiLim = 6.1

        self.assertEqual(metadata['numAvailStars'], numAvail)
        self.assertEqual(sum(self.catalog['use_psf']), metadata['numGoodStars'])
        self.assertLessEqual(metadata['numGoodStars'], metadata['numAvailStars'])

        self.assertEqual(
            psf.getAveragePosition(),
            geom.Point2D(
                np.mean([s.x for s in psf._piffResult.stars]),
                np.mean([s.y for s in psf._piffResult.stars])
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

    def testPiffDeterminer_default(self):
        """Test piff with the default config."""
        self.checkPiffDeterminer()

    def testPiffDeterminer_stampSize27(self):
        """Test Piff with a psf stampSize of 27."""
        self.checkPiffDeterminer(stampSize=27)

    def testPiffDeterminer_debugStarData(self):
        """Test Piff with debugStarData=True."""
        self.checkPiffDeterminer(debugStarData=True)

    def testPiffDeterminer_skyCoords(self):
        """Test Piff sky coords."""
        self.checkPiffDeterminer(useCoordinates='sky')

    def testPiffDeterminer_downsample(self):
        """Test Piff determiner with downsampling."""
        self.checkPiffDeterminer(downsample=True)


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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
