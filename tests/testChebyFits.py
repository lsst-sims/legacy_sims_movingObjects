import unittest
import os
import numpy as np
import warnings
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

class TestChebyFits(unittest.TestCase):
    def setUp(self):
        self.testdir = 'testOrbits'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsMBA.s3m'), skiprows=1)
        self.cheb = ChebyFits(self.orbits, 54800, 54830, ngran=64, skyTolerance=2.5)
        self.assertEqual(self.cheb.ngran, 64)

    def tearDown(self):
        del self.orbits
        del self.cheb

    def testPrecomputeMultipliers(self):
        self.cheb._precomputeMultipliers()
        for key in self.cheb.nCoeff:
            self.assertTrue(key in self.cheb.multipliers)

    def testSetGranularity(self):
        orbits = Orbits()
        # Expect MBAs with standard ngran and tolerance to have length ~2.0 days.
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsMBA.s3m'), skiprows=1)
        cheb = ChebyFits(orbits, 54800, 54830, ngran=64, skyTolerance=2.5, databaseCoeffs=True, nDecimal=2)
        cheb.calcGranularity()
        self.assertAlmostEqual(cheb.length, 2.0)
        cheb.calcGranularity(length = 1.5)
        self.assertEqual(cheb.length, 1.5)
        self.assertRaises(ValueError, cheb.calcGranularity(length = 1.9))
        with warnings.catch_warnings(record=True) as w:
            cheb.calcGranularity(length = 5.0)
            self.assertTrue(len(w), 1)
        # Check for multiple orbit types.
        for orbitFile in (['test_orbitsMBA.s3m', 'test_orbitsOuter.s3m', 'test_orbitsNEO.s3m']):
            orbits.readOrbits(os.path.join(self.testdir, orbitFile), skiprows=1)
            tStart = orbits.orbits['epoch'].iloc[0]
            cheb = ChebyFits(orbits, tStart, tStart+30, ngran=64)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 5.0, 10.0, 100.0, 500.0, 5000.0, 20000.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcGranularity()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                #print 'final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio
        for orbitFile in (['test_orbitsImpactors.s3m']):
            orbits.readOrbits(os.path.join(self.testdir, orbitFile), skiprows=1)
            tStart = orbits.orbits['epoch'].iloc[0]
            cheb = ChebyFits(orbits, tStart, tStart+30, ngran=64)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 10.0, 100.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcGranularity()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                #print 'final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio

    def testOneSegment(self):
        self.cheb.calcGranularity()
        times = self.cheb.getAllTimes()
        self.cheb.generateEphemerides(times, verbose=True)
        tstart = self.cheb.tStart
        tend = self.cheb.tStart + self.cheb.length
        #self.cheb.doOneSegment(tstart, tend)

"""
class TestBigFile(unittest.TestCase):
    def setUp(self):
        self.testdir = 'testOrbits'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, 'S0_1000.s3m'), skiprows=1)
        self.cheb = ChebyFits(self.orbits, 54800, 54830, ngran=64, skyTolerance=2.5)

    def tearDown(self):
        del self.orbits
        del self.cheb

    def testRun(self):
        self.cheb.calcGranularity()
        pos_resid, ratio = self.cheb._testResiduals(self.cheb.length)
        self.assertTrue(pos_resid < self.cheb.skyTolerance)
"""

if __name__ == '__main__':
    unittest.main()
