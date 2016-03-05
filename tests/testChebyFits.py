import unittest
import os
import warnings
import numpy as np
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

class TestChebyFits(unittest.TestCase):
    def setUp(self):
        self.testdir = 'orbits_testdata'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsMBA.s3m'), skiprows=1)
        self.cheb = ChebyFits(self.orbits, 54800, 54830, ngran=64, skyTolerance=2.5, nDecimal=2)
        self.assertEqual(self.cheb.ngran, 64)

    def tearDown(self):
        del self.orbits
        del self.cheb

    @classmethod
    def tearDownClass(cls):
        os.remove('tmpCoeff')
        os.remove('tmpResids')
        if os.path.isfile('tmpFailed'):
            os.remove('tmpFailed')

    def testPrecomputeMultipliers(self):
        # Precompute multipliers is done as an automatic step in __init__.
        # After setting up self.cheb, these multipliers should all exist.
        for key in self.cheb.nCoeff:
            self.assertTrue(key in self.cheb.multipliers)

    def testSetGranularity(self):
        # Expect MBAs with standard ngran and tolerance to have length ~2.0 days.
        self.cheb.calcGranularity()
        self.assertAlmostEqual(self.cheb.length, 2.0)
        # Test that we can set it to other values which fit into the 30 day window.
        self.cheb.calcGranularity(length=1.5)
        self.assertEqual(self.cheb.length, 1.5)
        # Test that we if we try to set it to a value which does not fit into the 30 day window,
        # that the actual value used is different - and smaller.
        self.cheb.calcGranularity(length=1.9)
        self.assertTrue(self.cheb.length < 1.9)
        # Test that we get a warning about the residuals if we try to set the length to be too long.
        with warnings.catch_warnings(record=True) as w:
            self.cheb.calcGranularity(length=5.0)
            self.assertTrue(len(w), 1)
        # Now check granularity works for other orbit types (which would have other standard lengths).
        # Check for multiple orbit types.
        for orbitFile in (['test_orbitsMBA.s3m', 'test_orbitsOuter.s3m', 'test_orbitsNEO.s3m']):
            self.orbits.readOrbits(os.path.join(self.testdir, orbitFile), skiprows=1)
            tStart = self.orbits.orbits['epoch'].iloc[0]
            cheb = ChebyFits(self.orbits, tStart, tStart + 30, ngran=64, nDecimal=2)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 5.0, 10.0, 100.0, 1000.0, 20000.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcGranularity()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                self.assertEqual((cheb.length * 100) % 1, 0)
                #print 'final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio
        # And check for challenging 'impactors'.
        for orbitFile in (['test_orbitsImpactors.s3m']):
            self.orbits.readOrbits(os.path.join(self.testdir, orbitFile), skiprows=1)
            tStart = self.orbits.orbits['epoch'].iloc[0]
            cheb = ChebyFits(self.orbits, tStart, tStart + 30, ngran=64, nDecimal=10)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 10.0, 100.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcGranularity()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                #print 'final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio

    def testSegments(self):
        # Test that we can create segments.
        self.cheb.calcGranularity()
        times = self.cheb.getAllTimes()
        self.cheb.generateEphemerides(times, verbose=False)
        self.cheb.calcSegments()

    def testWrite(self):
        # Test that we can write the output to files.
        self.cheb.calcGranularity()
        self.cheb.generateEphemerides(self.cheb.getAllTimes())
        self.cheb.calcSegments()
        self.cheb.write('tmpCoeff', 'tmpResids', 'tmpFailed')
        self.assertTrue(os.path.isfile('tmpCoeff'))
        self.assertTrue(os.path.isfile('tmpResids'))

class TestDbRun(unittest.TestCase):
    def setUp(self):
        self.testdir = 'orbits_testdata'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsMBA.s3m'), skiprows=1)
        self.coeffFile = 'tmpCoeff'
        self.residFile = 'tmpResid'
        self.failedFile = 'tmpFailed'

    def tearDown(self):
        del self.orbits
        os.remove(self.coeffFile)
        os.remove(self.residFile)
        if os.path.isfile(self.failedFile):
            os.remove(self.failedFile)

    def testRun(self):
        # Set up chebyshev fitter.
        tStart = self.orbits.orbits.epoch.iloc[0]
        interval = 30
        cheb = ChebyFits(self.orbits, tStart, tStart + interval, ngran=64, skyTolerance=2.5, nDecimal=2)
        # Set granularity. Use an value that will be too long, to trigger recursion below.
        cheb.calcGranularity(length=2.0)
        # Run through segments.
        cheb.calcSegments()
        # Check that write succeeds.
        cheb.write(self.coeffFile, self.residFile, self.failedFile)
        # Test that the segments for each individual object fit together start/end.
        for k in cheb.coeffs:
            cheb.coeffs[k] = np.array(cheb.coeffs[k])
        for objId in np.unique(cheb.coeffs['objId']):
            condition = (cheb.coeffs['objId'] == objId)
            te_prev = tStart
            for ts, te in zip(cheb.coeffs['tStart'][condition], cheb.coeffs['tEnd'][condition]):
                # Test that the start of the current interval = the end of the previous interval.
                self.assertEqual(te_prev, ts)
                te_prev = te
        # Test that the end of the last interval is equal to the end of the total interval
        self.assertEqual(te, tStart + interval)

if __name__ == '__main__':
    unittest.main()
