import unittest
import os
import warnings
import numpy as np

from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import PyOrbEphemerides
from lsst.sims.movingObjects import ChebyFits
from lsst.sims.movingObjects import ChebyValues

class TestChebyValues(unittest.TestCase):
    def setUp(self):
        self.testdatadir = 'orbits_testdata'
        self.coeffFile = 'test_coeffs'
        self.residFile = 'test_resids'
        self.failedFile = 'test_failed'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdatadir, 'test_orbitsMBA.s3m'), skiprows=1)
        self.pyephems = PyOrbEphemerides(os.path.join(os.getenv('OORB_DATA'), 'DE405.dat'))
        self.pyephems.setOrbits(self.orbits)
        self.tStart = self.orbits.orbits.epoch.iloc[0]
        self.interval = 15
        self.chebyFits = ChebyFits(self.orbits, self.tStart, self.tStart+self.interval, ngran=64,
                                   skyTolerance=2.5, nDecimal=2, nCoeff_position=14, obscode=807)
        self.chebyFits.calcGranularity(length=1.0)
        self.chebyFits.calcSegments()
        self.chebyFits.write(self.coeffFile, self.residFile, self.failedFile, append=False)
        self.coeffKeys = ['objId', 'tStart', 'tEnd', 'ra', 'dec', 'delta', 'vmag', 'elongation']

    def tearDown(self):
        del self.orbits
        del self.chebyFits
        os.remove(self.coeffFile)
        os.remove(self.residFile)
        if os.path.isfile(self.failedFile):
            os.remove(self.failedFile)

    def testSetCoeff(self):
        # Test setting coefficients directly from chebyFits outputs.
        chebyValues = ChebyValues()
        chebyValues.setCoefficients(self.chebyFits)
        for k in self.coeffKeys:
            self.assertTrue(k in chebyValues.coeffs)
            self.assertTrue(isinstance(chebyValues.coeffs[k], np.ndarray))
        self.assertEqual(len(chebyValues.coeffs['tStart']), self.interval * len(self.orbits))
        self.assertEqual(len(chebyValues.coeffs['ra'][0]), 14)
        self.assertTrue('meanRA' in chebyValues.coeffs)
        self.assertTrue('meanDec' in chebyValues.coeffs)

    def testReadCoeffs(self):
        # Test reading the coefficients from disk.
        chebyValues = ChebyValues()
        chebyValues.readCoefficients(self.coeffFile)
        chebyValues2 = ChebyValues()
        chebyValues2.setCoefficients(self.chebyFits)
        for k in chebyValues.coeffs:
            if k == 'objId':
                # Can't test strings with np.test.assert_almost_equal.
                np.testing.assert_equal(chebyValues.coeffs[k], chebyValues2.coeffs[k])
            else:
                # All of these will only be accurate to 2 less decimal places than they are
                # print out with in chebyFits. Since vmag, delta and elongation only use 7
                # decimal places, this means we can test to 5 decimal places for those.
                np.testing.assert_almost_equal(chebyValues.coeffs[k], chebyValues2.coeffs[k],
                                               decimal=5)

    def testGetEphemerides(self):
        chebyValues = ChebyValues()
        chebyValues.setCoefficients(self.chebyFits)
        time = self.tStart + self.interval / 2.0
        # Test for all objects.
        ephemerides = chebyValues.getEphemerides(time)
        pyephemerides = self.pyephems.generateEphemerides(time, obscode=807,
                                                          timeScale='TAI', byObject=False)
        # RA and Dec should agree to 2.5mas (skyTolerance above)
        pos_residuals = np.sqrt((ephemerides['ra'] - pyephemerides['ra'][0]) ** 2 +
                                (ephemerides['dec'] - pyephemerides['dec'][0]) ** 2)
        pos_residuals *= 3600.0 * 1000.0
        self.assertTrue(np.max(pos_residuals) <= 2.5)
        # Let's just look at the max residuals in all quantities.
        print ephemerides.keys()
        for k in ('ra', 'dec', 'dradt', 'ddecdt', 'delta'):
            resids = np.abs(ephemerides[k] - pyephemerides[k][0])
            print k, np.max(resids)
        resids = np.abs(ephemerides['elongation'] - pyephemerides['solarelon'][0])
        print 'elongation', np.max(resids)
        resids = np.abs(ephemerides['vmag'] - pyephemerides['magV'][0])
        print 'vmag', np.max(resids)
        # Test this for a subset of the objects.
        objIds = self.orbits.orbits.objId.head(3).as_matrix()
        ephemerides = chebyValues.getEphemerides(time, objIds)
        self.assertEqual(len(ephemerides['ra']), 3)

if __name__ == '__main__':
    unittest.main()
