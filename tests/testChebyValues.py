import unittest
import os
import warnings
import numpy as np
import pandas as pd
from astropy.time import Time

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
        self.nCoeffs = 14  # for NEOs, use 16. for everything else, use 14.
        self.chebyFits = ChebyFits(self.orbits, self.tStart, self.tStart+self.interval, ngran=64,
                                   skyTolerance=2.5, nDecimal=2, nCoeff_position=self.nCoeffs, obscode=807)
        self.setLength = 1.0
        self.chebyFits.calcSegmentLength(length=self.setLength)
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
        # This will only be true for carefully selected length/orbit type, where subdivision did not occur.
        self.assertEqual(len(np.unique(chebyValues.coeffs['objId'])), len(self.orbits))
        self.assertEqual(len(chebyValues.coeffs['tStart']), (self.interval / self.setLength) * len(self.orbits))
        self.assertEqual(len(chebyValues.coeffs['ra'][0]), self.nCoeffs)
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
        # Test that getEphemerides works and is accurate.
        chebyValues = ChebyValues()
        #chebyValues.setCoefficients(self.chebyFits)
        chebyValues.readCoefficients(self.coeffFile)
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
        for k in ('ra', 'dec', 'dradt', 'ddecdt', 'delta'):
            resids = np.abs(ephemerides[k] - pyephemerides[k][0])
            print 'max diff ', k, np.max(resids)
        resids = np.abs(ephemerides['elongation'] - pyephemerides['solarelon'][0])
        print 'max diff elongation', np.max(resids)
        resids = np.abs(ephemerides['vmag'] - pyephemerides['magV'][0])
        print 'max diff vmag', np.max(resids)
        # Test this for a subset of the objects.
        objIds = self.orbits.orbits.objId.head(3).as_matrix()
        ephemerides = chebyValues.getEphemerides(time, objIds)
        self.assertEqual(len(ephemerides['ra']), 3)


class TestJPLValues(unittest.TestCase):
    # Test the interpolation-generated RA/Dec values against JPL generated RA/Dec values.
    def setUp(self):
        # Read orbits.
        self.orbits = Orbits()
        self.orbits.readOrbits('jpl_testdata/S0_n747.des', skiprows=1)
        # Read JPL ephems.
        self.jpl = pd.read_table('jpl_testdata/807_n747.txt', delim_whitespace=True)
        # Add times in TAI and UTC, because.
        t = Time(self.jpl['epoch_mjd'], format='mjd', scale='utc')
        self.jpl['mjdTAI'] = t.tai.mjd
        self.jpl['mjdUTC'] = t.utc.mjd
        # Generate interpolation coefficients for the time period in the JPL catalog.
        self.coeffFile = 'test_coeffs'
        self.residFile = 'test_resids'
        self.failedFile = 'test_failed'
        tStart = self.jpl['mjdTAI'].min() - 0.2
        tEnd = np.max([self.jpl['mjdTAI'].max() + 0.2, tStart + 1])
        self.chebyFits = ChebyFits(self.orbits, tStart, tEnd,
                                   ngran=64, skyTolerance=2.5, nDecimal=14,
                                   nCoeff_position=16, obscode=807)
        self.chebyFits.calcSegmentLength()
        self.chebyFits.calcSegments()
        self.chebyFits.write(self.coeffFile, self.residFile, self.failedFile, append=False)
        self.coeffKeys = ['objId', 'tStart', 'tEnd', 'ra', 'dec', 'delta', 'vmag', 'elongation']
        self.chebyValues = ChebyValues()
        self.chebyValues.readCoefficients(self.coeffFile)

    def tearDown(self):
        del self.orbits
        del self.jpl
        #os.remove(self.coeffFile)
        #os.remove(self.residFile)
        #if os.path.isfile(self.failedFile):
        #    os.remove(self.failedFile)

    def testRADec(self):
        # We won't compare Vmag, because this also needs information on trailing losses.
        times = self.jpl['mjdTAI'].unique()
        deltaObjId = np.zeros(len(times), bool)
        deltaRA = np.zeros(len(times), float)
        deltaDec = np.zeros(len(times), float)
        for i, t in enumerate(times):
            # Find the JPL objIds visible at this time.
            j = self.jpl.query('mjdTAI == @t')
            objIds = j.objId.values
            ephs = self.chebyValues.getEphemerides(t, objIds)
            jargsort = np.argsort(objIds)
            eargsort = np.argsort(ephs['objId'])
            if np.all(ephs['objId'][eargsort] == objIds[jargsort]):
                deltaObjId[i] = False # No mismatches in object id
            else:
                deltaObjId[i] = True
            deltaRA[i] = np.abs(ephs['ra'][eargsort] - j['ra_deg'].as_matrix()[jargsort]).max()
            deltaDec[i] = np.abs(ephs['dec'][eargsort] - j['dec_deg'].as_matrix()[jargsort]).max()
            print deltaObjId[i], deltaRA[i]*3600*1000, deltaDec[i]*3600*1000
        # Convert to mas
        deltaRA *= 3600. * 1000.
        deltaDec *= 3600. * 1000.
        # Should be (given OOrb direct prediction):
        # Much of the time we're closer than 1mas, but there are a few which hit higher values. (this is not yet true here!)
        self.assertTrue(np.max(deltaRA) < 18)
        self.assertTrue(np.max(deltaDec) < 6)
        self.assertTrue(np.std(deltaRA) < 2)
        self.assertTrue(np.std(deltaDec) < 1)


if __name__ == '__main__':
    unittest.main()
