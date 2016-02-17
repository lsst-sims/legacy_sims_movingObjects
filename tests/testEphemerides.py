import unittest
import numpy as np
from pandas.util.testing import assert_frame_equal
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import PyOrbEphemerides

class testPyOrbEphemerides(unittest.TestCase):
    def setUp(self):
        self.orbits = Orbits()
        self.orbits.readOrbits('test_orbitsQ.des')
        self.orbitsA = Orbits()
        self.orbitsA.readOrbits('test_orbitsA.des')
        self.ephems = PyOrbEphemerides()

    def tearDown(self):
        del self.orbits
        del self.orbitsA
        del self.ephems

    def testSetOrbits(self):
        # Test that we can set orbits.
        self.ephems.setOrbits(self.orbits)
        assert_frame_equal(self.ephems.orbitObj.orbits, self.orbits.orbits)
        # Test that setting with something other than an Orbit object fails.
        with self.assertRaises(ValueError):
            self.ephems.setOrbits(self.orbits.orbits)

    def testConvertOorbArray(self):
        # Check that orbital elements are converted.
        self.ephems.orbitObj = self.orbits
        self.ephems._convertOorbElem()
        self.assertEqual(len(self.ephems.oorbElem), self.orbits.nSso)
        self.assertEqual(self.ephems.oorbElem[0][7], 2)
        self.assertEqual(self.ephems.oorbElem[0][9], 3)
        self.assertEqual(self.ephems.oorbElem[0][1], self.orbits.orbits['q'][0])
        # Test that we can convert KEP orbital elements too.
        self.ephems.orbitObj = self.orbitsA
        self.ephems._convertOorbElem()
        self.assertEqual(len(self.ephems.oorbElem), self.orbitsA.nSso)
        self.assertEqual(self.ephems.oorbElem[0][7], 3)
        self.assertEqual(self.ephems.oorbElem[0][1], self.orbitsA.orbits['a'][0])

    def testConvertFromOorbArray(self):
        self.ephems.orbitObj = self.orbits
        self.ephems._convertOorbElem()
        newOrbits = self.ephems._convertFromOorbElem(self.ephems.oorbElem)
        self.assertEqual(newOrbits, self.orbits)

    def testConvertTimes(self):
        times = np.arange(49353, 49353 + 10, 0.5)
        ephTimes = self.ephems._convertTimes(times, 'UTC')
        # Check that shape of ephTimes is correct.
        self.assertEqual(ephTimes.shape[0], len(times))
        self.assertEqual(ephTimes.shape[1], 2)
        # Check that 'timescale' for ephTimes is correct.
        self.assertEqual(ephTimes[0][1], 1)
        ephTimes = self.ephems._convertTimes(times, 'TAI')
        self.assertEqual(ephTimes[0][1], 4)

    def testOorbEphemeris(self):
        self.ephems.setOrbits(self.orbits)
        times = np.arange(49353, 49353 + 3, 0.25)
        ephTimes = self.ephems._convertTimes(times)
        oorbEphs = self.ephems._generateOorbEphs(ephTimes, obscode=807)
        # Check that it returned the right sort of array.
        self.assertEqual(oorbEphs.shape, (len(self.ephems.oorbElem), len(times), 10))

    def testEphemeris(self):
        # Calculate and convert ephemerides.
        self.ephems.setOrbits(self.orbits)
        times = np.arange(49353, 49353 + 2, 0.3)
        ephTimes = self.ephems._convertTimes(times)
        oorbEphs = self.ephems._generateOorbEphs(ephTimes, obscode=807)
        # Group by object, and check grouping.
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        self.assertEqual(len(ephs), self.orbits.nSso)
        # Group by time, and check grouping.
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=False)
        self.assertEqual(len(ephs), len(times))
        # And test all-wrapped-up method:
        ephsAll = self.ephems.generateEphemerides(times, obscode=807,
                                                  timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAll, ephs)
        # And calculate ephemerides using KEP elements.
        self.ephems.setOrbits(self.orbitsA)
        oorbEphs = self.ephems._generateOorbEphs(ephTimes, obscode=807)
        ephsA = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        self.assertEqual(len(ephsA), self.orbitsA.nSso)
        ephsA = self.ephems._convertOorbEphs(oorbEphs, byObject=False)
        self.assertEqual(len(ephsA), len(times))
        # Check that ephemerides calculated by each method are almost equal.
        for column in ephs.dtype.names:
            np.testing.assert_almost_equal(ephs[column], ephsA[column])
        # And test all-wrapped-up method:
        ephsAllA = self.ephems.generateEphemerides(times, obscode=807,
                                                   timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAllA, ephsA)
        for column in ephsAll.dtype.names:
            np.testing.assert_almost_equal(ephsAllA[column], ephsAll[column])


if __name__ == '__main__':
    unittest.main()