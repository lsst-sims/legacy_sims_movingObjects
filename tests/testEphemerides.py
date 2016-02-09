import unittest
import numpy as np
from lsst.sims.movingObjects import PyOrbEphemerides

class testPyOrbEphemerides(unittest.TestCase):
    def setUp(self):
        self.ephems = PyOrbEphemerides()
        self.ephems.readOrbits('test_orbitsQ.des')
        self.assertEqual(self.ephems.nSso, 4)
        self.ephemsA = PyOrbEphemerides()
        self.ephemsA.readOrbits('test_orbitsA.des')
        self.assertEqual(self.ephemsA.nSso, 4)

    def tearDown(self):
        del self.ephems
        del self.ephemsA

    def testPackOorbArray(self):
        # Check that orbital elements are converted.
        oorbElem = self.ephems.convertOorbElem()
        self.assertEqual(len(oorbElem), self.ephems.nSso)
        oorbElem = self.ephemsA.convertOorbElem()
        self.assertEqual(len(oorbElem), self.ephemsA.nSso)
        # Check that orbital elements are converted, for a subset dataframe.
        sso = self.ephems.orbits.head(1)
        oorbElem = self.ephems.convertOorbElem(sso=sso)
        self.assertEqual(len(oorbElem), len(sso))
        # Check that orbital elements are converted, for a series subset.
        for i, sso in self.ephems.orbits.iterrows():
            oorbElem = self.ephems.convertOorbElem(sso=sso)
            self.assertEqual(len(oorbElem), 1)

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
        times = np.arange(49353, 49353 + 3, 0.25)
        ephTimes = self.ephems._convertTimes(times)
        oorbElem = self.ephems.convertOorbElem()
        oorbEphs = self.ephems._generateOorbEphs(oorbElem, ephTimes, obscode=807)
        # Check that it returned the right sort of array.
        self.assertEqual(oorbEphs.shape, (len(oorbElem), len(times), 10))

    def testEphemeris(self):
        # Calculate and convert ephemerides.
        times = np.arange(49353, 49353 + 2, 0.3)
        ephTimes = self.ephems._convertTimes(times)
        oorbElem = self.ephems.convertOorbElem()
        oorbEphs = self.ephems._generateOorbEphs(oorbElem, ephTimes, obscode=807)
        # Group by object, and check grouping.
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        self.assertEqual(len(ephs), self.ephems.nSso)
        # Group by time, and check grouping.
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=False)
        self.assertEqual(len(ephs), len(times))
        # And test all-wrapped-up method:
        ephsAll = self.ephems.generateEphemerides(oorbElem, times, obscode=807,
                                             timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAll, ephs)
        # And calculate ephemerides using KEP elements.
        oorbElem = self.ephemsA.convertOorbElem()
        oorbEphs = self.ephemsA._generateOorbEphs(oorbElem, ephTimes, obscode=807)
        ephsA = self.ephemsA._convertOorbEphs(oorbEphs, byObject=True)
        self.assertEqual(len(ephsA), self.ephemsA.nSso)
        ephsA = self.ephems._convertOorbEphs(oorbEphs, byObject=False)
        self.assertEqual(len(ephsA), len(times))
        # Check that ephemerides calculated by each method are almost equal.
        for column in ephs.dtype.names:
            np.testing.assert_almost_equal(ephs[column], ephsA[column])
        # And test all-wrapped-up method:
        ephsAll = self.ephemsA.generateEphemerides(oorbElem, times, obscode=807,
                                             timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAll, ephsA)


if __name__ == '__main__':
    unittest.main()
