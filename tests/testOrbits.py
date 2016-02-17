import unittest
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
from lsst.sims.movingObjects import Orbits

class TestOrbits(unittest.TestCase):
    def setUp(self):
        self.testdir = 'testOrbits'

    def testBuiltIns(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(len(orbits), 4)
        orbits2 = Orbits()
        orbits2.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(orbits, orbits2)
        orbits3 = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        self.assertNotEqual(orbits, orbits3)
        orbits3 = orbits[0]
        assert_frame_equal(orbits3.orbits, orbits.orbits.head(1))
        for orb, (i, orbi) in zip(orbits, orbits.orbits.iterrows()):
            self.assertEqual(orb.orbits['objId'].values[0], orbi['objId'])

    def testReadOrbits(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(orbits.nSso, 4)
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        self.assertEqual(orbits.nSso, 4)
        with self.assertRaises(ValueError):
            orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsBad.des'))

    def testSetOrbits(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        # Test that we can set the orbits using a dataframe.
        suborbits = orbits.orbits.head(1)
        newOrbits = Orbits()
        newOrbits.setOrbits(suborbits)
        self.assertEqual(newOrbits.nSso, 1)
        self.assertEqual(newOrbits.format, 'COM')
        assert_frame_equal(newOrbits.orbits, suborbits)
        # Test that we can set the orbits using a Series.
        for i, sso in suborbits.iterrows():
            newOrbits = Orbits()
            newOrbits.setOrbits(sso)
            self.assertEqual(newOrbits.nSso, 1)
            self.assertEqual(newOrbits.format, 'COM')
            assert_frame_equal(newOrbits.orbits, suborbits)
        # Test that we can set the orbits using a numpy array with many objects.
        numpyorbits = orbits.orbits.to_records(index=False)
        newOrbits = Orbits()
        newOrbits.setOrbits(numpyorbits)
        self.assertEqual(newOrbits.nSso, orbits.nSso)
        self.assertEqual(newOrbits.format, 'COM')
        assert_frame_equal(newOrbits.orbits, orbits.orbits)
        # And test that this works for a single row of the numpy array.
        onenumpyorbits = numpyorbits[0]
        newOrbits = Orbits()
        newOrbits.setOrbits(onenumpyorbits)
        self.assertEqual(newOrbits.nSso, 1)
        self.assertEqual(newOrbits.format, 'COM')
        assert_frame_equal(newOrbits.orbits, suborbits)
        # And test that it fails appropriately when columns are not correct.
        neworbits = pd.DataFrame(orbits.orbits)
        newcols = neworbits.columns.values.tolist()
        newcols[0] = 'ssmId'
        newcols[3] = 'ecc'
        neworbits.columns = newcols
        newOrbits = Orbits()
        with self.assertRaises(ValueError):
            newOrbits.setOrbits(neworbits)

if __name__ == '__main__':
    unittest.main()
