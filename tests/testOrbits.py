import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from lsst.sims.movingObjects import Orbits

class testOrbits(unittest.TestCase):
    def testReadOrbits(self):
        orbits = Orbits()
        orbits.readOrbits('test_orbitsQ.des')
        self.assertEqual(orbits.nSso, 4)
        orbits.readOrbits('test_orbitsA.des')
        self.assertEqual(orbits.nSso, 4)
        with self.assertRaises(ValueError):
            orbits.readOrbits('test_orbitsBad.des')

    def testSetOrbits(self):
        orbits = Orbits()
        orbits.readOrbits('test_orbitsQ.des')
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
