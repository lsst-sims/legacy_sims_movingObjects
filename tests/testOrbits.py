import unittest
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

if __name__ == '__main__':
    unittest.main()
