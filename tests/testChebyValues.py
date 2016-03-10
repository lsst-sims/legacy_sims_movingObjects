import unittest
import os
import warnings
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

class TestChebyValues(unittest.TestCase):
    def setUp(self):
        self.testdatadir = 'orbits_testdata'
        self.coeffFile = 'test_coeffs'
        self.residFile = 'test_resids'
        self.failedFile = 'test_failed'
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdatadir, 'test_orbitsMBA.s3m'), skiprows=1)
        tStart = self.orbits.orbits.epoch.iloc[0]
        interval = 30
        self.chebyFits = ChebyFits(self.orbits, tStart, tStart+interval, ngran=64, skyTolerance=2.5, nDecimal=2)
        self.chebyFits.calcGranularity()
        self.chebyFits.calcSegments()
        self.chebyFits.write(self.coeffFile, self.residFile, self.failedFile)

    def tearDown(self):
        del self.orbits
        del self.chebyFits
        os.remove(self.coeffFile)
        os.remove(self.residFile)
        if os.path.isfile(self.failedFile)
            os.remove(self.failedFile)

    def testChebyValSetCoeff(self):
        # Test setting coefficients directly from chebyFits outputs.
        
