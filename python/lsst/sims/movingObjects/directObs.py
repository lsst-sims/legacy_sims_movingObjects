from __future__ import print_function, division
import numpy as np
from itertools import repeat
from scipy import interpolate

from .ephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['DirectObs']

stdTimeCol = 'expMJD'

class DirectObs(BaseObs):
    """
    Class to generate observations of a set of moving objects.
    Uses no interpolation to generate observations - direct ephemeris generation.
    This is slow because it's generating ephemerides for all opsim data times, 
    then culling observations that aren't within FOV.
    """
    def __init__(self, cameraFootprint=None, rFov=1.75,
                 ephfile=None, timescale='TAI', obscode='I11'):
        super(DirectObs, self).__init__(cameraFootprint, rFov)
        self.ephems = PyOrbEphemerides(ephfile=ephfile)
        self.timescale = timescale
        self.timescaleNum = self.ephems.timeScales[timescale]
        self.obscode = obscode

    def setTimes(self, times):
        """
        Set an array for oorb of the ephemeris times desired, given an explicit set of times.
        @ times : numpy array of the actual times of each ephemeris position.
        """
        self.ephTimes = np.array(zip(times, repeat(self.timescaleNum, len(times))),
                                 dtype='double', order='F')

    def generateEphs(self, sso):
        """Generate ephemerides for all times in self.ephTimes.
        """
        self.ephems.setOrbits(sso)
        oorbEphs = self.ephems._generateOorbEphs(self.ephTimes, obscode=self.obscode)
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=False)
        return ephs

    def run(self, obsData, outfileName, epoch=2000.0):
        """
        
        Parameters
        ----------
        obsData : np.recarray
        outfileName : str
        epoch : float, opt
        """
        times = obsData[stdTimeCol]
        self.setTimes(times)

        for sso in self.orbits:
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            ephs = self.generateEphs(sso)
            if self.cameraFootprint is None:
                idxObs = self.ssoInCircleFov(ephs, obsData, rFov=self.rFov)
            else:
                idxObs = self.cameraFootprint.inCameraFov(ephs, obsData, epoch)
            obsdat = obsData[idxObs]
            ephs = ephs[idxObs]
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=outfileName)
