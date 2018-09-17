from __future__ import print_function, division
import numpy as np

from .ooephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['DirectObs']


class DirectObs(BaseObs):
    """
    Generate observations of a set of moving objects: exact ephemeris at the times of each observation.

    First generates observations on a rough grid and looks for observations within a specified tolerance
    of the actual observations; for the observations which pass this cut, generates a precise ephemeris
    and checks if the object is within the FOV.


    """
    def __init__(self, cameraFootprint=True, rFov=1.75,
                 ephfile=None, timescale='TAI', obscode='I11',
                 **kwargs):
        super().__init__(cameraFootprint, rFov, **kwargs)
        self.ephems = PyOrbEphemerides(ephfile=ephfile)
        self.timescale = timescale
        self.obscode = obscode

    def generateEphs(self, sso, times, ephMode):
        """Generate ephemerides.
        """
        self.ephems.setOrbits(sso)
        ephTimes = self.ephems._convertTimes(times, self.timescale)
        if ephMode == '2body':
            oorbEphs = self.ephems._generateOorbEphs2body(ephTimes, obscode=self.obscode)
        else:
            oorbEphs = self.ephems._generateOorbEphs(ephTimes, obscode=self.obscode)
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        return ephs

    def run(self, obsData, outfileName, epoch=2000.0):
        """Generate the observations of the objects.
        
        Parameters
        ----------
        obsData : np.recarray
            Observation data. Must contain self.timeCol (time information) and self.raCol, self.decCol.
        outfileName : str
            Output file name.
        epoch : float, opt
            Epoch of the RA/Dec of the observations.
        """
        all_times = obsData[self.timeCol]

        for sso in self.orbits:
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            ephs = self.generateEphs(sso)[0]
            if self.cameraFootprint is None:
                idxObs = self.ssoInCircleFov(ephs, obsData, rFov=self.rFov)
            else:
                idxObs = self.cameraFootprint.inCameraFov(ephs, obsData, epoch, self.timeCol)
            obsdat = obsData[idxObs]
            ephs = ephs[idxObs]
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=outfileName)
