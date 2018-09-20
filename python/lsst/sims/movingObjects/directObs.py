from __future__ import print_function, division
import numpy as np

from .baseObs import BaseObs

__all__ = ['DirectObs']


class DirectObs(BaseObs):
    """
    Generate observations of a set of moving objects: exact ephemeris at the times of each observation.

    First generates observations on a rough grid and looks for observations within a specified tolerance
    of the actual observations; for the observations which pass this cut, generates a precise ephemeris
    and checks if the object is within the FOV.


    """
    def setup(self, ephfile, tstep, outfileName):
        """Set some parameters for generating ephemerides and writing to disk.

        Parameters
        ----------
        ephfile: str or None, opt
            The name of the planetary ephemerides file to use for ephemeris generation.
            Default is the default for PyOrbEphemerides.
        tstep : float, opt
            The time between points in the ephemeris grid, in days.
            Default 2 hours.
        outfileName : str
            The output file name.
        """
        self.setupEphemerides(ephfile)
        self.tstep = tstep
        self.outfileName = outfileName

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
