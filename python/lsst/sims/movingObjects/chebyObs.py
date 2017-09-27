from __future__ import print_function, division

from .chebyValues import ChebyValues
from .baseObs import BaseObs

__all__ = ['ChebyObs']


class ChebyObs(BaseObs):
    """
    Class to generate observations of a set of moving objects.
    Uses linear interpolation between gridpoint of ephemerides.
    """
    def __init__(self, cameraFootprint=None, rFov=1.75, **kwargs):
        super(ChebyObs, self).__init__(cameraFootprint, rFov, **kwargs)
        self.chebval = ChebyValues

    # Set cheby fits coefficients for object.
    def setChebyFits(self, chebfits):
        self.chebval.setCoefficients(chebfits)

    def generateEphs(self, sso, obsData):
        """Generate ephemerides for all times object possibly visible in obsData.
        """
        times = obsData[self.timeCol]
        ephs = self.chebval.getEphemerides(times, extrapolate=True)
        return ephs

    def run(self, obsData, outfileName, epoch):
        """
        
        Parameters
        ----------
        obsData
        outFileName
        """
        for sso in self.orbits:
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            ephs = self.generateEphs(sso, obsData)
            if self.cameraFootprint is None:
                idxObs = self.ssoInCircleFov(ephs, obsData, rFov=self.rFov)
            else:
                idxObs = self.cameraFootprint.inCameraFov(ephs, obsData, epoch, self.timeCol)
            obsdat = obsData[idxObs]
            ephs = self.generateEphs(sso, obsdat)
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=outfileName)
