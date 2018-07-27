from __future__ import print_function, division
import numpy as np
from itertools import repeat

from .ooephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['DirectObs']


class DirectObs(BaseObs):
    """Generate observations of a set of Orbits, using direct ephemeris generation.

    Uses no interpolation to generate observations - direct ephemeris generation.
    This is slow because it's generating ephemerides for all opsim data times,
    then culling observations that aren't within FOV.

    Parameters
    ----------
    cameraFootprint : CameraFootprint, opt
        A cameraFootprint class (which provides a way to determine if observations fall into a given
        camera footprint), such as lsstCameraFootprint.
        If none provided, a simple circular FOV is available from the BaseObs class.
    rFov : float, opt
        Radius of the fov, to use for a circular FOV if cameraFootprint is None, in degrees.
        Default 1.75 degrees.
    timeCol : str, opt
        Name of the column (in the simulated pointing history) for Time (MJD).
        Default observationStartMJD.
    seeingCol : str, opt
        Name of the column (in the simulated pointing history) for the physical size of the seeing.
        Default seeingFwhmGeom.
    visitExpTimeCol : str, opt
        Name of the column (in the simulated pointing history) for the visit exposure time.
        Default visitExposureTime.
    """
    def __init__(self, cameraFootprint=None, rFov=1.75,
                 ephfile=None, timescale='TAI', obscode='I11',
                 ephMode='nbody', **kwargs):
        super(DirectObs, self).__init__(cameraFootprint, rFov, **kwargs)
        self.ephems = PyOrbEphemerides(ephfile=ephfile)
        self.timescale = timescale
        self.timescaleNum = self.ephems.timeScales[timescale]
        self.obscode = obscode
        if ephMode.lower() not in ('2body', 'nbody'):
            raise ValueError('Ephemeris generation must be 2body or nbody.')
        self.ephMode = ephMode

    def setTimes(self, times):
        """Set an array for oorb of the ephemeris times desired, given an explicit set of times.

        Parameters
        ----------
        times : numpy.ndarray
            Array of the actual times of each ephemeris position.
        """
        self.ephTimes = np.array(list(zip(times, repeat(self.timescaleNum, len(times)))),
                                 dtype='double', order='F')

    def generateEphs(self, sso):
        """Generate ephemerides for all times in self.ephTimes.

        Parameters
        ----------
        sso : lsst.sims.movingObjects.Orbits
            The (single) object for which to generate ephemerides at times self.ephTimes.

        Returns
        -------
        numpy.ndarray
            The ephemeride values at all times self.ephTimes.
        """
        self.ephems.setOrbits(sso)
        if self.ephMode == '2body':
            oorbEphs = self.ephems._generateOorbEphs2body(self.ephTimes, obscode=self.obscode)
        else:
            oorbEphs = self.ephems._generateOorbEphs(self.ephTimes, obscode=self.obscode)
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        return ephs

    def run(self, obsData, outfileName, epoch=2000.0):
        """Find and write the observations of each object to disk.

        For each object, identify the observations where the object is
        within rFOV of the pointing boresight (potentially, also in the camera footprint),
        and write the ephemeris values and observation metadata to disk.
        Uses direct ephemeris generation for each pointing.

        Parameters
        ----------
        obsData : np.recarray
            The simulated pointing history data.
        outfileName : str
            The output file name.
        tstep : float, opt
            The time between points in the ephemeris grid, in days.
            Default 2 hours.
        epoch : float, opt
            The epoch of the RA/Dec reference frame.
            Default 2000.0
        """
        times = obsData[self.timeCol]
        self.setTimes(times)

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
