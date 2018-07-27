from __future__ import print_function, division
import numpy as np
from itertools import repeat
from scipy import interpolate

from .ooephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['LinearObs']


class LinearObs(BaseObs):
    """Generate observations for a set of Orbits using linear interpolation.

    Uses linear interpolations between grid of true ephemerides.
    Ephemerides can be generated using 2-body or n-body integration.

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
                 ephfile=None, timescale='TAI', obscode='I11', ephMode='2body', **kwargs):
        super(LinearObs, self).__init__(cameraFootprint, rFov, **kwargs)
        self.ephems = PyOrbEphemerides(ephfile=ephfile)
        self.timescale = timescale
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
        self.ephTimes = self.ephems._convertTimes(times, timeScale=self.timescale)

    def generateEphs(self, sso):
        """Generate ephemerides for all times in self.ephTimes.

        This sets up the grid of ephemerides to linearly interpolate between.
        """
        self.ephems.setOrbits(sso)
        if self.ephMode == '2body':
            oorbEphs = self.ephems._generateOorbEphs2body(self.ephTimes, obscode=self.obscode)
        else:
            oorbEphs = self.ephems._generateOorbEphs(self.ephTimes, obscode=self.obscode)
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        return ephs

    # Linear interpolation
    def makeInterps(self, ephs, i=0):
        """Generate the interpolation functions for the linear interpolation.

        Parameters
        ----------
        ephs : np.recarray
            Grid of actual ephemerides.
        i : int, opt
            The index of the object (in the ephemerides) to generate linear interpolations for.
            Default i = 0.

        Returns
        -------
        dictionary
            Dictionary of the interpolation functions.
        """
        interpfuncs = {}
        for n in ephs.dtype.names:
            if n == 'time':
                continue
            interpfuncs[n] = interpolate.interp1d(ephs['time'][i], ephs[n][i], kind='linear',
                                                  assume_sorted=True, copy=False)
        return interpfuncs

    def interpEphs(self, interpfuncs, times, columns=None):
        """Calculate the linear interpolation approximations of the ephemeride columns.

        Parameters
        ----------
        interpfuncs : dict
            Dictionary of the linear interpolation functions.
        times : np.ndarray
            Times at which to generate ephemerides.
        columns : list of str, opt
            List of the values to generate ephemerides for.
            Default None = generate all values.

        Returns
        -------
        np.recarray
            Array of interpolated ephemerides.
        """
        if columns is None:
            columns = interpfuncs.keys()
        dtype = []
        for col in columns:
            dtype.append((col, '<f8'))
        dtype.append(('time', '<f8'))
        ephs = np.recarray([len(times)], dtype=dtype)
        for col in columns:
            ephs[col] = interpfuncs[col](times)
        ephs['time'] = times
        return ephs

    def run(self, obsData, outfileName, tstep=2./24., epoch=2000.0):
        """Find and write the observations of each object to disk.

        For each object, identify the observations where the object is
        within rFOV of the pointing boresight (potentially, also in the camera footprint),
        and write the ephemeris values and observation metadata to disk.
        Uses linear interpolation between ephemeris gridpoints.

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
        # Set the times for the ephemeris grid.
        timeStep = float(tstep)
        timeStart = obsData[self.timeCol].min() - timeStep
        timeEnd = obsData[self.timeCol].max() + timeStep
        times = np.arange(timeStart, timeEnd + timeStep / 2.0, timeStep)
        self.setTimes(times)
        print('Generating ephemerides on a grid of %f day timesteps, then will extrapolate to opsim times.'
              % (tstep))
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for sso in self.orbits:
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            ephs = self.generateEphs(sso)
            interpfuncs = self.makeInterps(ephs)
            ephs = self.interpEphs(interpfuncs, times=obsData[self.timeCol], columns=['ra', 'dec'])
            if self.cameraFootprint is None:
                idxObs = self.ssoInCircleFov(ephs, obsData, rFov=self.rFov)
            else:
                idxObs = self.cameraFootprint.inCameraFov(ephs, obsData, epoch, self.timeCol)
            obsdat = obsData[idxObs]
            ephs = self.interpEphs(interpfuncs, times=obsdat[self.timeCol])
            # Write these observations to disk.
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=outfileName)
