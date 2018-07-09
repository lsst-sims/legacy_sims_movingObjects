from __future__ import print_function, division
import numpy as np
from itertools import repeat
from scipy import interpolate

from .ooephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['LinearObs']


class LinearObs(BaseObs):
    """
    Class to generate observations of a set of moving objects.
    Uses linear interpolation between gridpoint of ephemerides.
    Ephemerides can be generated using 2-body or n-body integration.
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

    # Setup for linear interpolation.
    def setTimesRange(self, timeStep=1., timeStart=59580., timeEnd=63230.):
        """
        Set an array for oorb of the ephemeris times desired, given the range of values.
        @ timeStep : timestep for ephemeris generation (days)
        @ timeStart : starting time of ephemerides (MJD)
        @ timeEnd : ending time of ephemerides (MJD)
        """
        # Extend times beyond first/last observation, so that interpolation doesn't fail
        timeStep = float(timeStep)
        timeStart = timeStart - timeStep
        timeEnd = timeEnd + timeStep
        times = np.arange(timeStart, timeEnd + timeStep/2.0, timeStep)
        # For pyoorb, we need to tag times with timescales;
        # 1= MJD_UTC, 2=UT1, 3=TT, 4=TAI
        self.ephTimes = self.ephems._convertTimes(times, timeScale=self.timescale)

    def setTimes(self, times):
        """
        Set an array for oorb of the ephemeris times desired, given an explicit set of times.
        @ times : numpy array of the actual times of each ephemeris position.
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
        """Calculate the linear interpolation approximations of the ephemerides.

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
        """

        Parameters
        ----------
        obsData : np.recarray
        outfileName : str
        tstep : float
        epoch : float, opt
        """
        self.setTimesRange(timeStep=tstep, timeStart=obsData[self.timeCol].min(),
                           timeEnd=obsData[self.timeCol].max())
        print('Generating ephemerides on a grid of %f day timesteps, then will extrapolate to opsim times.'
              % (tstep))

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
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=outfileName)
