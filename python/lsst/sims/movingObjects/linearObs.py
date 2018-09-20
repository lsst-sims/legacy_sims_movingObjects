from __future__ import print_function, division
import numpy as np
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
    cameraFootprint : bool, opt
        If True, then use the camera footprint exactly. If False, then only use rFov.
        Default False (use rFov only).
    rFov : float, opt
        Radius of the fov, to use for a circular FOV if cameraFootprint is None, in degrees.
        Default 1.75 degrees.
    ephMode: str, opt
        Mode for ephemeris generation - nbody or 2body. Default is 2body.
    obsCode: str, opt
        Observatory code for ephemeris generation. Default is "I11" - Cerro Pachon.
    obsTimeCol: str, opt
        Name of the time column in the obsData. Default 'observationStartMJD'.
    obsTimeScale: str, opt
        Type of timescale for MJD (TAI or UTC currently). Default TAI.
    seeingCol: str, opt
        Name of the seeing column in the obsData. Default 'seeingFwhmGeom'.
        This should be the geometric/physical seeing as it is used for the trailing loss calculation.
    visitExpTimeCol: str, opt
        Name of the visit exposure time column in the obsData. Default 'visitExposureTime'.
    obsRa: str, opt
        Name of the RA column in the obsData. Default 'fieldRA'.
    obsDec: str, opt
        Name of the Dec column in the obsData. Default 'fieldDec'.
    obsRotSkyPos: str, opt
        Name of the Rotator column in the obsData. Default 'rotSkyPos'.
    obsDegrees: bool, opt
        Whether the observational data is in degrees or radians. Default True (degrees).
    """
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

    def run(self, orbits, obsData):
        """Find and write the observations of each object to disk.

        For each object, identify the observations where the object is
        within rFOV of the pointing boresight (potentially, also in the camera footprint),
        and write the ephemeris values and observation metadata to disk.
        Uses linear interpolation between ephemeris gridpoints.

        Parameters
        ----------
        orbits: lsst.sims.movingObjects.Orbits
            The orbits to generate ephemerides for.
        obsData : np.recarray
            The simulated pointing history data.
        outfileName : str
            The output file name.
        tstep : float, opt
            The time between points in the ephemeris grid, in days.
            Default 2 hours.
        """
        # Set the times for the ephemeris grid.
        timeStep = float(self.tstep)
        timeStart = obsData[self.obsTimeCol].min() - timeStep
        timeEnd = obsData[self.obsTimeCol].max() + timeStep
        times = np.arange(timeStart, timeEnd + timeStep / 2.0, timeStep)
        print('Generating ephemerides on a grid of %f day timesteps, then will extrapolate to opsim times.'
              % (tstep))
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for sso in orbits:
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            ephs = self.generateEphemerides(sso, times)
            interpfuncs = self.makeInterps(ephs)
            ephs = self.interpEphs(interpfuncs, times=obsData[self.obsTimeCol], columns=['ra', 'dec'])
            if self.cameraFootprint:
                idxObs = self.ssoInCameraFov(ephs, obsData)
            else:
                idxObs = self.ssoInCircleFov(ephs, obsData)
            obsdat = obsData[idxObs]
            ephs = self.interpEphs(interpfuncs, times=obsdat[self.obsTimeCol])
            # Write these observations to disk.
            self.writeObs(objid, ephs, obsdat, sedname=sedname, outfileName=self.outfileName)
