import os
import numpy as np
from lsst.sims.utils import haversine
import .chebyshevUtils as cheb
from .orbits import Orbits
from .ephemerides import PyEphemerides

__all__ = ['ChebyFits']

def v_three_sixy_to_neg(element, min, max):
    if (min < 100) & (max > 270):
        if element > 270:
            return element - 360.
        else:
            return element
    else:
        return element
v_360_to_neg = np.vectorize(three_sixy_to_neg)


class ChebyFits(object):
    """Generates chebyshev coefficients for a provided set of orbits.

    Calculates true ephemerides using PyEphemerides, then fits these positions with a constrained
    Chebyshev Polynomial, using the routines in chebyshevUtils.py.
    Many chebyshev polynomials are used to fit one moving object over a given timeperiod;
    typically, the length of each timespan of each fit is about two days.
    The timestep between computed ephemerides varies with population;
    typically, the positions are calculated approximately every 30 minutes.
    The exact timestep and length of each timestep is adjusted so that the residuals in RA/Dec position
    are less than skyTolerance - default = 2.5mas.

    Default values are based on Yusra AlSayaad's work.

    Parameters
    ----------
    orbits : Orbits
        The orbits for which to fit chebyshev polynomial coefficients.
    tStart : float
        The starting point in time to fit coefficients. MJD.
    tEnd : float
        The end point in time to fit coefficients. MJD.
    timeScale : {'TAI', 'UTC', 'TT'}
        The timescale of the MJD times, tStart/tEnd, and the time that should be used with the chebyshev coefficients.
    obsCode : int, optional
        The observatory code of the location for which to generate ephemerides. Default 807 (CTIO).
    skyTolerance : float, optional
        The desired tolerance in mas between ephemerides calculated by OpenOrb and fitted values.
        Default 2.5 mas.
    coeff_position : int, optional
        The number of Chebyshev coefficients to fit for the RA/Dec positions. Default 14.
    coeff_vmag : int, optional
        The number of Chebyshev coefficients to fit for the V magnitude values. Default 9.
    coeff_delta : int, optional
        The number of Chebyshev coefficients to fit for the distance between Earth/Object. Default 5.
    coeff_elongation : int, optional
        The number of Chebyshev coefficients to fit for the solar elongation. Default 5.
    ngran : int, optional
        The number of ephemeris points within each Chebyshev polynomial segment. Default 64.
    ephFile : str, optional
        The path to the JPL ephemeris file to use. Default is '$OORB_DATA/de405.dat'.
    """
    def __init__(self, orbits, tStart, tEnd, timeScale='TAI',
                 obsCode=807, skyTolerance=2.5, coeff_position=14, coeff_vmag=9, coeff_delta=5,
                 coeff_elongation=5, ngran=64, ephFile=None):
        self._setOrbits(orbits)
        self.tStart = tStart
        self.tEnd = tEnd
        if timeScale.upper() == 'TAI':
            self.timeScale = 'TAI'
        elif timeScale.upper() == 'UTC':
            self.timeScale = 'UTC'
        elif timeScale.upper() == 'TT':
            self.timeScale = 'TT'
        else:
            raise ValueError('Do not understand timeScale; use TAI, UTC or TT.')
        self.tDays = (self.tEnd - self.tStart)
        self.obsCode = obsCode
        self.skyTolerance = skyTolerance
        self.coeff = {}
        self.coeff['position'] = coeff_position
        self.coeff['vmag'] = coeff_vmag
        self.coeff['delta'] = coeff_delta
        self.coeff['elongation'] = coeff_elongation
        self.ngran = ngran
        if ephfile is None:
            ephfile = os.path.join(os.getenv('OORB_DATA'), 'de405.dat')
        self.pyephems = PyEphemerides(ephfile)

    def _setOrbits(self, orbitObj):
        """Set the orbits, to be used to generate ephemerides.

        Parameters
        ----------
        orbitObj : Orbits
           The orbits to use to generate ephemerides.
        """
        if not isinstance(orbitObj, Orbits):
            raise ValueError('Need to provide an Orbits object.')
        self.orbitObj = orbitObj
        self.pyephems.setOrbits(self.orbitObj)

    def propagateOrbits(self, newEpoch):
        """Not working yet -- pyoorb problem."""
        raise NotImplementedError
        """
        self.pyephems.propagateOrbits(newEpoch)
        """

    def generateEphemerides(self, times):
        """Generate ephemerides using OpenOrb for all orbits.

        Saves the resulting ephemerides in self.ephems.

        Parameters
        ----------
        times : numpy.ndarray
            The times to use for ephemeris generation.
        """
        return self.pyephems.generateEphemerides(times, obscode=self.obscode, timeScale=self.timeScale, byObject=True)

    def precomputeMultipliers(self):
        """Calculate multipliers for Chebyshev fitting.

        Calculate these once, rather than for each segment.
        """
        # The weights and nPoints are predetermined here, based on Yusra's earlier work.
        self.multipliers = {}
        self.multipliers['position'] = cheb.makeChebMatrix(self.ngran + 1,
                                                           self.coeff['position'], weight=0.16)
        self.multipliers['vmag'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.coeff['vmag'])
        self.multipliers['delta'] = cheb.makeChebMatrix(self.ngran + 1, self.coeff['delta'], weight=0.16)
        self.multipliers['delta_x'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.coeff['delta'])
        self.multipliers['elongation'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.coeff['elongation'])

    def _setGranularity(self, distance_moved):
        """
        Set the first pass values for timestep (for generating ephemerides) and chebyshev segment length.

        If distance is:
        < 0.8 degrees/day  treat same as MBA
        < 1.6 degrees/day  try gen 1 day at 64 points per day.
        < 3.2 deg/day      try gen 0.5 day at 128 points per day
        < 6.4 deg/day       try gen 0.25 day at 256 points per day
        < 12.8 deg.day     try gen 0.125 day at 512 points per day
        < 25.6 deg/day     try gen 0.0625 day at 1024 points per day
        < 51.2 deg/day     try gen 0.03125 day at 2048 points per day
        > try gen 0.015625 day at 4096 points per day
        ngran = 64 always, ngran = int(range/timestep)

        Parameters
        ----------
        distance_moved : float
            Distance moved across the sky, in degrees.
        """
        if distance_moved < 0.8:
            self.timestep = 0.03125  # 1/32 day
        elif distance < 1.6:
            self.timestep = 0.015625  # 1/64 day
        elif distance < 3.2:
            self.timestep = 0.0078125  # 1/128 day
        elif distance < 6.4:
            self.timestep = 0.00390625  # 1/256 day
        elif distance < 12.8:
            self.timestep = 0.001953125  # 1/512 day
        elif distance < 25.6:
            self.timestep = 0.0009765625  # 1/1024 day
        elif distance < 51.2:
            self.timestep = 0.00048828125  # 1/2048 day
        elif distance < 102.4:
            self.timestep = 0.000244140625  # 1/4096 day
        else:  # fastest it can go
            self.timestep = 0.0001220703125  # 1/8192 day
        self.length = self.ngran * self.timestep

    def _updateGranularity(self, p_resid, dec):
        """Update the granularity if the residuals in the position are beyond the tolerance.

        Parameters
        ----------
        p_resid : float
            Maximum positional residual, mas.
        dec : float
            Declination of the object, deg.
        """
        factor = 1.
        if p_resid > 1000:
            factor = 16.
        elif p_resid > 100:
            factor = 8.
        elif p_resid > 15:
            factor = 6.
        elif p_resid > 5:
            factor = 4.
        elif p_resid > 2:
            factor = 2.
        self.timestep = self.timestep / factor
        self.length = self.length / factor
        # cut it in half once more if chance to go over poles
        if dec < -75. or dec > 75.:
            self.timestep = self.timestep/2.
            self.length = self.length/2.

    def calcGranularity(self):
        """Set the typical timestep and segment length for all objects between tStart/tEnd.
        """
        # Generate ephemerides for 1 day, at the midpoint.
        midPoint = self.tStart + self.tDays / 2.0
        times = np.array([midPoint, midPoint+1.0])
        ephs = self.pyephems.generateEphemerides(times, obscode=self.obscode, timeScale=self.timeScale, byObject=False)
        # Calculate the distances that each of these objects moved over a day.
        distances = haversine(np.radians(ephs['ra'][1]), np.radians(ephs['ra'][0]),
                              np.radians(ephs['dec'][1]), np.radians(ephs['dec'][0]))
        distances = np.degrees(distances)
        # Choose the (average? highest 75%? max?) distance.


    def doOneRecursiveSegment(self):
        pass
    # Need to make a version of doOneRecursiveSegment, and think about 'controller' script.
    # controller script would use one instance of this class, then set orbits (all objects)
    # then propagate all orbits to the start of the interval (need to test this)
    # then calculate default ephemerides over desired interval (default timestep)
    # Then in doOneRecursiveSegment, if residuals are too high, can redo ephemerides for that object
    # on a tighter grid and with shorter segment length (count how often this happens?)
