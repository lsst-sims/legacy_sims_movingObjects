import os
import warnings
import numpy as np
import chebyshevUtils as cheb
from .orbits import Orbits
from .ephemerides import PyOrbEphemerides

__all__ = ['ChebyFits']

def three_sixty_to_neg(ra):
    """Wrap discontiguous RA values into more-contiguous results."""
    if (ra.min() < 100) and (ra.max() > 270):
        ra = np.where(ra > 270, ra - 360, ra)
    return ra

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
        The timescale of the MJD times, tStart/tEnd, and the timeScale that should be
        used with the chebyshev coefficients.
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
    databaseCoeffs : bool, optional
        A flag indicating whether or not (True/False) the output coefficients are headed for the LSST SIMS database.
        If True, the starting/end times of each segment must match *exactly*.
    nDecimal : int, optional
        The number of decimal places to allow in the segment length (and thus the times of the endpoints) can be
        optionally limited to nDecimal places.
        For the LSST SIMS moving object database, this should be 2 decimal places for non-NEOs, 14 for NEOs.
        For any other use, a default of 10 should allow plenty of freedom while still providing a limit for file output.
    """
    def __init__(self, orbitObj, tStart, tEnd, timeScale='TAI',
                 obscode=807, skyTolerance=2.5,
                 nCoeff_position=14, nCoeff_vmag=9, nCoeff_delta=5,
                 nCoeff_elongation=5, ngran=128, ephFile=None, databaseCoeffs=False, nDecimal=10):
        # Set up PyOrbEphemerides.
        if ephFile is None:
            ephFile = os.path.join(os.getenv('OORB_DATA'), 'de405.dat')
        self.pyephems = PyOrbEphemerides(ephFile)
        # And then set orbits.
        self._setOrbits(orbitObj)
        # Save input parameters.
        self.tStart = tStart
        self.tEnd = tEnd
        self.midPoint = self.tStart + (self.tEnd - self.tStart)/2.0
        if timeScale.upper() == 'TAI':
            self.timeScale = 'TAI'
        elif timeScale.upper() == 'UTC':
            self.timeScale = 'UTC'
        elif timeScale.upper() == 'TT':
            self.timeScale = 'TT'
        else:
            raise ValueError('Do not understand timeScale; use TAI, UTC or TT.')
        self.obscode = obscode
        self.skyTolerance = skyTolerance
        self.nCoeff = {}
        self.nCoeff['position'] = nCoeff_position
        self.nCoeff['vmag'] = nCoeff_vmag
        self.nCoeff['delta'] = nCoeff_delta
        self.nCoeff['elongation'] = nCoeff_elongation
        self.ngran = ngran
        self.databaseCoeffs = databaseCoeffs
        self.nDecimal = 14 # 2 for MBA, 14 for NEO
        # Precompute multipliers (we only do this once, instead of per segment).
        self._precomputeMultipliers()

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

    def _precomputeMultipliers(self):
        """Calculate multipliers for Chebyshev fitting.

        Calculate these once, rather than for each segment.
        """
        # The weights and nPoints are predetermined here, based on Yusra's earlier work.
        self.multipliers = {}
        self.multipliers['position'] = cheb.makeChebMatrix(self.ngran + 1,
                                                           self.nCoeff['position'], weight=0.16)
        self.multipliers['vmag'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['vmag'])
        self.multipliers['delta'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['delta'])
        self.multipliers['elongation'] = cheb.makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['elongation'])

    def propagateOrbits(self, newEpoch):
        """Not working yet -- pyoorb problem."""
        raise NotImplementedError
        """
        self.pyephems.propagateOrbits(newEpoch)
        """

    def getAllTimes(self):
        """Return an array of all times for all ephemerides, of all objects.
        """
        try:
            self.timestep
        except AttributeError:
            raise AttributeError('Need to set self.timestep first, using calcGranularity.')
        times = np.arange(self.tStart, self.tEnd + self.timestep / 2.0, self.timestep)
        return times

    def generateEphemerides(self, times, byObject=True, verbose=False):
        """Generate ephemerides using OpenOrb for all orbits.

        Parameters
        ----------
        times : numpy.ndarray
            The times to use for ephemeris generation.
        """
        return self.pyephems.generateEphemerides(times, obscode=self.obscode,
                                                 timeScale=self.timeScale, byObject=byObject,
                                                 verbose=verbose)

    def _roundLength(self, length):
        """Modify length, to fit in an 'integer multiple' within the tStart/tEnd,
        and to have the desired number of decimal values.

        Parameters
        ----------
        length : float
            The input length value to be rounded.

        Returns
        -------
        float
            The rounded length value.
        """
        # Make length an integer value within the time interval.
        int_factor = np.ceil((self.tEnd - self.tStart) / length)
        length = (self.tEnd - self.tStart) / int_factor
        if self.nDecimal is not None:
            length = int(length * 10**(self.nDecimal)) / float(10**self.nDecimal)
        if self.databaseCoeffs:
            if ((self.tEnd - self.tStart) % length) != 0:
                raise ValueError('The start and end times for this segment are incompatible with the'
                                 ' desired length, given that the segment end points must match exactly for database use.')
        return length

    def _testResiduals(self, length, cutoff=85):
        """Calculate the position residual, for a test case.
        Convenience function to make calcGranularity easier to read.
        """
        # The pos_resid used will be the 'cutoff' percentile of all max residuals per object.
        max_pos_resids = np.zeros(len(self.orbitObj), float)
        timestep = length / float(self.ngran)
        # Test for one segment at the midpoint.
        times = np.arange(self.midPoint, self.midPoint + length + timestep / 2.0, timestep)
        # We must regenerate ephemerides here, because the timestep is different each time.
        ephs = self.generateEphemerides(times, byObject=True)
        # Look for the coefficients and residuals.
        for i, e in enumerate(ephs):
            coeff_ra, coeff_dec, max_pos_resids[i] = self._getCoeffsPosition(times, e)
        # Find a representative value and return.
        pos_resid = np.percentile(max_pos_resids, cutoff)
        ratio = pos_resid / self.skyTolerance
        return pos_resid, ratio

    def calcGranularity(self, length=None):
        """Set the typical timestep and segment length for all objects between tStart/tEnd.

        Sets self.length and self.timestep, (length / timestep = self.ngran)
        The resulting segment length will fit into the time period between tStart/tEnd an approximately integer
        multiple of times, and will only have a given number of decimal places.

        Parameters
        ----------
        length : float, optional
            If specified, the length and timestep given by 'length' (plus self.ngran) is
            used, instead of calculating it here.
        """
        # If length is specified, use it and do nothing else.
        if length is not None:
            length = self._roundLength(length)
            pos_resid, ratio = self._testResiduals(length)
            if pos_resid > self.skyTolerance:
                warnings.warn('Will set length and timestep, but this value of length '
                              'produces residuals (%f) > skyTolerance (%f).' % (pos_resid, self.skyTolerance))
            self.length = length
            self.timestep = self.length / float(self.ngran)
            return
        # Otherwise, calculate an appropriate length and timestep.
        # Give a guess at a very approximate segment length, given the skyTolerance,
        # purposefully trying to overestimate this value.
        # The actual behavior of the residuals is not linear with segment length.
        # There is a linear increase at low residuals < ~2 mas / segment length < 2 days
        # Then at around 2 days the residuals blow up, increasing rapidly to about 5000 mas
        #   (depending on orbit .. TNOs, for example, increase but only to about 300 mas,
        #    when the residuals resume ~linear growth out to 70 day segments if ngran=128)
        # Make an arbitrary cap on segment length at 60 days, (25000 mas) ~.5 arcminute accuracy.
        maxLength = 60
        maxIterations = 100
        if self.skyTolerance < 5:
            # This is the cap of the low-linearity regime, looping below will refine this value.
            length = 2.0
        elif self.skyTolerance >= 5000:
            # Make a very rough guess.
            length = np.round((5000.0 / 20.0) * (self.skyTolerance - 5000.)) + 5.0
            length = np.min([maxLength, int(length * 10) / 10.0])
        else:
            # Try to pick a length somewhere in the middle of the fast increase.
            length = 4.0
        # Check the resulting residuals.
        pos_resid, ratio = self._testResiduals(length)
        counter = 1
        #print 'Start', counter, length, pos_resid, ratio
        # Now should be relatively close. Start to zero in using slope around the value.
        while pos_resid > self.skyTolerance and counter <= maxIterations:
            if length > 6.0:
                # In the high residual regime, look for wider gap to avoid fast swings in residual values.
                y = length * 0.1
                dy = y * 2
            elif length < 2.0:
                # In the low residual regime, look for a small gap to avoid getting into the fast-rise if possible.
                y = length * 0.05
                dy = y * 2
            else:
                # In between, try to step by a day.
                y = 1.0
                dy = 2.0
            pos_resid = [0, 0]
            for i, l in enumerate([(length - y), (length + y)]):
                pos_resid[i], ratio = self._testResiduals(l)
            slope = dy / (pos_resid[1] - pos_resid[0])
            length = slope * (self.skyTolerance - pos_resid[0]) + (length - y)
            length = np.min([maxLength, length])
            pos_resid, ratio = self._testResiduals(length)
            counter += 1
            #print 'looping', counter, length, y, pos_resid, ratio
        # Tidy up some characteristics of "length": make it fit an integer number of times into overall timespan,
        # and use a given number of decimal places (easier for database storage).
        length = self._roundLength(length)
        pos_resid, ratio = self._testResiduals(length)
        #print 'final', counter, length, pos_resid, ratio
        self.length = length
        self.timestep = self.length / float(self.ngran)
        if counter > maxIterations:
            raise ValueError('Could not find appropriate segment length and timestep within %d iterations'
                             % maxIterations)

    def _getCoeffsPosition(self, time, ephs):
        """Calculate coefficients for the ra/dec values of a single objects ephemerides.

        Parameters
        ----------
        times : numpy.ndarray
            The times of the ephemerides.
        ephs : numpy.ndarray
            The structured array returned by PyOrbEphemerides holding ephemeris values, for one object.

        Returns
        -------
        numpy.ndarray
            The ra coefficients
        numpy.ndarray
            The dec coefficients
        float
            The positional error residuals between fit and ephemeris values, in mas.
        """
        dradt_coord = ephs['dradt'] / np.cos(np.radians(ephs['dec']))
        coeff_ra, resid_ra, rms_ra_resid, max_ra_resid = cheb.chebfit(time, three_sixty_to_neg(ephs['ra']),
                                                                      dradt_coord,
                                                                      self.multipliers['position'][0],
                                                                      self.multipliers['position'][1],
                                                                      self.nCoeff['position'])
        coeff_dec, resid_dec, rms_dec_resid, max_dec_resid = cheb.chebfit(time, ephs['dec'],
                                                                          ephs['ddecdt'],
                                                                          self.multipliers['position'][0],
                                                                          self.multipliers['position'][1],
                                                                          self.nCoeff['position'])
        max_pos_resid = np.max(np.sqrt(resid_dec**2 +
                                       (resid_ra * np.cos(np.radians(ephs['dec'])))**2))
        # Convert position residuals to mas.
        max_pos_resid *= 3600.0*1000.0
        return coeff_ra, coeff_dec, max_pos_resid

    def _getCoeffsOther(self, time, ephs):
        """Calculate coefficients for the ra/dec values of a single objects ephemerides.

        Parameters
        ----------
        times : numpy.ndarray
            The times of the ephemerides.
        ephs : numpy.ndarray
            The structured array returned by PyOrbEphemerides holding ephemeris values, for one object.

        Returns
        -------
        dict
            Dictionary containing the coefficients for each of 'delta', 'vmag', 'elongation'
        dict
            Dictionary containing the max residual values for each of 'delta', 'vmag', 'elongation'.
        """
        coeffs = {}
        max_resids = {}
        for key, ephValue in zip(('delta', 'vmag', 'elongation'), ('delta', 'magV', 'solarelon')):
            coeffs[key], resid, rms, max_resids[key] = cheb.chebfit(time, ephs[ephValue], None,
                                                                    self.multipliers[key][0],
                                                                    self.multipliers[key][1],
                                                                    self.nCoeff[key])
        return coeffs, max_resids

    def doOneSegment(self, tSegmentStart, tSegmentEnd):
        """Calculate the coefficients for a single Chebyshev segment.

        Parameters
        ----------
        """
        pass
