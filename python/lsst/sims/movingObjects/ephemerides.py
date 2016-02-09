import os
from itertools import repeat
import numpy as np
import pandas as pd
import pyoorb as oo
from .orbits import Orbits

__all__ = ['PyOrbEphemerides']

class PyOrbEphemerides(Orbits):
    """Generate ephemerides and propagate orbits using the python interface to Oorb.
    Inherits from Orbits and uses parent class to set orbital parameters.
    """
    def __init__(self, ephfile=None):
        # Call the parent init.
        super(PyOrbEphemerides, self).__init__()

        # Set translation from timescale to OpenOrb numerical representation.
        # Note all orbits are assumed to be in TT timescale.
        # Also, all dates are expected to be in MJD.
        self.timeScales = {'UTC': 1, 'UT1': 2, 'TT': 3, 'TAI': 4}
        self.elemType = {'COM': 2, 'KEP': 3}

        # Set up oorb. Call this once.
        if ephfile is None:
            ephfile = os.path.join(os.getenv('OORB_DATA'), 'de405.dat')
        oo.pyoorb.oorb_init(ephemeris_fname=ephfile)

    def convertOorbElem(self, sso=None):
        """Convert orbital elements into the numpy fortran-format array OpenOrb requires
        as input for ephemeris generation.

        The OpenOrb element format is a single array with elemenets:
        0 : orbitId (cannot be a string)
        1-6 : orbital elements, using radians for angles
        7 : element 'type' code (2 = COM, 3 = KEP)
        8 : epoch
        9 : timescale for epoch (1 = UTC, 2=UT1, 3=TT, 4=TAI : always assumes TT)
        10 : magHv
        11 : g

        Parameters
        ----------
        sso : pandas.Dataframe or pandas.Series or numpy.ndarray, optional
            A single or set of rows from self.orbits. Default = None (uses all of self.orbits).

        Returns
        -------
        numpy.ndarray
           The numpy array containing the converted orbital elements.
        """
        if sso is None:
            sso = self.orbits
        # Do we have a single item (Series) or multiples (Dataframe)?
        if isinstance(sso, pd.Series):
            # Passed a single SSO in Series.
            nSso = 1
        elif isinstance(sso, pd.DataFrame):
            # Multiple SSO in dataframe.
            nSso = len(sso)
        else:
            # Handle generic cases of numpy ndarray.
            if len(sso.shape) == 0:
                # Single SSO, in a numpy array.
                nSso = 1
            else:
                # Multiple SSSO in numpy array (or something else?).
                nSso = len(sso)
        # Add the appropriate element and epoch types:
        elem_type = self.elemType[self.format]
        epoch_scale = self.timeScales['TT']
        if nSso == 1:
            orbids = 0
        else:
            orbids = np.arange(0, nSso, 1)
            elem_type = np.zeros(nSso) + elem_type
            epoch_scale = np.zeros(nSso) + epoch_scale
        # Convert to format for pyoorb, INCLUDING converting inclination, node, argperi to RADIANS
        if self.format == 'KEP':
            oorbElem = np.column_stack((orbids, sso['a'], sso['e'], np.radians(sso['inc']),
                                        np.radians(sso['Omega']), np.radians(sso['argPeri']),
                                        np.radians(sso['meanAnomaly']), elem_type, sso['epoch'], epoch_scale,
                                        sso['H'], sso['g']))
        else: # assume format = COM
            oorbElem = np.column_stack((orbids, sso['q'], sso['e'], np.radians(sso['inc']),
                                        np.radians(sso['Omega']), np.radians(sso['argPeri']),
                                        sso['tPeri'], elem_type, sso['epoch'], epoch_scale,
                                        sso['H'], sso['g']))
        return oorbElem

    def _convertTimes(self, times, timeScale='UTC'):
        """Generate an oorb-format array of the times desired for the ephemeris generation.

        Parameters
        ----------
        times : numpy.ndarray
            The ephemeris times (MJD) desired
        timeScale : str, optional
            The timescale (UTC, UT1, TT, TAI) of the ephemeris MJD values. Default = UTC, MJD.

        Returns
        -------
        numpy.ndarray
            The oorb-formatted 'ephTimes' array.
        """
        ephTimes = np.array(zip(times, repeat(self.timeScales[timeScale], len(times))),
                            dtype='double', order='F')
        return ephTimes

    def _generateOorbEphs(self, oorbElem, ephTimes, obscode=807):
        """Generate ephemerides using OOrb.

        Parameters
        ----------
        oorbElem : numpy.ndarray
            Orbital elements converted to oorb format (see self.convertOorbElem)
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int, optional
            The observatory code for ephemeris generation. Default=807 (Cerro Tololo).

        Returns
        -------
        numpy.ndarray
            The oorb-formatted ephemeris array.
        """
        oorbEphems, err = oo.pyoorb.oorb_ephemeris(in_orbits=oorbElem, in_obscode=obscode,
                                                   in_date_ephems=ephTimes)
        if err != 0:
            print 'Oorb returned error %s' % (err)
        return oorbEphems

    def _convertOorbEphs(self, oorbEphs, byObject=True):
        """Converts oorb ephemeris array to pandas dataframe, with labelled columns.

        The oorb ephemeris array is a 3-d array organized as: (object / times / eph@time)
        [objid][time][ephemeris information @ that time] with ephemeris elements
        0 : distance (geocentric distance)
        1 : ra (deg)
        2 : dec (deg)
        3 : mag
        4 : ephem mjd
        5 : ephem mjd timescale
        6 : dra/dt (deg/day) sky motion
        7 : ddec/dt (deg/day) sky motion
        8 : phase angle (deg)
        9 : solar elongation angle (deg)

        Here we convert to a numpy recarray, grouped either by object (default)
        or by time (if byObject=False).
        The resulting numpy recarray is a 3-d array with axes
        - if byObject = True : [object][ephemeris elements][@time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [time][ephemeris elements][@object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length) equals the number of objects)

        Parameters
        ----------
        oorbEphs : numpy.ndarray
            The oorb-formatted ephemeris values
        byObject : boolean, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        numpy.ndarray
            The re-arranged ephemeris values, in a 3-d array.
        """
        ephs = np.swapaxes(oorbEphs, 2, 0)
        velocity = np.sqrt(ephs[6]**2 + ephs[7]**2)
        if byObject:
            ephs = np.swapaxes(ephs, 2, 1)
            velocity = np.swapaxes(velocity, 1, 0)
        # Create a numpy recarray.
        ephs = np.rec.fromarrays([ephs[0], ephs[1], ephs[2], ephs[3], ephs[4],
                                  ephs[6], ephs[7], ephs[8], ephs[9], velocity],
                                 names=['delta', 'ra', 'dec', 'magV', 'time', 'dradt',
                                        'ddecdt', 'phase', 'solarelon', 'velocity'])
        return ephs


    def generateEphemerides(self, oorbElem, times, timeScale='UTC', obscode=807, byObject=True):
        """Convenience method wrapping self._convertTimes, self._generateOorbEphs and self._convertOorbEphs into one step.

        Parameters
        ----------
        oorbElem : numpy.ndarray
            Orbital elements converted to oorb format (see self.convertOorbElem)
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int, optional
            The observatory code for ephemeris generation. Default=807 (Cerro Tololo).
         byObject : boolean, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        numpy.ndarray
            The re-arranged ephemeris values, in a 3-d array.
        """
        ephTimes = self._convertTimes(times, timeScale=timeScale)
        oorbEphs = self._generateOorbEphs(oorbElem, ephTimes, obscode=obscode)
        ephs = self._convertOorbEphs(oorbEphs, byObject=byObject)
        return ephs

    def propagateOrbits(self):
        pass
