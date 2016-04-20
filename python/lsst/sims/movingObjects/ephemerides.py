import os
from itertools import repeat
import warnings
import numpy as np
import pandas as pd
import pyoorb as oo
from .orbits import Orbits

import time

__all__ = ['PyOrbEphemerides']


def dtime(time_prev):
    return (time.time() - time_prev, time.time())


class PyOrbEphemerides(object):
    """Generate ephemerides and propagate orbits using the python interface to Oorb.
    Inherits from Orbits and uses parent class to set orbital parameters.
    """
    def __init__(self, ephfile=None):
        # Set translation from timescale to OpenOrb numerical representation.
        # Note all orbits are assumed to be in TT timescale.
        # Also, all dates are expected to be in MJD.
        self.timeScales = {'UTC': 1, 'UT1': 2, 'TT': 3, 'TAI': 4}
        self.elemType = {'COM': 2, 'KEP': 3}

        # Set up oorb. Call this once.
        if ephfile is None:
            ephfile = os.path.join(os.getenv('OORB_DATA'), 'de405.dat')
        oo.pyoorb.oorb_init(ephemeris_fname=ephfile)

        self.orbitObj = None
        self.oorbElem = None

    def setOrbits(self, orbitObj):
        """Set the orbits, to be used to generate ephemerides.

        Immediately calls self._convertOorbElem to (also) save in Oorb format.

        Parameters
        ----------
        orbitObj : Orbits
           The orbits to use to generate ephemerides.
        """
        if not isinstance(orbitObj, Orbits):
            raise ValueError('Need to provide an Orbits object, to validate orbital parameters.')
        if len(orbitObj) == 0:
            raise ValueError('There are no orbits in the Orbit instance.')
        self.orbitObj = orbitObj
        self._convertToOorbElem()

    def _convertToOorbElem(self):
        """Convert orbital elements into the numpy fortran-format array OpenOrb requires
        as input for ephemeris generation.

        The OpenOrb element format is a single array with elemenets:
        0 : orbitId (cannot be a string)
        1-6 : orbital elements, using radians for angles
        7 : element 'type' code (2 = COM, 3 = KEP)
        8 : epoch
        9 : timescale for epoch (1 = UTC, 2 = UT1, 3 = TT, 4 = TAI : always assumes TT)
        10 : magHv
        11 : g

        Sets self.oorbElem, the orbit parameters in an array formatted for OpenOrb.
        """
        # Add the appropriate element and epoch types:
        orbids = np.arange(0, len(self.orbitObj), 1)
        elem_type = np.zeros(len(self.orbitObj)) + self.elemType[self.orbitObj.format]
        epoch_scale = np.zeros(len(self.orbitObj)) + self.timeScales['TT']
        # Convert to format for pyoorb, INCLUDING converting inclination, node, argperi to RADIANS
        if self.orbitObj.format == 'KEP':
            oorbElem = np.column_stack((orbids, self.orbitObj.orbits['a'], self.orbitObj.orbits['e'],
                                        np.radians(self.orbitObj.orbits['inc']),
                                        np.radians(self.orbitObj.orbits['Omega']),
                                        np.radians(self.orbitObj.orbits['argPeri']),
                                        np.radians(self.orbitObj.orbits['meanAnomaly']),
                                        elem_type, self.orbitObj.orbits['epoch'], epoch_scale,
                                        self.orbitObj.orbits['H'], self.orbitObj.orbits['g']))
        elif self.orbitObj.format == 'COM':
            oorbElem = np.column_stack((orbids, self.orbitObj.orbits['q'], self.orbitObj.orbits['e'],
                                        np.radians(self.orbitObj.orbits['inc']),
                                        np.radians(self.orbitObj.orbits['Omega']),
                                        np.radians(self.orbitObj.orbits['argPeri']),
                                        self.orbitObj.orbits['tPeri'], elem_type,
                                        self.orbitObj.orbits['epoch'], epoch_scale,
                                        self.orbitObj.orbits['H'], self.orbitObj.orbits['g']))
        else:
            raise ValueError('Unknown orbit format %s: should be COM or KEP.' % self.orbitObj.format)
        self.oorbElem = oorbElem

    def _convertFromOorbElem(self, oorbElem):
        """Translate pyoorb-style orbital element array back into dataframe.

        Parameters
        ----------
        oorbElem : numpy.ndarray
            The orbital elements in OpenOrb format.

        Returns
        -------
        Orbits
            A new Orbits instance, containing the propagated orbits.
        """
        if self.orbitObj.format == 'KEP':
            newOrbits = pd.DataFrame(self.oorbElem, columns=['objId', 'a', 'e', 'inc', 'Omega', 'argPeri',
                                                             'meanAnomaly', 'elem_type', 'epoch',
                                                             'epoch_type',
                                                             'H', 'g'])
            newOrbits['meanAnomaly'] = np.degrees(newOrbits['meanAnomaly'])
        elif self.orbitObj.format == 'COM':
            newOrbits = pd.DataFrame(self.oorbElem, columns=['objId', 'q', 'e', 'inc', 'Omega', 'argPeri',
                                                             'tPeri', 'elem_type', 'epoch', 'epoch_type',
                                                             'H', 'g'])
        else:
            raise ValueError('Unknown orbit format %s: should be COM or KEP.' % self.orbitObj.format)
        # Convert from radians to degrees.
        newOrbits['inc'] = np.degrees(newOrbits['inc'])
        newOrbits['Omega'] = np.degrees(newOrbits['Omega'])
        newOrbits['argPeri'] = np.degrees(newOrbits['argPeri'])
        # Drop columns we don't need and don't include in our standard columns.
        del newOrbits['elem_type']
        del newOrbits['epoch_type']
        # Swap orbit ids back to original values.
        newOrbits['objId'] = self.orbitObj.orbits['objId'].as_matrix()
        newOrbits['sed_filename'] = self.orbitObj.orbits['sed_filename'].as_matrix()
        # Assign to new Orbit instance.
        newOrb = Orbits()
        newOrb.setOrbits(newOrbits)
        return newOrb

    def _convertTimes(self, times, timeScale='UTC'):
        """Generate an oorb-format array of the times desired for the ephemeris generation.

        Parameters
        ----------
        times : numpy.ndarray or float
            The ephemeris times (MJD) desired
        timeScale : str, optional
            The timescale (UTC, UT1, TT, TAI) of the ephemeris MJD values. Default = UTC, MJD.

        Returns
        -------
        numpy.ndarray
            The oorb-formatted 'ephTimes' array.
        """
        if isinstance(times, float):
            times = np.array([times])
        if len(times) == 0:
            raise ValueError('Got zero times to convert for OpenOrb')
        ephTimes = np.array(zip(times, repeat(self.timeScales[timeScale], len(times))),
                            dtype='double', order='F')
        return ephTimes

    def _generateOorbEphs(self, ephTimes, obscode=807):
        """Generate ephemerides using OOrb.

        Parameters
        ----------
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int, optional
            The observatory code for ephemeris generation. Default=807 (Cerro Tololo).

        Returns
        -------
        numpy.ndarray
            The oorb-formatted ephemeris array.
        """
        oorbEphems, err = oo.pyoorb.oorb_ephemeris(in_orbits=self.oorbElem, in_obscode=obscode,
                                                   in_date_ephems=ephTimes)
        if err != 0:
            warnings.warn('Oorb returned error %s' % (err))
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
        The resulting numpy recarray is composed of columns (of each ephemeris element),
        where each column is 2-d array with first axes either 'object' or 'time'.
        - if byObject = True : [ephemeris elements][object][time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [ephemeris elements][time][object]
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

    def generateEphemerides(self, times, timeScale='UTC', obscode=807, byObject=True,
                            verbose=False):
        """Calculate ephemerides for all orbits at times `times`.

        This is a public method, wrapping self._convertTimes, self._generateOorbEphs
        and self._convertOorbEphs (which include dealing with oorb-formatting of arrays).

        The return ephemerides are in a numpy recarray, with axes
        - if byObject = True : [ephemeris values][object][@time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [ephemeris values][time][@object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length) equals the number of objects)

        The ephemeris values returned to the user (== columns of the recarray) are:
        ['delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt', 'phase', 'solarelon', 'velocity']
        where positions/angles are all in degrees, velocities are deg/day, and delta is the
        distance between the Earth and the object in AU.

        Parameters
        ----------
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int, optional
            The observatory code for ephemeris generation. Default=807 (Cerro Tololo).
        byObject : boolean, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.
        verbose: boolean, optional
            If True, prints time required to calculate ephemerides. Default is False.

        Returns
        -------
        numpy.ndarray
            The ephemeris values, organized as chosen by the user.
        """
        t = time.time()
        ephTimes = self._convertTimes(times, timeScale=timeScale)
        oorbEphs = self._generateOorbEphs(ephTimes, obscode=obscode)
        ephs = self._convertOorbEphs(oorbEphs, byObject=byObject)
        dt, t = dtime(t)
        if verbose:
            print "# Calculating ephemerides for %d objects over %d times required %f seconds"\
                  % (len(self.orbitObj), len(times), dt)
        return ephs

    def propagateOrbits(self, newEpoch):
        """DOES NOT YET WORK DUE TO ERRORS IN PYOORB!!!!
        Propagate orbits from self.orbits.epoch to new epoch (MJD TT).

        Parameters
        ----------
        new_epoch : float
            MJD TT time for new epoch.
        sso : pandas.Dataframe or pandas.Series or numpy.ndarray, optional
            A single or set of rows from self.orbits. Default = None (uses all of self.orbits).

        Returns
        -------
        PyOrbEphemerides
            New PyOrbEphemerides object, containing updated orbital elements for orbits specified by 'sso'.
        """
        newEpoch = self._convertTimes([newEpoch], timeScale='TT')
        newOorbElems, err = oo.pyoorb.oorb_propagation_nb(in_orbits=self.oorbElems, in_epoch=newEpoch)
        if err != 0:
            warnings.warn('Orbit propagation returned error %d' % err)
        # Convert new orbital elements to normal form, and return new Orbits instance.
        newOrbits = PyOrbEphemerides()
        newOrbits.setOrbits(self._convertFromOorbElem(newOorbElems))
        return newOrbits
