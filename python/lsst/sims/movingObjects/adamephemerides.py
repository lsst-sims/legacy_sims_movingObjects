from __future__ import print_function
import os
from itertools import repeat
import warnings
import numpy as np
import pandas as pd
from .orbits import Orbits
from .tempAdam import *

import time

__all__ = ['AdamEphemerides']


def dtime(time_prev):
    return (time.time() - time_prev, time.time())


class AdamEphemerides(object):
    """Generate ephemerides and propagate orbits using the python interface to ADAM.
    Inherits from Orbits and uses parent class to set orbital parameters.
    """
    def __init__(self, ephfile=None):
        # Set up / read DE ephemeris file.
        #if ephfile is None:
        #    ephfile = os.path.join(os.getenv('OORB_DATA'), 'de405.dat')
        #oo.pyoorb.oorb_init(ephemeris_fname=ephfile)
        self.orbitObj = None
        self.states = None
        self.initial_state = None

    def setOrbits(self, orbitObj):
        """Set the orbits, to be used to generate ephemerides.

        Immediately calls self._convertOorbElem to translate to state vectors.

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
        self._convertToState()

    def _convertToState(self):
        """Convert orbital elements into the initial state vector.
        """
        self.initial_state = None

    def _convertToOrbElem(self, format='COM', time=None):
        """Translate state vector at given time back to orbital elements in 'format'.
        """
        # This is code from oorbephemerides and needs to be modified to convert from state.
        # Keeping here for now, to show what building the orbits/data would look like.
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

    def generateEphemerides(self, times, timeScale='UTC', obscode='I11', byObject=True,
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
        # DO THINGS HERE
        dt, t = dtime(t)
        if verbose:
            print("# Calculating ephemerides for %d objects over %d times required %f seconds"
                  % (len(self.orbitObj), len(times), dt))
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
        newEpoch = self._convertTimes(newEpoch, timeScale='TT')
        newOorbElems, err = oo.pyoorb.oorb_propagation_nb(in_orbits=self.oorbElem, in_epoch=newEpoch)
        if err != 0:
            warnings.warn('Orbit propagation returned error %d' % err)
        # Convert new orbital elements to normal form, and return new Orbits instance.
        newOrbits = PyOrbEphemerides()
        newOrbits.setOrbits(self._convertFromOorbElem(newOorbElems))
        return newOrbits
