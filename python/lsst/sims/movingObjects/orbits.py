from itertools import repeat
import warnings
import numpy as np
import pandas as pd

__all__ = ['Orbits']


class Orbits(object):
    """Orbits reads and stores orbit parameters for moving objects.
    """
    def __init__(self):
        self.orbits = None
        self.nSso = 0
        self.format = None

        # Specify the required columns/values in the self.orbits dataframe.
        # Which columns are required depends on self.format.
        self.dataCols = {}
        self.dataCols['COM'] = ['objId', 'q', 'e', 'inc', 'Omega', 'argPeri',
                                'tPeri', 'epoch', 'H', 'g', 'sed_filename']
        self.dataCols['KEP'] = ['objId', 'a', 'e', 'inc', 'Omega', 'argPeri',
                                'meanAnomaly', 'epoch', 'H', 'g', 'sed_filename']

    def readOrbits(self, orbitfile, delim=None, skiprows=None):
        """Read orbits from a file, generating a pandas dataframe containing columns matching
        self.dataCols, for the appropriate orbital parameter format (currently accepts COM or KEP formats).

        If objid is not present in the input file, a sequential series of integers will be used.
        If H is not present in the input file, a default value of 20 will be used.
        If g is not present in the input file, a default value of 0.15 will be used.
        If sed_filename is not present in the input file, a default value of 'C.dat' will be used.

        Parameters
        ----------
        orbitfile : str
            The name of the input file containing orbital parameter information.
        delim : str, optional
            The delimiter for the input orbit file -- default = None will use delim_whitespace=True.
        skiprows : int, optional
            The number of rows to skip before reading the header information for pandas.
        """
        self.orbitfile = orbitfile
        # Read the data from disk.
        if delim is None:
            orbits = pd.read_table(orbitfile, delim_whitespace=True, skiprows=skiprows)
        else:
            orbits = pd.read_table(orbitfile, sep=delim, skiprows=skiprows)
        # Normalize the column names to standard values and identify the orbital element types.
        ssoCols = orbits.columns.values.tolist()
        self.nSso = len(orbits)

        # These are the alternative possibilities for various column headers
        # (depending on file version, origin, etc.)
        # that might need remapping from the on-file values to our standardized values.
        altNames = {}
        altNames['objId'] = ['objId', 'objid', '!!ObjID', '!!OID', 'objid(int)', 'full_name', '# name']
        altNames['q'] = ['q']
        altNames['a'] = ['a']
        altNames['e'] = ['e', 'ecc']
        altNames['inc'] = ['inc', 'i', 'i(deg)']
        altNames['Omega'] = ['Omega', 'omega', 'node', 'om', 'node(deg)',
                             'BigOmega', 'Omega/node', 'longNode']
        altNames['argPeri'] = ['argPeri', 'argperi', 'omega/argperi', 'w', 'argperi(deg)']
        altNames['tPeri'] = ['tPeri', 't_p', 'timeperi', 't_peri']
        altNames['epoch'] = ['epoch', 't_0', 'Epoch', 'epoch_mjd']
        altNames['H'] = ['H', 'magH', 'magHv', 'Hv', 'H_v']
        altNames['g'] = ['g', 'phaseV', 'phase', 'gV', 'phase_g']
        altNames['meanAnomaly'] = ['meanAnomaly', 'meanAnom', 'M', 'ma']
        altNames['sed_filename'] = ['sed_filename', 'sed']

        # Update column names that match any of the alternatives above.
        for name, alternatives in altNames.iteritems():
            intersection = list(set(alternatives) & set(ssoCols))
            if len(intersection) > 1:
                raise ValueError('Received too many possible matches to %s in orbit file %s'
                                 % (name, orbitfile))
            if len(intersection) == 1:
                idx = ssoCols.index(intersection[0])
                ssoCols[idx] = name
        # Assign the new column names back to the orbits dataframe.
        orbits.columns = ssoCols

        # Discover which type of orbital parameters we have on disk.
        if 'q' in orbits:
            self.format = 'COM'
        elif 'a' in orbits:
            self.format = 'KEP'
        else:
            raise ValueError('Cannot determine orbital type - neither q nor a in input orbital elements')

        # If these columns are not available in the input data, auto-generate them.
        if 'objId' not in orbits:
            orbits['objId'] = np.arange(0, self.nSso, 1)
        if 'H' not in orbits:
            orbits['H'] = np.zeros(self.nSso) + 20.0
        if 'g' not in orbits:
            orbits['g'] = np.zeros(self.nSso) + 0.15
        if 'sed_filename' not in orbits:
            sedvals = [sed for sed in repeat('C.dat', self.nSso)]
            orbits['sed_filename'] = np.array(sedvals)

        # Make sure we gave all the columns we need.
        for col in self.dataCols[self.format]:
            if col not in orbits:
                raise ValueError('Missing required orbital element %s for orbital format type %s'
                                 % (col, self.format))
        # Check to see if we have duplicates.
        if len(orbits['objId'].unique()) != self.nSso:
            warnings.warn('There are duplicates in the orbit objID values - was this intended? (continuing).')
        # Assign orbits to self.
        self.orbits = orbits
