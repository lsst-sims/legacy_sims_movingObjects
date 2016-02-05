from itertools import repeat
import warnings
import numpy as np
import pandas as pd

class Orbits(object):
    """Orbits stores orbit parameters for moving objects.
    """
    def __init__(self):
        self.orbits = None
        self.nSso = 0

    def readOrbits(self, orbitfile, delim=None, skiprows=None):
        """Read orbits from a file, generating a pandas dataframe containing columns:

        A set of orbital elements appropriate to generate ephemerides and LSST magnitudes with oorb, one of either:
        'objID q(AU) e inc(deg) node(deg) argPeri(deg) tPeri(MJD TDT) epoch(MJD TDT) H g sed_filename' (COM format)
        'objID q(AU) e inc(deg) node(deg) argPeri(deg) trueAnomaly(deg) epoch(MJD TDT) H g sed_filename' (COT format)
        'objID a(AU) e inc(deg) node(deg) argPeri(deg) meanAnomaly(deg) epoch(MJD TDT) H g sed_filename' (KEP format)

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

        # These are the alternative possibilities for various column headers (depending on file version, origin, etc.)
        # that might need remapping from the on-file values to our standardized values.
        altNames = {}
        altNames['objId'] = ['objId', 'objid', '!!ObjID', '!!OID', 'objid(int)', 'full_name', '# name']
        altNames['q'] = ['q']
        altNames['a'] = ['a']
        altNames['e'] = ['e', 'ecc']
        altNames['inc'] = ['inc', 'i', 'i(deg)']
        altNames['Omega'] = ['Omega', 'omega', 'node', 'om', 'node(deg)', 'BigOmega', 'Omega/node', 'longNode']
        altNames['argPeri'] = ['argPeri', 'argperi', 'omega/argperi', 'w', 'argperi(deg)']
        altNames['tPeri'] = ['tPeri', 't_p', 'timeperi', 't_peri']
        altNames['epoch'] = ['epoch', 't_0', 'Epoch', 'epoch_mjd']
        altNames['H'] = ['H', 'magH', 'magHv', 'Hv', 'H_v']
        altNames['g'] = ['g', 'phaseV', 'phase', 'gV', 'phase_g']
        altNames['meanAnomaly'] = ['meanAnomaly', 'meanAnom', 'M', 'ma']
        altNames['trueAnomaly'] = ['trueAnomaly', 'trueAnom']
        # Update column names that match any of the alternatives above.
        for name, alternatives in altNames.iteritems():
            intersection = list(set(alternatives) & set(ssoCols))
            if len(intersection) > 1:
                raise ValueError('Received too many possible matches to %s in orbit file %s' % (name, orbitfile))
            if len(intersection) == 1:
                idx = ssoCols.index(intersection[0])
                ssoCols[idx] = name
        # Assign the new column names back to the orbits dataframe.
        orbits.columns = ssoCols

        # These are the possible sets of orbital parameters and 'extras' that we need.
        outCols = {}
        outCols['COM'] = ['objId', 'q', 'e', 'inc', 'Omega', 'argPeri', 'tPeri', 'epoch', 'H', 'g', 'sed_filename']
        outCols['COT'] = ['objId', 'q', 'e', 'inc', 'Omega', 'argPeri', 'trueAnomaly', 'epoch', 'H', 'g', 'sed_filename']
        outCols['KEP'] = ['objId', 'a', 'e', 'inc', 'Omega', 'argPeri', 'meanAnomaly', 'epoch', 'H', 'g', 'sed_filename']
        # Discover which type of orbital parameters we have on disk.
        if 'q' in orbits:
            if 'trueAnomaly' in orbits:
                self.format = 'COT'
            else:
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
        for col in outCols[self.format]:
            if col not in orbits:
                raise ValueError('Missing required orbital element %s for orbital format type %s' % (col, self.format))
        # Check to see if we have duplicates.
        if len(orbits['objId'].unique()) != self.nSso:
            warnings.warn('There are duplicates in the orbit objID values - was this intended? (continuing).')
        # Assign orbits to self.
        self.orbits = orbits
