from __future__ import print_function
import os
import warnings

import lsst.sims.photUtils.Bandpass as Bandpass
import lsst.sims.photUtils.Sed as Sed

from lsst.sims.utils import haversine
from lsst.sims.utils import ModifiedJulianDate
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import chipNameFromRaDec

from lsst.sims.maf.db import OpsimDatabase

__all__ = ['LsstObs']

class LsstObs(object):
    """
    Class to generate actual LSST observations of a set of moving objects.
    Currently uses ChebyShev polynomials fit, should allow simple linear interpolation too.
    """
    def __init__(self, logfile='obslog', cameraFootprint=True,
                 filterDir=None, sedDir=None):
        self.logfile = open(logfile, 'w')
        # Set up camera object (used for footprint).
        if cameraFootprint:
            self._setupCamera()
        else:
            self.mapper = None
            self.camera = None
            self.epoch = 2000.0
            self.cameraFov = np.radians(1.75)
        # Set up dictionary to store colors of various object seds.
        self._setupFilters(filterDir=filterDir, vDir=sedDir)
        self.colors = {}


    def _setupCamera(self):
        """
        Initialize LSST camera mapper.
        """
        self.mapper = LsstSimMapper()
        self.camera = self.mapper.camera
        self.epoch = 2000.0
        self.cameraFov=np.radians(2.1)

    def _setupFilters(self, filterDir=None, vDir=None,
                      filterlist=('u', 'g', 'r', 'i', 'z', 'y')):
        """
        Read LSST and Harris (V) filters.

        Parameters
        ----------
        filterDir : str (opt)
            Directory containing the filter throughput curves ('total_*.dat')
            Default set by 'LSST_THROUGHPUTS_BASELINE' env variable.
        vDir : str (opt)
            Directory containing the V band throughput curve.
            Default set by 'SED_DIR' env variable.
        filterlist : list (opt)
            List containing the filter names to use to calculate colors.
            Default ('u', 'g', 'r', 'i', 'z', 'y')
        """
        if filterDir is None:
            filterDir = os.getenv('LSST_THROUGHPUTS_BASELINE')
        if filterDir is None:
            raise ValueError('Please set filterDir or env variable LSST_THROUGHPUTS_BASELINE')
        if vDir is None:
            vDir = os.getenv('SED_DIR')
        if vDir is None:
            raise ValueError('Please set vDir or env variable SED_DIR')
        self.filterlist = filterlist
        # Read filter throughput curves from disk.
        self.lsst = {}
        for f in self.filterlist:
            self.lsst[f] = Bandpass()
            self.lsst[f].readThroughput(os.path.join(filterDir, 'total_' + f + '.dat'))
        self.vband = Bandpass()
        self.vband.readThroughput(os.path.join(vDir, 'harris_V.dat'))

    def readOpsim(self, opsimfile, sqlconstraint=None, dbcols=None):
        # Read opsim database.
        opsdb = OpsimDatabase(opsimfile)
        if dbcols is None:
            dbcols = []
        # Be sure the minimum columns that we need are in place.
        # reqcols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
        #           'visitExpTime', 'finSeeing', 'fiveSigmaDepth', 'solarElong']
        reqcols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
                   'visitExpTime', 'FWHMeff', 'FWHMgeom', 'fiveSigmaDepth', 'solarElong']
        for col in reqcols:
            if col not in dbcols:
                dbcols.append(col)
        simdata = opsdb.fetchMetricData(dbcols, sqlconstraint=sqlconstraint)
        print("Queried data from opsim %s, fetched %d visits." % (opsimfile, len(simdata['expMJD'])),
              file=self.logfile)

    def _calcColors(self, sedname='C.dat'):
        """
        Calculate the colors for a given SED.

        Parameters
        ----------
        sedname : str (opt)
            Name of the SED. Default 'C.dat'.
        """

        # See if the sed's colors are in memory already.
        if sedname not in self.colors:
            moSed = Sed()
            moSed.readSED_flambda(os.path.join(self.sedDir, sedname))
            vmag = moSed.calcMag(self.vband)
            self.colors[sedname] = {}
            for f in self.filterlist:
                self.colors[sedname][f] = moSed.calcMag(self.lsst[f]) - vmag
        return

    def calcMagLosses(self, velocity, seeing, texp=30.):
        """
        Calculate the magnitude losses due to trailing and not matching the point-source detection filter.
        """
        a_trail = 0.76
        b_trail = 1.16
        a_det = 0.42
        b_det = 0.00
        x = velocity * texp / seeing / 24.0
        dmagTrail = 1.25 * np.log10(1 + a_trail*x**2/(1+b_trail*x))
        dmagDetect = 1.25 * np.log10(1 + a_det*x**2 / (1+b_det*x))
        return dmagTrail, dmagDetect

    def _openOutput(self, outfileName):
        self.outfile = open(outfileName, 'w')
        self.wroteHeader = False


    #REWRITE
    def writeObs(self, objId, interpfuncs, simdata, idxObs, outfileName='out.txt',
                 sedname='C.dat', tol=1e-8,
                 seeingCol='FWHMgeom', expTimeCol='visitExpTime'):
        """
        Call for each object; write out the observations of each object.
        """
        # Return if there's nothing to write out.
        if len(idxObs) == 0:
            return
        # Open file if needed.
        try:
            self.outfile
        except AttributeError:
            self._openOutput(outfileName)
        # Calculate the ephemerides for the object, using the interpfuncs, for the times in simdata[idxObs].
        tvis = simdata['expMJD'][idxObs]
        ephs = np.recarray([len(tvis)], dtype=([('delta', '<f8'), ('ra', '<f8'), ('dec', '<f8'),
                                                ('magV', '<f8'), ('time', '<f8'), ('dradt', '<f8'),
                                                ('ddecdt', '<f8'),
                                                ('phase', '<f8'), ('solarelon', '<f8'), ('velocity', '<f8')]))
        for n in interpfuncs:
            ephs[n] = interpfuncs[n](tvis)
        ephs['time'] = tvis
        # Calculate the extra columns we want to write out
        # (dmag due to color, trailing loss, and detection loss)
        # First calculate and match the color dmag term.
        dmagColor = np.zeros(len(idxObs), float)
        dmagColorDict = self._calcColors(sedname)
        filterlist = np.unique(simdata[idxObs]['filter'])
        for f in filterlist:
            if f not in dmagColorDict:
                raise UserWarning('Could not find filter %s in calculated colors!' % (f))
            match = np.where(simdata[idxObs]['filter'] == f)[0]
            dmagColor[match] = dmagColorDict[f]
        magFilter = ephs['magV'] + dmagColor
        # Calculate trailing and detection loses.
        dmagTrail, dmagDetect = self._calcMagLosses(ephs['velocity'], simdata[seeingCol][idxObs],
                                                    simdata[expTimeCol][idxObs])
        # Turn into a recarray so it's easier below.
        dmags = np.rec.fromarrays([magFilter, dmagColor, dmagTrail, dmagDetect],
                                  names=['magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect'])

        outCols = ['objId', ] + list(ephs.dtype.names) + list(simdata.dtype.names) + list(dmags.dtype.names)

        if not self.wroteHeader:
            writestring = ''
            for col in outCols:
                writestring += '%s ' % (col)
            self.outfile.write('%s\n' % (writestring))
            self.wroteHeader = True

        # Write results.
        for eph, simdat, dm in zip(ephs, simdata[idxObs], dmags):
            writestring = '%s ' % (objId)
            for col in ephs.dtype.names:
                writestring += '%s ' % (eph[col])
            for col in simdat.dtype.names:
                writestring += '%s ' % (simdat[col])
            for col in dm.dtype.names:
                writestring += '%s ' % (dm[col])
            self.outfile.write('%s\n' % (writestring))
        self.outfile.flush()
