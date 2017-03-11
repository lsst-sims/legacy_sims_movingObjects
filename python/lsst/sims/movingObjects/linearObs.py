import os
import warnings
import numpy as np
import pandas as pd

from itertools import repeat
import pyoorb as oo
from scipy import interpolate

import lsst.sims.photUtils.Bandpass as Bandpass
import lsst.sims.photUtils.Sed as Sed

from lsst.sims.utils import haversine
from lsst.sims.utils import ModifiedJulianDate
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import chipNameFromRaDec

from .orbits import Orbits
from .ephemerides import PyOrbEphemerides

__all__ = ['LinearObs', 'runLinearObs']


class LinearObs(Orbits):
    """
    Class to generate observations of a set of moving objects.
    Linear interpolation between gridpoint of ephemerides.
    Inherits from Orbits to read orbits.
    """
    def __init__(self, orbitFile, delim=None, skiprows=None, ephfile=None):
        super(LinearObs, self).__init__()
        self.readOrbits(orbitfile=orbitFile, delim=None, skiprows=None)
        self.pyephem = PyOrbEphemerides(ephfile=ephfile)

    def setTimesRange(self, timeStep=1., timeStart=49353., timeEnd=453003.):
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
        self.ephTimes = np.array(zip(times, repeat(4, len(times))), dtype='double', order='F')

    def setTimes(self, times):
        """
        Set an array for oorb of the ephemeris times desired, given an explicit set of times.
        @ times : numpy array of the actual times of each ephemeris position.
        """
        self.ephTimes = np.array(zip(times, repeat(4, len(times))), dtype='double', order='F')


    def generateEphs(self, sso, obscode=807):
        """Generate ephemerides for all times in self.ephTimes.
        """
        self.pyephem.setOrbits(sso)
        ephs = pyephem.generateEphemerides(self.ephTimes, timeScale='UTC', obscode=obscode, byObject=True)
        return ephs

    # Linear interpolation
    def interpolateEphs(self, ephs, i=0):
        """
        Generate linear interpolations between the quantities in ephs over time.
        """
        interpfuncs = {}
        for n in ephs.dtype.names:
            if n == 'time':
                continue
            interpfuncs[n] = interpolate.interp1d(ephs['time'][i], ephs[n][i], kind='linear',
                                                  assume_sorted=True, copy=False)
        return interpfuncs

    def _setupCamera(self):
        """
        Initialize camera mapper, etc.
        """
        self.mapper = LsstSimMapper()
        self.camera = self.mapper.camera
        self.epoch = 2000.0
        self.cameraFov=np.radians(2.1)

    def ssoInFov(self, interpfuncs, simdata, rFov=np.radians(1.75),
                 useCamera=True,
                 simdataRaCol = 'fieldRA', simdataDecCol='fieldDec'):
        """
        Return the indexes of the simdata observations where the object was inside the fov.
        """
        # See if the object is within 'rFov' of the center of the boresight.
        raSso = np.radians(interpfuncs['ra'](simdata['expMJD']))
        decSso = np.radians(interpfuncs['dec'](simdata['expMJD']))
        sep = haversine(raSso, decSso, simdata[simdataRaCol], simdata[simdataDecCol])
        if not useCamera:
            idxObsRough = np.where(sep<rFov)[0]
            return idxObsRough
        # Or go on and use the camera footprint.
        try:
            self.camera
        except AttributeError:
            self._setupCamera()
        idxObs = []
        idxObsRough = np.where(sep<self.cameraFov)[0]
        for idx in idxObsRough:
            mjd_date = simdata[idx]['expMJD']
            mjd = ModifiedJulianDate(TAI=mjd_date)
            obs_metadata = ObservationMetaData(pointingRA=np.degrees(simdata[idx][simdataRaCol]),
                                               pointingDec=np.degrees(simdata[idx][simdataDecCol]),
                                               rotSkyPos=np.degrees(simdata[idx]['rotSkyPos']),
                                               mjd=mjd)
            raObj = np.array([interpfuncs['ra'](simdata[idx]['expMJD'])])
            decObj = np.array([interpfuncs['dec'](simdata[idx]['expMJD'])])
            # Catch the warnings from astropy about the time being in the future.
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                chipNames = chipNameFromRaDec(ra=raObj,dec=decObj, epoch=self.epoch,
                                              camera=self.camera, obs_metadata=obs_metadata)
            if chipNames != [None]:
                idxObs.append(idx)
        idxObs = np.array(idxObs)
        return idxObs


    def _calcColors(self, sedname='C.dat'):
        """
        Calculate the colors for a moving object with sed 'sedname'.
        """
        # Do we need to read in the LSST bandpasses?
        try:
            self.lsst
        except AttributeError:
            filterdir = os.getenv('LSST_THROUGHPUTS_BASELINE')
            self.filterlist = ('u', 'g', 'r', 'i', 'z', 'y')
            self.lsst ={}
            for f in self.filterlist:
                self.lsst[f] = Bandpass()
                self.lsst[f].readThroughput(os.path.join(filterdir, 'total_'+f+'.dat'))
            self.seddir = os.getenv('SED_DIR')
            if self.seddir is None:
                self.seddir = '.'
            self.vband = Bandpass()
            self.vband.readThroughput(os.path.join(self.seddir, 'harris_V.dat'))
            self.colors = {}
        # See if the sed's colors are in memory already.
        if sedname not in self.colors:
            moSed = Sed()
            moSed.readSED_flambda(os.path.join(self.seddir, sedname))
            vmag = moSed.calcMag(self.vband)
            self.colors[sedname] = {}
            for f in self.filterlist:
                self.colors[sedname][f] = moSed.calcMag(self.lsst[f]) - vmag
        return self.colors[sedname]


    def _calcMagLosses(self, velocity, seeing, texp=30.):
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
                                                ('magV', '<f8'), ('time', '<f8'), ('dradt', '<f8'), ('ddecdt', '<f8'),
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
                raise UserWarning('Could not find filter %s in calculated colors!' %(f))
            match = np.where(simdata[idxObs]['filter'] == f)[0]
            dmagColor[match] = dmagColorDict[f]
        magFilter = ephs['magV'] + dmagColor
        # Calculate trailing and detection loses.
        dmagTrail, dmagDetect = self._calcMagLosses(ephs['velocity'], simdata[seeingCol][idxObs],
                                                    simdata[expTimeCol][idxObs])
        # Turn into a recarray so it's easier below.
        dmags = np.rec.fromarrays([magFilter, dmagColor, dmagTrail, dmagDetect],
                                  names=['magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect'])

        outCols = ['objId',] + list(ephs.dtype.names) + list(simdata.dtype.names) + list(dmags.dtype.names)

        if not self.wroteHeader:
            writestring = ''
            for col in outCols:
                writestring += '%s ' %(col)
            self.outfile.write('%s\n' %(writestring))
            self.wroteHeader = True

        # Write results.
        for eph, simdat, dm in zip(ephs, simdata[idxObs], dmags):
            writestring = '%s ' %(objId)
            for col in ephs.dtype.names:
                writestring += '%s ' %(eph[col])
            for col in simdat.dtype.names:
                writestring += '%s ' %(simdat[col])
            for col in dm.dtype.names:
                writestring += '%s ' %(dm[col])
            self.outfile.write('%s\n' %(writestring))
        self.outfile.flush()


## Function to link the above class methods to generate an output file with moving object observations.

def runLinearObs(orbitfile, outfileName, opsimfile,
            dbcols=None, tstep=2./24., sqlconstraint='',
            rFov=np.radians(1.75), useCamera=True):

    from lsst.sims.maf.db import OpsimDatabase

    # Read orbits.
    moogen = LinearObs(orbitfile)
    print "Read orbit information from %s" %(orbitfile)

    # Check rfov/camera choices.
    if useCamera:
        print "Using camera footprint"
    else:
        print "Not using camera footprint; using circular fov with %f degrees radius" %(np.degrees(rFov))

    # Read opsim database.
    opsdb = OpsimDatabase(opsimfile)
    if dbcols is None:
        dbcols = []
    # Be sure the columns that we need are in place.
    #reqcols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
    #           'visitExpTime', 'finSeeing', 'fiveSigmaDepth', 'solarElong']
    reqcols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
               'visitExpTime', 'FWHMeff', 'FWHMgeom', 'fiveSigmaDepth', 'solarElong']
    for col in reqcols:
        if col not in dbcols:
            dbcols.append(col)
    simdata = opsdb.fetchMetricData(dbcols, sqlconstraint=sqlconstraint)
    print "Queried data from opsim %s, fetched %d visits." %(opsimfile, len(simdata['expMJD']))

    moogen.setTimesRange(timeStep=tstep, timeStart=simdata['expMJD'].min(), timeEnd=simdata['expMJD'].max())
    print "Will generate ephemerides on grid of %f day timesteps, then extrapolate to opsim times." %(tstep)

    for sso in moogen.orbits:
        ephs = moogen.generateEphs(sso)
        interpfuncs = moogen.interpolateEphs(ephs)
        idxObs = moogen.ssoInFov(interpfuncs, simdata, rFov=rFov, useCamera=useCamera)
        moogen.writeObs(sso.orbits['objId'], interpfuncs, simdata, idxObs,
                        sedname=sso.orbits['sed_filename'], outfileName=outfileName)
    print "Wrote output observations to file %s" %(outfileName)

# Test example:
if __name__ == '__main__':
    runLinearObs('pha20141031.des', 'test_allObs.txt', 'enigma_1189_sqlite.db', sqlconstraint='night<365')
