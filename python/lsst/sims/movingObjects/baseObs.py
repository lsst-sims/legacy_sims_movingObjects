import os
import numpy as np
import warnings

from lsst.utils import getPackageDir
from lsst.sims.photUtils import Bandpass
from lsst.sims.photUtils import Sed
from lsst.sims.utils import angularSeparation
from lsst.sims.utils import ModifiedJulianDate
from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import chipNameFromRaDecLSST
from lsst.afw.cameraGeom import SCIENCE, WAVEFRONT, GUIDER, FOCUS

from .ooephemerides import PyOrbEphemerides

__all__ = ['BaseObs']


class BaseObs(object):
    """
    Base class to generate observations of a set of moving objects.

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
    def __init__(self, cameraFootprint=False, rFov=1.75,
                 ephMode='2body', obsCode='I11',
                 obsTimeCol='observationStartMJD', obsTimeScale='TAI',
                 seeingCol='seeingFwhmGeom', visitExpTimeCol='visitExposureTime',
                 obsRA='fieldRA', obsDec='fieldDec', obsRotSkyPos='rotSkyPos', obsDegrees=True):
        self.cameraFootprint = cameraFootprint
        if self.cameraFootprint:
            self.camera = lsst_camera()
            self.ccd_type_dict = {SCIENCE: 'science', WAVEFRONT: 'wavefront',
                                  GUIDER: 'guider', FOCUS: 'focus'}
        self.rFov = rFov
        if ephMode.lower() not in ('2body', 'nbody'):
            raise ValueError('Ephemeris generation must be 2body or nbody.')
        self.ephMode = ephMode
        self.obsCode = obsCode
        self.obsTimeCol = obsTimeCol
        self.obsTimeScale = obsTimeScale
        self.seeingCol = seeingCol
        self.visitExpTimeCol = visitExpTimeCol
        self.obsRA = obsRA
        self.obsDec = obsDec
        self.obsRotSkyPos = obsRotSkyPos
        self.obsDegrees = obsDegrees
        self.colors = {}

    def setupEphemerides(self, ephfile=None):
        """Set up the ephemeris generator, with ephfile. Save the setup PyOrbEphemeris class.

        This uses the default engine, pyoorb - however this could be overwritten to use another generator.

        Parameters
        ----------
        ephfile: str or None, opt
            The name of the planetary ephemerides file to use for ephemeris generation.
            Default is the default for PyOrbEphemerides.
        """
        self.ephems = PyOrbEphemerides(ephfile=ephfile)

    def generateEphemerides(self, sso, times, ephMode=None):
        """Generate ephemerides for 'sso' at times 'times' (assuming MJDs, with timescale self.obsTimeScale).

        The default engine here is pyoorb, however this method could be overwritten to use another ephemeris
        generator, such as ADAM.

        The initialized pyoorb class (PyOrbEphemerides) is saved, to skip setup on subsequent calls.

        Parameters
        ----------
        sso: lsst.sims.movingObjects.Orbits
            Typically this will be a single object.
        times: np.ndarray
            The times at which to generate ephemerides. MJD.
        ephMode: str or None, opt
            Potentially override default ephMode (self.ephMode). Must be '2body' or 'nbody'.

        Returns
        -------
        pandas.Dataframe
            Ephemerides of the sso.
        """
        if ephMode is None:
            ephMode = self.ephMode
        ephTimes = self.ephems._convertTimes(times, timescale=self.obsTimeScale)
        self.ephems.setOrbits(sso)
        if ephMode == '2body':
            oorbEphs = self.ephems._generateOorbEphs2body(ephTimes, obscode=self.obscode)
        else:
            oorbEphs = self.ephems._generateOorbEphs(ephTimes, obscode=self.obscode)
        ephs = self.ephems._convertOorbEphs(oorbEphs, byObject=True)
        return ephs

    def calcTrailingLosses(self, velocity, seeing, texp=30.):
        """Calculate the detection and SNR trailing losses.

        'Trailing' losses = loss in sensitivity due to the photons from the source being
        spread over more pixels; thus more sky background is included when calculating the
        flux from the object and thus the SNR is lower than for an equivalent brightness
        stationary/PSF-like source. dmagTrail represents this loss.

        'Detection' trailing losses = loss in sensitivity due to the photons from the source being
        spread over more pixels, in a non-stellar-PSF way, while source detection is (typically) done
        using a stellar PSF filter and 5-sigma cutoff values based on assuming peaks from stellar PSF's
        above the background; thus the SNR is lower than for an equivalent brightness stationary/PSF-like
        source (and by a greater factor than just the simple SNR trailing loss above).
        dmagDetect represents this loss.

        Parameters
        ----------
        velocity : np.ndarray or float
            The velocity of the moving objects, in deg/day.
        seeing : np.ndarray or float
            The seeing of the images, in arcseconds.
        texp : np.ndarray or float, opt
            The exposure time of the images, in seconds. Default 30.

        Returns
        -------
        (np.ndarray, np.ndarray) or (float, float)
            dmagTrail and dmagDetect for each set of velocity/seeing/texp values.
        """
        a_trail = 0.761
        b_trail = 1.162
        a_det = 0.420
        b_det = 0.003
        x = velocity * texp / seeing / 24.0
        dmagTrail = 1.25 * np.log10(1 + a_trail * x ** 2 / (1 + b_trail * x))
        dmagDetect = 1.25 * np.log10(1 + a_det * x ** 2 / (1 + b_det * x))
        return (dmagTrail, dmagDetect)

    def readFilters(self, filterDir=None,
                    bandpassRoot='total_', bandpassSuffix='.dat',
                    filterlist=('u', 'g', 'r', 'i', 'z', 'y'),
                    vDir=None, vFilter='harris_V.dat'):
        """
        Read (LSST) and Harris (V) filter throughput curves.

        Only the defaults are LSST specific; this can easily be adapted for any survey.

        Parameters
        ----------
        filterDir : str, opt
            Directory containing the filter throughput curves ('total*.dat')
            Default set by 'LSST_THROUGHPUTS_BASELINE' env variable.
        bandpassRoot : str, opt
            Rootname of the throughput curves in filterlist.
            E.g. throughput curve names are bandpassRoot + filterlist[i] + bandpassSuffix
            Default 'total_' (appropriate for LSST throughput repo).
        bandpassSuffix : str, opt
            Suffix for the throughput curves in filterlist.
            Default '.dat' (appropriate for LSST throughput repo).
        filterlist : list, opt
            List containing the filter names to use to calculate colors.
            Default ('u', 'g', 'r', 'i', 'z', 'y')
        vDir : str, opt
            Directory containing the V band throughput curve.
            Default None = $SIMS_MOVINGOBJECTS_DIR/data.
        vFilter : str, opt
            Name of the V band filter curve.
            Default harris_V.dat.
        """
        if filterDir is None:
            filterDir = os.getenv('LSST_THROUGHPUTS_BASELINE')
        if filterDir is None:
            raise ValueError('Please set filterDir or env variable LSST_THROUGHPUTS_BASELINE')
        if vDir is None:
            vDir = os.path.join(getPackageDir('SIMS_MOVINGOBJECTS'), 'data')
        self.filterlist = filterlist
        # Read filter throughput curves from disk.
        self.lsst = {}
        for f in self.filterlist:
            self.lsst[f] = Bandpass()
            self.lsst[f].readThroughput(os.path.join(filterDir, bandpassRoot + f + bandpassSuffix))
        self.vband = Bandpass()
        self.vband.readThroughput(os.path.join(vDir, vFilter))

    def calcColors(self, sedname='C.dat', sedDir=None):
        """Calculate the colors for a given SED.

        If the sedname is not already in the dictionary self.colors, this reads the
        SED from disk and calculates all V-[filter] colors for all filters in self.filterlist.
        The result is stored in self.colors[sedname][filter], so will not be recalculated if
        the SED + color is reused for another object.

        Parameters
        ----------
        sedname : str (opt)
            Name of the SED. Default 'C.dat'.
        sedDir : str (opt)
            Directory containing the SEDs of the moving objects.
            Default None = $SIMS_MOVINGOBJECTS_DIR/data.

        Returns
        -------
        dict
            Dictionary of the colors in self.filterlist for this particular Sed.
        """
        if sedname not in self.colors:
            if sedDir is None:
                sedDir = os.path.join(getPackageDir('SIMS_MOVINGOBJECTS'), 'data')
            moSed = Sed()
            moSed.readSED_flambda(os.path.join(sedDir, sedname))
            vmag = moSed.calcMag(self.vband)
            self.colors[sedname] = {}
            for f in self.filterlist:
                self.colors[sedname][f] = moSed.calcMag(self.lsst[f]) - vmag
        return self.colors[sedname]

    def ssoInCircleFov(self, ephems, obsData, rFov=2.1):
        """Determine which observations are within a circular fov for a series of observations.
        Note that ephems and obsData must be the same length.

        Parameters
        ----------
        ephems : np.recarray
            Ephemerides for the objects.
        obsData : np.recarray
            The observation pointings.
        rFov : float, opt
            The radius of the field of view, in degrees.
            Default 2.1 is appropriate for LSST fov if later applying camera footprint.
            A value of 1.75 would be appropriate for a simple circular LSST FOV assumption.

        Returns
        -------
        numpy.ndarray
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        if not self.obsDegrees:
            sep = angularSeparation(ephems['ra'], ephems['dec'],
                                    np.degrees(obsData[self.obsRA]), np.degrees(obsData[self.obsDec]))
        else:
            sep = angularSeparation(ephems['ra'], ephems['dec'],
                                    obsData[self.obsRA], obsData[self.obsDec])
        idxObs = np.where(sep <= rFov)[0]
        return idxObs

    def ssoInCameraFov(self, ephems, obsData):
        """Determine which observations are within the actual camera footprint for a series of observations.
        Note that ephems and obsData must be the same length.

        Parameters
        ----------
        ephems : np.recarray
            Ephemerides for the objects.
        obsData : np.recarray
            Observation pointings.

        Returns
        -------
        np.ndarray
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        epoch = 2000.0
        # See if the object is within 'rFov' of the center of the boresight.
        idxObsRough = self.ssoInCircleFov(ephems, obsData, rFov=2.1)
        # Then test for the camera footprint exactly.
        idxObs = []
        for idx in idxObsRough:
            mjd_date = obsData[idx][self.obsTimeCol]
            if self.obsTimeScale == 'TAI':
                mjd = ModifiedJulianDate(TAI=mjd_date)
            elif self.obsTimeScale == 'UTC':
                mjd = ModifiedJulianDate(UTC=mjd_date)
            if not self.obsDegrees:
                obs_metadata = ObservationMetaData(pointingRA=np.degrees(obsData[idx][self.obsRA]),
                                                   pointingDec=np.degrees(obsData[idx][self.obsDec]),
                                                   rotSkyPos=np.degrees(obsData[idx][self.obsRotSkyPos]),
                                                   mjd=mjd)
            else:
                obs_metadata = ObservationMetaData(pointingRA=obsData[idx][self.obsRA],
                                                   pointingDec=obsData[idx][self.obsDec],
                                                   rotSkyPos=obsData[idx][self.obsRotSkyPos],
                                                   mjd=mjd)
            # Catch the warnings from astropy about the time being in the future.
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                chipName = chipNameFromRaDecLSST(ra=ephems['ra'][idx], dec=ephems['dec'][idx],
                                                 epoch=epoch, obs_metadata=obs_metadata)
            if chipName != None:
                tt = self.ccd_type_dict[self.camera[chipName].getType()]
                if tt == 'science':
                    idxObs.append(idx)
        idxObs = np.array(idxObs, int)
        return idxObs

    # Put together the output.
    def _openOutput(self, outfileName):
        # Make sure the directory exists to write the output file into.
        outDir = os.path.split(outfileName)[0]
        if len(outDir) > 0:
            if not os.path.isdir(outDir):
                os.makedirs(outDir)
        # Open the output file for writing.
        self.outfile = open(outfileName, 'w')
        self.wroteHeader = False

    def writeObs(self, objId, objEphs, obsData, outfileName='out.txt',
                 sedname='C.dat'):
        """
        Call for each object; write out the observations of each object.

        This method is called once all of the ephemeris values for each observation are calculated;
        the calling method should have already done the matching between ephemeris & simulated observations
        to find the observations where the object is within the specified fov.
        Inside this method, the trailing losses and color terms are calculated and added to the output
        observation file.

        The first time this method is called, a header will be added to the output file.

        Parameters
        ----------
        objId : str or int or float
            The identifier for the object (from the orbital parameters)
        objEphs : numpy.ndarray
            The ephemeris values of the object at each observation.
            Note that the names of the columns are encoded in the numpy structured array,
            and any columns included in the returned ephemeris array will also be propagated to the output.
        obsData : numpy.ndarray
            The observation details from the simulated pointing history, for all observations of
            the object. All columns automatically propagated to the output file.
        outfileName : str, opt
            Output file name. Default 'out.txt'
        sedname : str, out
            The sed_filename for the object (from the orbital parameters).
            Used to calculate the appropriate color terms for the output file.
            Default "C.dat".
        """
        # Return if there's nothing to write out.
        if len(objEphs) == 0:
            return
        # Open file if needed.
        try:
            self.outfile
        except AttributeError:
            self._openOutput(outfileName)
        # Calculate the extra columns we want to write out
        # (dmag due to color, trailing loss, and detection loss)
        # First calculate and match the color dmag term.
        dmagColor = np.zeros(len(obsData), float)
        dmagColorDict = self.calcColors(sedname)
        filterlist = np.unique(obsData['filter'])
        for f in filterlist:
            if f not in dmagColorDict:
                raise UserWarning('Could not find filter %s in calculated colors!' %(f))
            match = np.where(obsData['filter'] == f)[0]
            dmagColor[match] = dmagColorDict[f]
        magFilter = objEphs['magV'] + dmagColor
        # Calculate trailing and detection loses.
        dmagTrail, dmagDetect = self.calcTrailingLosses(objEphs['velocity'],
                                                        obsData[self.seeingCol],
                                                        obsData[self.visitExpTimeCol])
        # Turn into a recarray so it's easier below.
        dmags = np.rec.fromarrays([magFilter, dmagColor, dmagTrail, dmagDetect],
                                  names=['magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect'])

        outCols = ['objId',] + list(objEphs.dtype.names) + list(obsData.dtype.names) + list(dmags.dtype.names)

        if not self.wroteHeader:
            writestring = ''
            for col in outCols:
                writestring += '%s ' %(col)
            self.outfile.write('%s\n' %(writestring))
            self.wroteHeader = True

        # Write results.
        for eph, simdat, dm in zip(objEphs, obsData, dmags):
            writestring = '%s ' %(objId)
            for col in eph.dtype.names:
                writestring += '%s ' %(eph[col])
            for col in simdat.dtype.names:
                writestring += '%s ' %(simdat[col])
            for col in dm.dtype.names:
                writestring += '%s ' %(dm[col])
            self.outfile.write('%s\n' %(writestring))
        self.outfile.flush()
