from __future__ import print_function, division
import os
import numpy as np

from lsst.utils import getPackageDir
from lsst.sims.photUtils import Bandpass
from lsst.sims.photUtils import Sed
from lsst.sims.utils import angularSeparation

from .orbits import Orbits

__all__ = ['fixObsData', 'BaseObs']


# These are the values for seeing and time expected in other parts of this code.


def fixObsData(simData, degreesIn=False):
    """Format opsim data to expected ra/dec/time/rotSkyPos/FWHM column names
    and degree values..

    Parameters
    ----------
    simData : np.recarray
        Array of data from opsim.
    degreesIn : bool, opt
        Is the data in simdata in degrees? Default False (opsimv3).

    Returns
    -------
    np.recarray
    """
    # Create description of new recarray.
    newdtype = simData.dtype.descr
    newdtype += ([('ra', '<f8'), ('dec', '<f8')])
    obsData =  np.empty(simData.shape, dtype=newdtype)
    # Add references to old data.
    for col in simData.dtype.names:
        obsData[col] = simData[col]
    if degreesIn:
        obsData['ra'] = simData['fieldRA']
        obsData['dec'] = simData['fieldDec']
    else:
        obsData['ra'] = np.degrees(simData['fieldRA'])
        obsData['dec'] = np.degrees(simData['fieldDec'])
        obsData['rotSkyPos'] = np.degrees(simData['rotSkyPos'])
    return obsData


class BaseObs(object):
    """
    Base class to generate observations of a set of moving objects.

    Parameters
    ----------
    cameraFootprint : CameraFootprint, opt
        A cameraFootprint class (which provides a way to determine if observations fall into a given
        camera footprint), such as lsstCameraFootprint.
        If none provided, a simple circular FOV is available from the BaseObs class.
    rFov : float, opt
        Radius of the fov, to use for a circular FOV if cameraFootprint is None, in degrees.
        Default 1.75 degrees.
    timeCol : str, opt
        Name of the column (in the simulated pointing history) for Time (MJD).
        Default observationStartMJD.
    seeingCol : str, opt
        Name of the column (in the simulated pointing history) for the physical size of the seeing.
        Default seeingFwhmGeom.
    visitExpTimeCol : str, opt
        Name of the column (in the simulated pointing history) for the visit exposure time.
        Default visitExposureTime.
    """
    def __init__(self, cameraFootprint=None, rFov=1.75,
                 timeCol='observationStartMJD', seeingCol='seeingFwhmGeom',
                 visitExpTimeCol='visitExposureTime'):
        self.cameraFootprint = cameraFootprint
        self.rFov = rFov
        self.timeCol = timeCol
        self.seeingCol = seeingCol
        self.visitExpTimeCol = visitExpTimeCol
        self.colors = {}

    def setOrbits(self, orbitObj):
        """Set the orbital parameters.

        Parameters
        ----------
        orbitObj: lsst.sims.movingObjects.Orbits
            An "Orbits" object with orbital parameter values.
        """
        if isinstance(orbitObj, Orbits):
            self.orbits = orbitObj
        else:
            raise ValueError('Expected an lsst.sims.movingObjects.Orbit object.')

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
            Directory containing the filter throughput curves ('total_*.dat')
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

        Parameters
        ----------
        ephems : np.recarray
            Ephemerides for the objects, with RA and Dec as 'ra' and 'dec' columns (in degrees).
        obsData : np.recarray
            The observation pointings, with RA and Dec as 'ra' and 'dec' columns (in degrees).
        rFov : float, opt
            The radius of the field of view, in degrees.
            Default 2.1 is appropriate for LSST fov if later applying camera footprint.
            A value of 1.75 would be appropriate for simple circular LSST FOV assumption.

        Returns
        -------
        numpy.ndarray
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        sep = angularSeparation(ephems['ra'], ephems['dec'], obsData['ra'], obsData['dec'])
        idxObs = np.where(sep <= rFov)[0]
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
