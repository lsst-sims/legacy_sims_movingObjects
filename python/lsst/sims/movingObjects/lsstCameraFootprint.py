import numpy as np
import warnings

from lsst.sims.utils import angularSeparation
from lsst.sims.utils import ModifiedJulianDate
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import chipNameFromRaDecLSST
from lsst.afw.cameraGeom import SCIENCE, WAVEFRONT, GUIDER, FOCUS

__all__ = ['LsstCameraFootprint']


class LsstCameraFootprint(object):
    """
    Class to provide the capability for identifying observations within an LSST camera footprint.
    """
    def __init__(self):
        self.camera = lsst_camera()
        self.ccd_type_dict = {SCIENCE: 'science', WAVEFRONT: 'wavefront',
                              GUIDER: 'guider', FOCUS: 'focus'}


    def inCameraFov(self, ephems, obsData, epoch=2000.0, timeCol='observationStartMJD'):
        """Determine which observations are within the actual camera footprint for a series of observations.

        Parameters
        ----------
        ephems : np.recarray
            Ephemerides for the objects, with RA and Dec as 'ra' and 'dec' columns (in degrees).
        obsData : np.recarray
            Observation pointings, with RA and Dec as 'ra' and 'dec' columns (in degrees).
            The telescope rotation angle should be in 'rotSkyPos' (in degrees), and the time of each
            pointing should be in the 'expMJD' column.
        epoch: float, opt
            The epoch of the ephemerides and pointing data. Default 2000.0.

        Returns
        -------
        np.ndarray
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        # See if the object is within 'rFov' of the center of the boresight.
        sep = angularSeparation(ephems['ra'], ephems['dec'], obsData['ra'], obsData['dec'])
        idxObsRough = np.where(sep <= 2.1)[0]
        # Or go on and use the camera footprint.
        idxObs = []
        for idx in idxObsRough:
            mjd_date = obsData[idx][timeCol]
            mjd = ModifiedJulianDate(TAI=mjd_date)
            obs_metadata = ObservationMetaData(pointingRA=obsData[idx]['ra'],
                                               pointingDec=obsData[idx]['dec'],
                                               rotSkyPos=obsData[idx]['rotSkyPos'],
                                               mjd=mjd)
            # Catch the warnings from astropy about the time being in the future.
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                chipName = chipNameFromRaDecLSST(ra=ephems['ra'][idx],dec=ephems['dec'][idx],
                                                  epoch=epoch, obs_metadata=obs_metadata)
            if chipName != None:
                tt = self.ccd_type_dict[self.camera[chipName].getType()]
                if tt == 'science':
                    idxObs.append(idx)
        idxObs = np.array(idxObs, int)
        return idxObs