from __future__ import division, print_function
import numpy as np
import warnings

from lsst.sims.utils import angularSeparation
from lsst.sims.utils import ModifiedJulianDate
from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import chipNameFromRaDecLSST


__all__ = ['LsstCameraFootprint']

stdTimeCol = 'expMJD'

class LsstCameraFootprint(object):
    """
    Class to provide the capability for identifying observations within an LSST camera footprint.
    """
    def inCameraFov(self, ephems, obsData, epoch=2000.0):
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
            mjd_date = obsData[idx][stdTimeCol]
            mjd = ModifiedJulianDate(TAI=mjd_date)
            obs_metadata = ObservationMetaData(pointingRA=obsData[idx]['ra'],
                                               pointingDec=obsData[idx]['dec'],
                                               rotSkyPos=obsData[idx]['rotSkyPos'],
                                               mjd=mjd)
            # Catch the warnings from astropy about the time being in the future.
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                chipNames = chipNameFromRaDecLSST(ra=ephems['ra'][idx],dec=ephems['dec'][idx],
                                                  epoch=epoch, obs_metadata=obs_metadata)
            if chipNames != [None]:
                idxObs.append(idx)
        idxObs = np.array(idxObs, int)
        return idxObs