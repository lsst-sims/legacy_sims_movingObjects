import os
import numpy as np
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

import time
def dtime(time_prev):
    return (time.time()-time_prev, time.time())

if __name__ == '__main__':

    testdir = 'testOrbits'
    orbitfiles = ['test_orbitsNEO.s3m', 'test_orbitsMBA.s3m', 'test_orbitsOuter.s3m']
    print "ncoeff ngran otype length timestep resid mean_resid max_resid dt"
    for ocount, ofile in enumerate(orbitfiles):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(testdir, ofile), skiprows=1)
        tStart = orbits.orbits['epoch'].iloc[0]
        # Use a variety of segment length and timestep to see the effect on the residuals.
        length = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 1.75,
                            2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.,
                            11, 12, 13, 14, 15, 17, 18, 20, 22, 25, 27, 30, 35, 40, 45, 50, 75])
        for nCoeff_position in (7, 14, 28):
            for ngran in (32, 64, 128):
                cheb = ChebyFits(orbits, tStart, tStart + 30.,
                                 nCoeff_position=nCoeff_position, ngran=ngran)
                for il, l in enumerate(length):
                    cheb.length = l
                    cheb.timestep = cheb.length / float(cheb.ngran)
                    times = np.arange(tStart, tStart + cheb.length + cheb.timestep / 2.0, cheb.timestep)
                    t = time.time()
                    ephs = cheb.generateEphemerides(times, byObject=True)
                    max_pos_resids = np.zeros(len(ephs), float)
                    for i, e in enumerate(ephs):
                        coeff_ra, coeff_dec, max_pos_resids[i] = cheb._getCoeffsPosition(times, e)
                    dt, t = dtime(t)
                    max_resid = np.percentile(max_pos_resids, 75)
                    print nCoeff_position, ngran, ocount, cheb.length, cheb.timestep, max_resid,\
                            max_pos_resids.mean(), max_pos_resids.max(), dt
        # Residuals tend to be ~linear with length below length=1.5 (although slope varies with orbit)
        # Then they blow up to fast increase as segment length exceeds 2 days
        # Similar behavior for mean, 75%ile, and max.
        # npoints = 128 gives smoother behavior than npoints=64,
        #   at cost of increased time for calculation (need to generate more ephemeris points).
        # There is a big 'bump' upwards between 2-7 days, then things go back to slowly increasing again
