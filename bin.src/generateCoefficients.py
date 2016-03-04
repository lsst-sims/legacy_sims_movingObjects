#!/usr/bin/env python

import os
import argparse
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Chebyshev polynomial coefficients" +
                                     " for a set of orbits, over a given timespan.")
    parser.add_argument("--orbitFile", type=str, default=None, help="File containing the orbits.")
    parser.add_argument("--tStart", type=float, default=None, help="Start of timespan to generate coefficients.")
    parser.add_argument("--tEnd", type=float, default=None, help="End of timespan to generate coefficients.")
    parser.add_argument("--tSpan", type=float, default=None, help="Length of timespan (specify this OR tEnd).")
    parser.add_argument("--skyTol", type=float, default=2.5, help="Sky tolerance for position residuals; default 2.5 mas")
    parser.add_argument("--nDecimal", type=int, default=2, help="Number of decimal places to use for timespan.")
    parser.add_argument("--nCoeff", type=int, default=14, help="Number of coefficients to use for the position polynomials.")
    parser.add_argument("--outFile", type=str, default=None, help="Coefficient output file. "
                        + "Default is <orbitFile>_coeffs_tStart_tEnd.")
    args = parser.parse_args()

    if args.orbitFile is None:
        print "Must specify orbit file to use."
        exit()

    if not os.path.isfile(args.orbitFile):
        print "Could not find orbit file %s" %(args.orbitFile)
    if args.orbitFile.endswith('S3M'):
        skiprows = 1
    else:
        skiprows = 0

    orbits = Orbits()
    orbits.readOrbits(args.orbitFile, skiprows=skiprows)

    if args.tStart is None:
        tStart = orbits.orbits.epoch.iloc[0]
        print "tStart was not specified: using the first epoch in the orbits file: %f" %(tStart)
    else:
        tStart = args.tStart

    if (args.tEnd is not None) and (args.tSpan is not None):
        if args.tEnd != args.tStart + args.tSpan:
            print "Provided incompatible values of tEnd and tSpan - please specify one or the other."
            exit()

    if args.tEnd is not None:
        tEnd = args.tEnd
    else:
        if args.tSpan is None:
            print "Neither tEnd nor tSpan was specified: using 30 days."
            tEnd = tStart + 30
        else:
            tEnd = tStart + args.tSpan

    cheb = ChebyFits(orbits, tStart, tEnd, skyTolerance=args.skyTol,
                     nDecimal=args.nDecimal, nCoeff_position=args.nCoeff,
                     ngran=64, nCoeff_vmag=9, nCoeff_delta=5, nCoeff_elongation=6,
                     obscode=807, timeScale='TAI')

    cheb.calcGranularity()
    cheb.calcSegments()

    # Write out coefficients.
    if args.outFile is not None:
        outFile = args.outFile
    else:
        outFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_coeffs_%.2f_%.2f' % (tStart, tEnd)

    with open(outFile, 'w') as f:
        for i, coeff in enumerate(cheb.coeffs):
            #print >>f, "%i %s %.10f %.10f %s %s %s %s %s"%(0, coeff['objId'], coeff['tStart'], coeff['tEnd'],
            print >>f, "%s %.10f %.10f %s %s %s %s %s"%(coeff['objId'], coeff['tStart'], coeff['tEnd'],
                                                        " ".join('%.14e' % j for j in coeff['ra']),
                                                        " ".join('%.14e' % j for j in coeff['dec']),
                                                        " ".join('%.7e' % j for j in coeff['delta']),
                                                        " ".join('%.7e' % j for j in coeff['vmag']),
                                                        " ".join('%.7e' % j for j in coeff['elongation']))

    residFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_resids_%.2f_%.2f' %(tStart, tEnd)
    with open(residFile, 'w') as f:
        for i, resid in enumerate(cheb.resids):
            print >> f, "%s %i %.14f %.14f %.14f %.14e %.14e %.14e %.14e" % (resid['objId'], i+1,
                                                                             resid['tStart'], resid['tEnd'],
                                                                             (resid['tEnd'] - resid['tStart']),
                                                                             resid['pos'], resid['delta'],
                                                                             resid['vmag'], resid['elongation'])


    failedFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_failed_%.2f_%.2f' %(tStart, tEnd)
    with open(failedFile, 'w') as f:
        for i, failed in enumerate(cheb.failed):
            print >>f, failed
