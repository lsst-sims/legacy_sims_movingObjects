#!/usr/bin/env python

import os
import argparse
from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import ChebyFits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Chebyshev polynomial coefficients" +
                                     " for a set of orbits, over a given timespan.")
    parser.add_argument("--orbitFile", type=str, default=None,
                        help="File containing the orbits.")
    parser.add_argument("--tStart", type=float, default=None,
                        help="Start of timespan to generate coefficients.")
    parser.add_argument("--tEnd", type=float, default=None,
                        help="End of timespan to generate coefficients.")
    parser.add_argument("--tSpan", type=float, default=None,
                        help="Length of timespan (specify this OR tEnd).")
    parser.add_argument("--skyTol", type=float, default=2.5,
                        help="Sky tolerance for position residuals; default 2.5 mas")
    parser.add_argument("--nObj", type=int, default=None,
                        help="If defined, then only nObj coefficients are calculated "
                        "and written to disk at once.")
    parser.add_argument("--length", type=float, default=None,
                        help="Chebyshev polynomial length (will self-determine if not given).")
    parser.add_argument("--nDecimal", type=int, default=2,
                        help="Number of decimal places to use for timespan. Default 2.")
    parser.add_argument("--nCoeff", type=int, default=14,
                        help="Number of coefficients to use for the position polynomials. Default 14.")
    parser.add_argument("--coeffFile", type=str, default=None,
                        help="Coefficient output file. Default is <orbitFile>_coeffs_tStart_tEnd.")
    args = parser.parse_args()

    # Parse orbit file input values.
    if args.orbitFile is None:
        print "Must specify orbit file to use."
        exit()

    if not os.path.isfile(args.orbitFile):
        print "Could not find orbit file %s" %(args.orbitFile)
    if args.orbitFile.lower().endswith('s3m'):
        skiprows = 1
    else:
        skiprows = 0

    # Read orbits.
    orbits = Orbits()
    orbits.readOrbits(args.orbitFile, skiprows=skiprows)

    # Parse start, end and timespan values.
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

    if args.nObj is None:
        nObj = len(orbits)
    else:
        nObj = args.nObj

    # Cycle through nObj at once to fit and write data files.
    append = False
    for n in range(0, len(orbits), nObj):
        subset = orbits.orbits[n:n + nObj]
        subsetOrbits = Orbits()
        subsetOrbits.setOrbits(subset)
        # Fit chebyshev polynomials.
        print "Working on objects %d to %d" % (n, n + nObj)
        cheb = ChebyFits(subsetOrbits, tStart, tEnd, skyTolerance=args.skyTol,
                         nDecimal=args.nDecimal, nCoeff_position=args.nCoeff,
                         ngran=64, nCoeff_vmag=9, nCoeff_delta=5, nCoeff_elongation=6,
                         obscode=807, timeScale='TAI')
        cheb.calcSegmentLength(length=args.length)
        cheb.calcSegments()

        # Set output file names.
        if args.coeffFile is not None:
            coeffFile = args.coeffFile
        else:
            coeffFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_coeffs_%.2f_%.2f' % (tStart, tEnd)
            residFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_resids_%.2f_%.2f' %(tStart, tEnd)
            failedFile = '.'.join(args.orbitFile.split('.')[:-1]) + '_failed_%.2f_%.2f' %(tStart, tEnd)

        # Write out coefficients.
        cheb.write(coeffFile, residFile, failedFile, append=append)
        append = True
