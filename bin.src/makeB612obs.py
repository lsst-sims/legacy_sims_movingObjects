#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np

from lsst.sims.movingObjects import Orbits
from lsst.sims.movingObjects import LsstCameraFootprint
from lsst.sims.movingObjects import LinearObs
from lsst.sims.movingObjects import DirectObs
from lsst.sims.movingObjects import fixObsData

from lsst.sims.maf.db import OpsimDatabase
from lsst.sims.maf.batches import getColMap


def readOpsim(opsimfile, constraint=None, dbcols=None):
    # Read opsim database.
    opsdb = OpsimDatabase(opsimfile)
    if dbcols is None:
        dbcols = []
    colmap = getColMap(opsdb)
    reqcols = [colmap['mjd'], colmap['ra'], colmap['dec'],
               colmap['filter'], colmap['exptime'], colmap['seeingEff'],
               colmap['seeingGeom'], colmap['fiveSigmaDepth']]
    degreesIn = colmap['raDecDeg']
    for col in reqcols:
        if col not in dbcols:
            dbcols.append(col)
    simdata = opsdb.fetchMetricData(dbcols, sqlconstraint=constraint)
    opsdb.close()
    print("Queried data from opsim %s, fetched %d visits." % (opsimfile, len(simdata)))
    simdata = fixObsData(simdata, degreesIn=degreesIn)
    return simdata, colmap

def readOrbits(orbitfile):
    if not os.path.isfile(orbitfile):
        print("Could not find orbit file %s" % (orbitfile))
    orbits = Orbits()
    orbits.readOrbits(orbitfile)
    print("Read orbit information from %s" % (orbitfile))
    return orbits

def setupColors(obs, filterlist):
    # Set up filters
    obs.readFilters(filterlist=filterlist)
    # Calculate all colors ahead of time.
    sednames = np.unique(obs.orbits.orbits['sed_filename'])
    for sedname in sednames:
        obs.calcColors(sedname)
    return obs

def linearObs(orbits, opsimdata, obsFile, cameraFootprint, rFov, obscode, tstep, ephMode, colmap):
    colkwargs = {'timeCol': colmap['mjd'],
                 'seeingCol': colmap['seeingGeom'],
                 'visitExpTimeCol': colmap['exptime']}
    obs = LinearObs(cameraFootprint=cameraFootprint, rFov=rFov, obscode=obscode, timescale='TAI',
                    ephMode=ephMode, **colkwargs)
    # Set orbits.
    obs.setOrbits(orbits)
    # Set up filters
    filterlist = np.unique(opsimdata['filter'])
    obs = setupColors(obs, filterlist)
    # Generate observations and write to output file.
    obs.run(opsimdata, obsFile, tstep=tstep)
    print("Wrote output observations to file %s using linear interpolation." % (obsFile))

def directObs(orbits, opsimdata, obsFile, cameraFootprint, rFov, obscode, ephMode, colmap):
    colkwargs = {'timeCol': colmap['mjd'],
                 'seeingCol': colmap['seeingGeom'],
                 'visitExpTimeCol': colmap['exptime']}
    obs = DirectObs(cameraFootprint=cameraFootprint, rFov=rFov, obscode=obscode, timescale='TAI',
                    ephMode=ephMode, **colkwargs)
    # Set orbits.
    obs.setOrbits(orbits)
    # Set up filters
    filterlist = np.unique(opsimdata['filter'])
    obs = setupColors(obs, filterlist)
    # Generate observations and write to output file.
    obs.run(opsimdata, obsFile)
    print("Wrote output observations to file %s using direct ephemeris generation." % (obsFile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate moving object detections.")
    parser.add_argument("opsimDb", type=str, help="Opsim run db file")
    parser.add_argument("--orbitFile", type=str, default='phas_2k.des',
                        help="File containing the moving object orbits.")
    parser.add_argument("--outDir", type=str,
                        default='.', help="Output directory for moving object detections.")
    parser.add_argument("--obsFile", type=str, default=None,
                        help="Output file name for moving object observations."
                             " Default will build outDir/opsimRun_orbitFile_obs.txt.")
    parser.add_argument("--sqlConstraint", type=str, default='',
                        help="SQL constraint to use to select data from opsimDb. Default no constraint.")
    parser.add_argument("--useCamera", action='store_true', default=False,
                        help="YES use the LSST camera footprint.")
    parser.add_argument("--rFov", type=float, default=5.0,
                        help="When not using camera footprint, use a circular FOV with radius rFOV degrees."
                        " Default 5.0 degrees.")
    parser.add_argument("--interpolation", type=str, default='linear',
                        help="Type of interpolation between ephemerides to use. Options include"
                             " linear, chebyshev, and direct (no interpolation)."
                             " Default is linear.")
    parser.add_argument("--obscode", type=str, default='I11',
                        help="Obscode for generating observations with linear or direct interpolation."
                             " Default is I11 (Cerro Pachon).")
    parser.add_argument("--tStep", type=float, default=2./24.0,
                        help="Timestep between ephemeris generation / linear interpolation steps (in days)."
                             " Relevant for linear interpolation only! Default 2 hours.")
    parser.add_argument("--ephMode", type=str, default='nbody',
                        help="2body or nbody mode for ephemeris generation. Default is nbody.")
    args = parser.parse_args()


    # Check interpolation type.
    if args.interpolation not in ('linear', 'chebyshev', 'direct'):
        print("Must choose linear, chebyshev or direct interpolation method.")
        exit()

    orbitbase = '.'.join(os.path.split(args.orbitFile)[-1].split('.')[:-1])
    opsimRun = os.path.split(args.opsimDb)[-1].replace('_sqlite.db', '').replace('.db', '')

    if args.obsFile is None:
        obsFile = os.path.join(args.outDir, '%s__%s_obs.txt' % (opsimRun, orbitbase))
    else:
        obsFile = os.path.join(args.outDir, args.obsFile)

    # Read orbits.
    orbits = readOrbits(args.orbitFile)

    # Read opsim data
    opsimdata, colmap = readOpsim(args.opsimDb, constraint=args.sqlConstraint, dbcols=None)

    # Set up camera.
    if args.useCamera:
        print("Using camera footprint.")
        cameraFootprint = LsstCameraFootprint()
    else:
        print("Not using camera footprint: using circular fov with %f degrees radius" % (args.rFov))
        cameraFootprint = None


    if args.interpolation == 'linear':
        linearObs(orbits, opsimdata, obsFile, cameraFootprint, args.rFov, args.obscode,
                  args.tStep, args.ephMode, colmap)

    if args.interpolation == 'direct':
        directObs(orbits, opsimdata, obsFile, cameraFootprint, args.rFov, args.obscode, args.ephMode,
                  colmap)
