================================================
A brief sims_movingObjects overview
================================================

The sims_movingObjects package generates observations of
moving objects as 'seen' in a simulated pointing history.

The original use-case for sims_movingObjects was to generate
the observations of a small set of moving objects (defined by a set
of orbital parameters), as they would be seen in an LSST Operations
Simulator simulated survey history. With these 'observations', users
can then evaluate metrics about the moving objects via `MAF`_.

.. _MAF: https://sims-maf.lsst.io

This package requires the LSST software stack to run, and uses
the python bindings in the LSST fork of `OpenOrb`_ to generate ephemerides.

.. _OpenOrb: https://github.com/lsst/oorb


There are examples of sims_movingObjects usage as Jupyter notebooks in the
`examples`_ directory.

.. _examples: https://github.com/lsst/sims_movingObjects/examples

