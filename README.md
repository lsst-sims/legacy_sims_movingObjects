# sims_movingObjects

[Documentation](https://epyc.astro.washington.edu/~lynnej/sims_movingObjects/) 

Installation requires the LSST stack (see [installation instructions](https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF) for 'lsst_sims') as well as the [LSST OpenOrb](https://github.com/lsst/oorb) package. Following the installation and `setup` of these pre-requisites, sims_movingObjects can be installed as follows:

```
git clone git@github.com:/lsst/sims_movingObjects sims_movingobjects
cd sims_movingObjects
eups declare sims_movingObjects -r . -t $USER
setup sims_movingObjects -t $USER
scons
```
