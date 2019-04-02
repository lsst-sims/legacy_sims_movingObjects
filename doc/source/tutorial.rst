====================================================
A Jupyter notebook tutorial: generating observations
====================================================

The classes in sims_movingObjects can be accessed to generate
observations directly, and an example of this is created in the
in-line jupyter notebook below. However, a more typical use-case is
to simply use the high-level script, ``makeLSSTobs.py``.

Documentation on the use of makeLSSTobs.py can be obtained by entering
``makeLSSTobs.py --help`` at the shell prompt. A brief explanation of the various
flags for makeLSSTobs.py is given in the resulting output. A demonstration `notebook`_
shows some examples of the input data and generating observations, step-by-step inside
the makeLSSTobs.py script.

.. _notebook: demo_notebook.ipynb

Some more thoughts on the usage of makeLSSTobs.py:

- Input data includes an opsim database (specified with the ``--opsimDb`` flag) and a set of orbits
  (specified with the ``--orbitFile`` flag).

  - The opsim database: It's easiest to use a full LSST opsim database, but technically this isn't
    actually a requirement.  If using the full makeLSSTobs.py script, the database must be a sqlite database
    with a table called "SummaryAllProps" which contains the observations, and the observations
    must consist of records containing columns named::

      ['observationStartMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
      'visitExposureTime', 'seeingFwhmEff', 'seeingFwhmGeom', 'fiveSigmaDepth', 'solarElong']

 - The orbit file: This is just a text file containing information on the orbits. A variety of orbit
   formats is accomodated, with a variety of headers. The orbits will be fed into the python bindings of
   Oorb, so the basic requirements (for units, etc.) are the same as those for `Oorb orbits`_.


.. _Oorb orbits: https://github.com/oorb/oorb/tree/master/python#defining-orbits



- The ``--sqlConstraint`` flag is probably most likely to be used for constraining tests of the observation
  generation to a small subset of nights. An example would be ``--sqlConstraint 'night < 365'`` to find only
  observations within the first year of the survey. The value of this sqlConstraint will be propagated into
  the output metrics and metadata, if obsMetadata is not set.


- The ``--obsMetadata`` flag is useful to set to describe your input population; whatever is set here will
  be propagated into the names of the output metrics and metadata.


- The ``--footprint`` flag sets the desired footprint for each observation; if an object lands in the
  footprint, then it is output as an observation of that object. The available footprints are 'camera' (the
  true camera footprint), 'circle' (a simple circle, so you then must specify ``--rFov`` for the circle size),
  'rectangle' (a simple rectangle, so then specify ``--xTol`` and ``--yTol``).


Finally, some thoughts on the ``--obsType`` flag, which gives you two very different options on how to
run the observation generation code.

- With ``--obsType linear``, a linear grid of times will be set up from the start to the end of
  observations with intervals of ``--tStep``. The ephemerides for each object calculated at those points, and
  then linear interpolation between those times will be used to decide which fields the object would be
  observed in. The linear interpolation will be used to report the positions of the objects; these
  observations will not generally be suitable for detailed tests for orbit fitting, but statistically are
  fine for representing numbers of observations, etc.

- With ``--obsType direct``, there are two passes to the code: In the first pass, ephemerides are
  calculated at times separated by ``--tStep``, the observations from OpSim are grouped into the same time
  steps, and observations which lie within ``--roughTol`` of the relevant ephemeris position are identified
  as potential observations (i.e. if tStep is 1 and roughTol is 20, then ephemerides are calculated for each
  night, observations are sorted by time to identify their closest ephemeris (i.e. identify observations
  within time +/- 0.5*tstep of each ephemeris), and then observations within 20 deg of their closest
  ephemeris are tagged as 'potential observations). In the second pass, ephemerides are calculated for each
  of the exact times of the potential observations and then matched against the position of the individual
  observation and its footprint (i.e. a circle if that is the ``--footprint`` selected). Thus, there is no
  interpolation at all. These observations would be suitable for detailed tests of orbit fitting, but MAY
  take longer to calculate.

With the ``--obsType direct`` observations, you must specify:

- the mode of ephemeris generation in the first pass (nbody or 2body?). 2body is not a bad option if you
  give it a wider roughTol to account for it, as long as there are no close encounters. This is not true for
  impactors (close encounters!), so ``--prelimEph`` must be set to 'nbody' in that or similar cases.

- the value of the roughTol and the value of the tStep. Obviously, these are correlated. If you have a
  large tStep you should have a larger roughTol to account for the larger distance the object will travel
  between timesteps. The assumption is that you will set roughTol to be about the maximum distance an object
  would move in the time between tSteps.

So how do you know how to set roughTol and tStep? Generally, I'd look at the max velocity of an object at
any time over the lifetime of the survey and then set roughTol accordingly (roughTol = maxVel * tStep).
You SHOULD then be able to set tStep to any value you like although there may be different efficiencies
regarding how many false positives you get in the potential observation list, of course (if roughTol is
large you will be matching against many observations where the object could never really be). How do you
know if you made a mistake in roughTol and tStep? Generally, I'd say look at the logfiles -- there is an
output line that identifies how many of the potential observations (from the first pass) turned out to
really have actual observations of the object (in the second pass) -- if this number is 100% or very close
to 100%, there is some chance you missed some observations.
It is a work in progress to make this roughTol and tStep selection automatic. As part of that work, it is
expected that sims_movingObjects in general will speed up dramatically (Here's hoping Mario and I have some
time available soon!).




