Getting started with exptool v.0.1.1

Ensure that you have a Python distribution with numpy and
astropy. This can be easily accomplished by installing Anaconda
(heartily recommended).

Build the python files in the traditional way

$ cd /where/you/unpacked/Phurt

$ python setup.py install

Make a data reduction script for the object you would like to
reduce. This can be called anything, it will be passed to P-HURT below. P-HURT ships with an
example script in the main Phurt/ directory (reduce.sc.example); copy a version to
whatever directory you'd like to work in. Depending upon your desired operation, this file can hold a variety of
parameters. They are better explained and exemplified in the example script.

The quickest way from Point A (raw data for an object) and Point B (a
median stacked image of the observations of that object in a specified
filter) is done interactively in python.

The moment of truth:

>>> import Phurt

>>> Phurt.read_cals.DivineObject('/path/to/reduce.sc')

This call creates the filelists from reduce.sc to divine the specific
components (bias, flat, science--currently no support for darks. Bug
me if this is important to you). Feel free to stop and use the outputs in IRAF after this
step (IRAF can recognize the output lists). The other reason this is a separate step is that occasionally bad data
can sneak into the calibrations (you had to flush the detector with a
bias after a saturation, for example). Take a moment to make sure that
the files are what you think they are.

(Alternately, you could skip this step and make your own input file
lists in some other way, then do the next step.)

If you want to press ahead to get a reduced image, type (and be a
little patient, though you'll get plenty of updates):

>>> Phurt.reduce.run_all('/path/to/reduce.sc')

Which will create the output files specified in reduce.sc (in
nightdir). It's that easy! This operation currently takes ~3 min start
to finish for 5 bias, 6 flat, 4 science images on a
2010 2GB Macbook Pro (yes I use a laughably weak computer).


There are also many modular pieces you can edit to suit your own
purposes, or execute as standalone modules. These are currently best
understood by examining the code. Please feel free to let me know if
you run into any bugs, don't understand an operation, or find my
secret notes in the code about improvements and would like to add them
to the source.






