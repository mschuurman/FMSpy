nomad
=====
.. image:: https://travis-ci.org/schuurman-group/nomad.svg?branch=master
  :target: https://travis-ci.org/schuurman-group/nomad

.. image:: https://codecov.io/gh/schuurman-group/nomad/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/schuurman-group/nomad

Nonadiabatic Multistate Adaptive Dynamics

Created Nov. 11, 2015 -- M.S. Schuurman

Requirements
------------
Requires at least Python 3.4, NumPy v1.7.0, SciPy v0.12.0, H5Py v2.5.0 and
MPI4Py v2.0.0.  The `Anaconda package distrubution <https://anaconda.org/>`_
is strongly suggested.

Installation
------------
To create a local nomad directory and compile, use::

    $ git clone https://github.com/mschuurman/nomad.git
    $ cd nomad
    $ python setup.py install

This will also install the nomad driver (`nomad_driver`) and the
checkpoint file extractor (`nomad_extract`) to the path.
