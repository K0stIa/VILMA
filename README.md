VILMA
=====

VILMA is a algorithm for learning of ordinal classifier from interval annotations.

**Authors:**
  - K.Antoniuk antonkos@cmp.felk.cvut.cz
  - V.Franc xfrancv@cmp.felk.cvut.cz
  - V.Hlavac hlavac@fel.cvut.cz

==================
VILMA installation
==================

**VILMA** is implemented in C++. Internal optimization problem is solved using Oracle
Based Optimization Engine (**OBOE**) implementation of Analytic Center Cutting Plane Method (**ACCPM**)
as a part of COmputatiomal INfrastructure for Operations Research (**COIN-OR**).
As for Bundle Method Risk Minimization (**BMRM**) we use implementation from the `SHOGUN <http://www.shogun-toolbox.org>`_ mahine learning toolbox.

To build **VILMA** you must have installed the follwing software on your PC:
  - **BLAS** 
  - **LAPACK**
  - **Lapackpp** 
  - **aclocal**
  - **autoconf** 
  - **cmake**


To grab code make ``git clone git@github.com:K0stIa/VILMA.git``

.. code:: bash

    mkdir build && cd build
    cmake -DOBOE=/path/to/oboe/lib -DLAPACKPP=/path/to/lapackpp/lib -DCMAKE_CXX_COMPILER=${Your compiler >= gcc-4.7} ..
    make
    cd ..

Tested with cmake (>= 2.8)

You need to have installed ACCPM OBOE library from the next section.

ACCPM installation
------------------

We have addopted original a bit outdated implementation of `ACCPM <https://projects.coin-or.org/OBOE>`_ for VILMA.

..
  ``svn co https://projects.coin-or.org/svn/OBOE/releases/1.0.3 oboe``.

We have published modified code at `Github <https://github.com/K0stIa/OBOE>`_. 
Make ``git clone git@github.com:K0stIa/OBOE.git`` to grab modified code.

To run OBOE for ACCPM you must have following packages installed on your machine:


To install modified version, first you have to install following required packages:
  - **BLAS**
  - **LAPACK**
  - `Lapackpp <http://lapackpp.sourceforge.net>`_
  - **aclocal**
  - **autoconf**
 
`install.sh <https://github.com/K0stIa/OBOE/blob/master/install.sh>`_ script installs ACCPM assuming you have all requireed software


VILMA installation is tested on MacOs Yosemite 10.10.5 and Gentoo Base System release 2.1.
