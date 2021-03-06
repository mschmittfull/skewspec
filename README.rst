skewspec
=========================================
Measure large-scale structure skew-spectra

The code reads an input catalog of objects or generates a synthetic density
field, and computes skew-spectra corresponding to the contributions to the 
tree-level galaxy bispectrum in redshift space.
For details see https://arxiv.org/abs/2010.14267 (joint work with Azadeh 
Moradinezhad Dizgah).


Running
-------

- The basic usage is as follows.

  .. code-block:: python

    from nbodykit.source.mesh.field import FieldMesh
    from skewspec import smoothing
    from skewspec.skew_spectrum import SkewSpectrum

    # Given an nbodykit CatalogSource object `cat' (e.g. containing a halo
    # catalog), paint the overdensity delta on a 3D mesh using nbodykit.
    delta_mesh = FieldMesh(cat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, 
        window='cic', interlaced=False, compensated=False).compute()-1)

    # Make a copy of the density and apply Gaussian smoothing
    delta_mesh_smoothed = FieldMesh(delta_mesh.compute(mode='real'))
    delta_mesh_smoothed = smoothing.GaussianSmoother(R=20.0).apply_smoothing(
        delta_mesh_smoothed)

    # Compute skew spectra
    LOS = numpy.array([0,0,1])
    skew_spectra = SkewSpectrum.get_list_of_standard_skew_spectra(
        LOS=LOS, redshift_space_spectra=True)
    for skew_spec in skew_spectra:
        # Compute skew spectrum and store in skew_spec.Pskew
        skew_spec.compute_from_mesh(
          mesh=delta_mesh_smoothed,
          second_mesh=delta_mesh_smoothed,
          third_mesh=delta_mesh)



Running from the command line
-----------------------------

- To run the code from the command line, see `main_calc_spectra.py`_. General usage:

  .. code-block:: bash

    $ python main_calc_spectra.py [-h] [--SimSeed SIMSEED] [--boxsize BOXSIZE]
                            [--ApplyRSD APPLYRSD] [--Rsmooth RSMOOTH]
                            [--Ngrid NGRID] [--SubsampleRatio SUBSAMPLERATIO]
                            [--MaxDisplacement MAXDISPLACEMENT]
                            [--DensitySource DENSITYSOURCE] [--b1 B1]
                            [--b2 B2] [--bG2 BG2] [--fLogGrowth FLOGGROWTH]

    optional arguments:
    -h, --help            show this help message and exit
    --SimSeed SIMSEED     Simulation seed to load.
    --boxsize BOXSIZE     Boxsize in Mpc/h.
    --ApplyRSD APPLYRSD   0: No RSD. 1: Include RSD in catalog.
    --Rsmooth RSMOOTH     Smoothing of quad field.
    --Ngrid NGRID         Ngrid used to compute skew spectra.
    --SubsampleRatio SUBSAMPLERATIO
                          Subsample ratio of DM snapshot to use as input.
    --MaxDisplacement MAXDISPLACEMENT
                          Maximum RSD displacement in Mpc/h.
    --DensitySource DENSITYSOURCE
                          Source from which to compute the density. catalog or
                          delta_2SPT
    --b1 B1               b1 bias. Only used if DensitySource=delta_2SPT.
    --b2 B2               b2 bias. Only used if DensitySource=delta_2SPT.
    --bG2 BG2             bG2 bias. Only used if DensitySource=delta_2SPT.
    --fLogGrowth FLOGGROWTH
                          Logarithmic growth factor f. Only used if
                          DensitySource=delta_2SPT.


.. _main_calc_spectra.py: main_calc_spectra.py

- For an example SLURM script to run on a cluster, see `main_calc_spectra.job.helios.spawn.py`_ and use  

  .. code-block:: bash

    $ python main_calc_spectra.job.helios.spawn.py


.. _main_calc_spectra.job.helios.spawn.py: main_calc_spectra.job.helios.spawn.py

- The output is stored in text files.


Jupyter notebooks
-----------------------------

- Notebooks to plot results are in the notebooks folder.


.. _notebooks: ./notebooks/


Installation
------------
The code requires `nbodykit <https://github.com/bccp/nbodykit>`_ version 0.3.x or higher.

To install this it is best to follow the instructions on the nbodykit website.

To install in a new anaconda environment, use for example

.. code-block:: bash

  $ cd ~/anaconda/anaconda/envs
  $ conda create -n nbodykit-0.3.7-env -c bccp -c astropy python=2.7 nbodykit=0.3.7 bigfile pmesh ujson

Newer versions of nbodykit should also work fine. 

To activate the environment, use

.. code-block:: bash

  $ source activate nbodykit-0.3.7-env

To deactivate it, use 

.. code-block:: bash

  $ source deactivate

To run the skewspec code, clone the github repository to a local folder. Then add it to your PYTHONPATH by adding this line to ~/.bash_profile:

.. code-block:: bash

  export PYTHONPATH=/Users/mschmittfull/Dropbox/CODE/skewspec:$PYTHONPATH


Contributing
------------
To contribute, create a fork on github, make changes and commits, and submit a pull request on github.
