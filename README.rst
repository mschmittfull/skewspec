skewspec
=========================================
Measure large-scale structure skew-spectra

The code reads an input catalog of objects or generates a synthetic density
field, and compute skew-spectra corresponding to the contributions to the 
tree-level galaxy bispectrum in redshift space.
For details of the algorithm see https://arxiv.org/abs/TODO.


Running
-------

- The basic usage is like this, assuming delta_mesh is an nbodykit mesh object:

  .. code-block:: python

    from skewspec import smoothing
    from skewspec.skew_spectrum import SkewSpectrum

    # Apply smoothing
    smoothers = [smoothing.GaussianSmoother(R=20.0)]
    delta_mesh_smoothed = FieldMesh(delta_mesh.compute(mode='real'))
    for smoother in smoothers:
        delta_mesh_smoothed = smoother.apply_smoothing(delta_mesh_smoothed)

    # Compute skew spectra
    LOS = numpy.array([0,0,1])
    skew_spectra = SkewSpectrum.get_list_of_standard_skew_spectra(LOS=LOS)
    for skew_spec in skew_spectra:
        # Compute skew spectrum and store in skew_spec.Pskew
        skew_spec.compute_from_mesh(
          mesh=delta_mesh_smoothed, third_mesh=delta_mesh)



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

.. _main_calc_spectra.py: main_calc_spectra.py

- For an example SLURM script to run on a cluster, see `main_calc_spectra.job.helios`_ and use  

  .. code-block:: bash

    $ sbatch scripts/reconstruct_ms_gadget_sim.job.helios

.. main_calc_spectra.job.helios: main_calc_spectra.job.helios

- The output is stored in text files.


Jupyter notebooks
-----------------------------

- Notebooks to plot results are in the `notebooks' folder.

.. notebooks: notebooks/


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
