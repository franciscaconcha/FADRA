.. FADRA documentation master file, created by
   sphinx-quickstart on Wed Jul 13 23:59:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
Welcome to FADRA's documentation!
*********************************
FADRA: A CPU/GPU Framework for Astonomical Data Reduction and Analysis
******************************************************************************************

FADRA is a framework for the reduction and analysis of astronomical images. It is designed with the focus of providing GPU-accelerated versions of many common algorithms for astronomical image analysis.
FADRA is currently under development. The modules currently working are for image reduction (CPU and GPU) and light curve obtention through aperture photometry (CPU and GPU).

Requirements
============
Python packages:

* dataproc
* SciPy: https://www.scipy.org/
* PyOpenCL: https://mathema.tician.de/software/pyopencl/
* PyFits: http://www.stsci.edu/institute/software_hardware/pyfits
* warnings: https://docs.python.org/3.1/library/warnings.html
* copy: https://pymotw.com/2/copy/

If GPU reduction or photometry are to be performed: GPU correctly installed with its corresponding drivers to run OpenCL.


Documentation contents
======================

.. toctree::
   :maxdepth: 1

   example
   modindex
   CPUmath
   reduction
   photometry
   timeseries


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


