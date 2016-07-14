# FADRA: a CPU/GPU Framework for Astronomical Data Reduction and Analysis

## Synopsis

FADRA is a framework for the reduction and analysis of astronomical images. It is designed with the focus of providing GPU-accelerated versions of many common algorithms for astronomical image analysis.
FADRA is currently under development. The modules currently working are for image reduction (CPU and GPU) and light curve obtention through aperture photometry (CPU and GPU). 

## Requirements
Python packages:
• dataproc
• SciPy: https://www.scipy.org/
• PyOpenCL: https://mathema.tician.de/software/pyopencl/
• PyFits: http://www.stsci.edu/institute/software_hardware/pyfits
• warnings: https://docs.python.org/3.1/library/warnings.html
• copy: https://pymotw.com/2/copy/

If GPU reduction or photometry are to be performed: GPU correctly installed with its corresponding drivers to run OpenCL.

## Motivation

GPU acceleration provides acceleration crucial to the development of algorithms in the big data era. Astronomy is only one of the areas which can take advantage of this technology. 

## Example

In the 'example' folder you will find an example.py file along with some data. You can run the example.py file and play with the code to see how the FADRA API works.  

## Contributors

FADRA is open-source. Feel free to join as a contributor and collaborate in the development or report issues and bugs.

## License

FADRA has been released under GNU GPLv3 license.