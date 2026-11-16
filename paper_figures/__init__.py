"""Figure components for the TRO paper (see README.md).

Importing the package pins the BLAS thread count before NumPy loads: the
equilibrium solves here are small dense systems for which a threaded BLAS is
~1000x slower (see src/blas_threads.py), and the landscape sweeps run one
process per core.
"""
from paper_figures.common.paths import use_src

use_src()
