import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

lib= ctypes.CDLL ("./fit_1d-response.so")
#lib.omp_set_num_threads (4)   # if you do not want to use all cores


lib.comp_dft_for_secspec.argtypes= [
    ctypes.c_int,   # ntime
    ctypes.c_int,   # nfreq
    ctypes.c_int,   # nr    Doppler axis
    ctypes.c_double,  # r0
    ctypes.c_double,  # delta r
    ndpointer (dtype=np.float64, flags='CONTIGUOUS', ndim=1), # freqs [nfreq]
    ndpointer (dtype=np.float64, flags='CONTIGUOUS', ndim=1), # src [ntime]
    ndpointer (dtype=np.float64, flags='CONTIGUOUS', ndim=2), # input pow [ntime,nfreq]
    ndpointer (dtype=np.complex128, flags='CONTIGUOUS', ndim=2), # result [nr,nfreq]
    ]




"""
Finally the call goes like this:


dynspec (best to subtract the mean and apply some tapering) must have
shape [ntime,nfreq] and type np.float64 (this should of course be
changed to float32, but this has to be done in python and C)

Along the Doppler axis we compute nr values starting at r0 in steps of
delta_r.  Probably best to check the C code to understand the units.
Frequencies in freqs are in Hz, type np.float64.

For the source position (1-d) as function of time I use an array
src[ntime], type np.float64. I could also have used a linear range as
for the Doppler range, but with an array one can also use an orbit...

The result is a complex array (complex128, which should be changed to
complex64).
"""

# Load data, from 1D image sim

dynspec = np.load('dynspec_u_asym.npy').astype('float64')
ntime = dynspec.shape[0]
nfreq = dynspec.shape[1]
freqs = np.linspace(140, 142, nfreq)

r0 = np.fft.fftfreq(ntime)
delta_r = r0[1] - r0[0]
src = np.linspace(0,1,ntime).astype('float64')

# declare the result array:

result= np.empty( (ntime,nfreq), dtype=np.complex128)

# call the DFT:

lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, freqs,
                          src, dynspec, result)
