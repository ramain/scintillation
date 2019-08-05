import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

def slow_FT(dynspec, freqs):
    """ Slow FT of dynamic spectrum along points of 
    t*(f / fref), account for phase scaling of f_D
    Given a uniform t axis, this reduces to a regular FT
    
    Uses Olaf's c-implemation if possible, otherwise reverts
    to a slow, pure Python / numpy method

    Parameters
    ----------
    
    dynspec: [time, frequency] ndarray
        Dynamic spectrum to be Fourier Transformed
    f: array of floats
        Frequencies of the channels in dynspec
    """
    
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'fit_1d-response.so')
    lib = ctypes.CDLL(filename)
    #lib= ctypes.CDLL ("./fit_1d-response.so")
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

    ntime = dynspec.shape[0]
    nfreq = dynspec.shape[1]
    r0 = np.fft.fftfreq(ntime)
    delta_r = r0[0] - r0[1]
    src = np.linspace(0,1,ntime).astype('float64')
    src = np.arange(ntime).astype('float64')
    
    # declare the result array:
    result= np.empty( (ntime,nfreq), dtype=np.complex128)

    # Reference freq. to middle of band
    midf = len(freqs)//2
    fref = freqs[midf]
    fscale = freqs / fref
    fscale = fscale.astype('float64')
    
    # call the DFT:
    lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale,
                             src, dynspec, result)

    # Still need to FFT y axis, should change to pyfftw for memory and speed improvement
    result = np.fft.fft(result, axis=1)
    result = np.fft.fftshift(result, axes=1)
    
    return result
