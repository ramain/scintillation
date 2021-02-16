import numpy as np
import os

def slow_FT(dynspec, freqs):
    """
    Slow FT of dynamic spectrum along points of
    t*(f / fref), account for phase scaling of f_D.
    Given a uniform t axis, this reduces to a regular FT

    Uses Olaf's c-implemation if possible, otherwise reverts
    to a slow, pure Python / numpy method

    Reference freq is currently hardcoded to the middle of the band

    Parameters
    ----------

    dynspec: [time, frequency] ndarray
        Dynamic spectrum to be Fourier Transformed
    f: array of floats
        Frequencies of the channels in dynspec
    """

    from numpy.ctypeslib import ndpointer
    import ctypes

    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'fit_1d-response.so')
    lib = ctypes.CDLL(filename)
    # lib.omp_set_num_threads (4)   # if you do not want to use all cores

    lib.comp_dft_for_secspec.argtypes = [
        ctypes.c_int,   # ntime
        ctypes.c_int,   # nfreq
        ctypes.c_int,   # nr    Doppler axis
        ctypes.c_double,  # r0
        ctypes.c_double,  # delta r
        ndpointer(dtype=np.float64,
                  flags='CONTIGUOUS', ndim=1),  # freqs [nfreq]
        ndpointer(dtype=np.float64,
                  flags='CONTIGUOUS', ndim=1),  # src [ntime]
        ndpointer(dtype=np.float64,
                  flags='CONTIGUOUS', ndim=2),  # input pow [ntime,nfreq]
        ndpointer(dtype=np.complex128,
                  flags='CONTIGUOUS', ndim=2),  # result [nr,nfreq]
    ]

    # cast dynspec as float 64
    dynspec = dynspec.astype(np.float64)

    ntime = dynspec.shape[0]
    nfreq = dynspec.shape[1]
    r0 = np.fft.fftfreq(ntime)
    delta_r = r0[1] - r0[0]
    src = np.linspace(0, 1, ntime).astype('float64')
    src = np.arange(ntime).astype('float64')

    # declare the empty result array:
    SS = np.empty((ntime, nfreq), dtype=np.complex128)

    # Reference freq. to middle of band, should change this
    midf = len(freqs)//2
    fref = freqs[midf]
    fscale = freqs / fref
    fscale = fscale.astype('float64')

    # call the DFT:
    if os.path.isfile(filename):
        print("Computing slow FT using C-implementation, fit_1d-response")
        lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale,
                                 src, dynspec, SS)

        # flip along time
        SS = SS[::-1]
    else:
        print("C-implentation fit_1d-response not installed, "
              "computing slowFT with numpy")
        ft = np.fft.fftfreq(ntime, 1)

        # Scaled array of t * f/fref
        tscale = src[:, np.newaxis]*fscale[np.newaxis, :]
        FTphase = -2j*np.pi*tscale[:, np.newaxis, :] * \
            ft[np.newaxis, :, np.newaxis]
        SS = np.sum(dynspec[:, np.newaxis, :]*np.exp(FTphase), axis=0)
        SS = np.fft.fftshift(SS, axis=0)

    # Still need to FFT y axis, should change to pyfftw for memory and
    #   speed improvement
    SS = np.fft.fft(SS, axis=1)
    SS = np.fft.fftshift(SS, axes=1)

    return SS

