import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

import matplotlib.pyplot as plt

from slowft import slow_FT

def readpsrfits(fname, undo_scaling=True, dedisperse=False):
   
    """Read folded spectrum from psrfits file, based on Dana's code
    Parameters
    ----------
    fname : string
        Fits file to be opened
    undo_scaling : True or False
        Remove the automatic normalization applied by psrfits.  This renders
        output identical to psrchive within machine precision
    dedisperse : True or False
        Roll in phase by closest incoherent time-shift, using the in-header
        values of DM, tbin
    """

    f = fits.open(fname)
    freq = f['SUBINT'].data['DAT_FREQ'][0]
    shape = f['SUBINT'].data[0][-1].shape
    nt = f['SUBINT'].header['NAXIS2']
    nchan = shape[1]

    # Read and polulate data with each subint, one at a time
    data = np.zeros((nt,shape[0],shape[1],shape[2]), dtype=np.float32)
    for i in np.arange(nt):
        data[i,:,:,:] = f['SUBINT'].data[i][-1]
    
    if undo_scaling:
        # Data scale factor (outval=dataval*scl + offs) 
        scl = f['SUBINT'].data['DAT_SCL'].reshape(nt, 4, nchan)
        offs = f['SUBINT'].data['DAT_OFFS'].reshape(nt, 4, nchan)
        data = data * scl[...,np.newaxis] + offs[..., np.newaxis]

    # Remove best integer bin incoherent time shift
    if dedisperse:
        fref = f['HISTORY'].data['CTR_FREQ'][0]
        DM = f['SUBINT'].header['DM']
        tbin = f['HISTORY'].data['Tbin'][0]

        for i in np.arange(len(freq)):
            dt = (1 / 2.41e-4) * DM * (1./fref**2.-1./freq[i]**2.)
            tshift = np.int(np.rint(dt/tbin))
            data[:,:,i,:] = np.roll(data[:,:,i,:],tshift,axis=-1)

    return data, freq


def readpsrarch(fname, dedisperse=False):
    import psrchive
    
    arch = psrchive.Archive_load(fname)
    if dedisperse:
        arch.dedisperse()
    data = arch.get_data()
    freq = arch.get_frequencies()
    return data, freq


def clean_foldspec(f, plots=True, mask=False):
    """
    Clean and rescale folded spectrum
    
    Parameters
    ----------
    f: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots

    Returns folded spectrum, after offset subtraction, scale
    multiplication, and RFI masking
    """
    
    # Sum to form total intensity, mostly a memory/speed concern
    if len(f.shape) == 4:
        print(f.shape)
        f = f[:,(0,1)].mean(1)
    # Offset and scaling in each subint
    offs = np.mean(f, axis=-1)
    scl = np.std(f, axis=-1)
    scl_inv = 1 / (scl)
    scl_inv[np.isinf(scl_inv)] = 0
    
    # Create boolean mask from scale factors
    mask = np.zeros_like(scl_inv)
    mask[scl_inv<0.2] = 0
    mask[scl_inv>0.2] = 1
    
    # Apply scales, sum time, freq, to find profile and off-pulse region
    f_scl = (f - offs[...,np.newaxis]) * scl_inv[...,np.newaxis]
    prof = f_scl.mean(0).mean(0)
    off_gates = np.argwhere(prof < np.median(prof)).squeeze()

    # Re-compute scales from off-pulse region
    offs = np.mean(f[...,off_gates], axis=-1)
    scl = np.std(f[...,off_gates], axis=-1)
    scl_inv = 1 / (scl)
    scl_inv[np.isinf(scl_inv)] = 0
    
    f_scl = (f - offs[...,np.newaxis]) * scl_inv[...,np.newaxis]
    if mask:
        f_scl *= mask[...,np.newaxis]
    
    if plots:
        plt.figure(figsize=(14,4))
        plt.subplot(131)
        plt.plot(f_scl.mean(0).mean(0))
        
        plt.subplot(132)
        plt.title('Manual off-gate offset (log10)')
        plt.xlabel('time (bins)')
        plt.ylabel('freq (bins)')
        plt.imshow(np.log10(offs).T, aspect='auto')

        plt.subplot(133)
        plt.title('Manual off-gate scaling')
        plt.imshow(scl_inv.T, aspect='auto')
        plt.xlabel('time (bins)')

    return f_scl, mask

def create_dynspec(foldspec, profsig=5.):
    """
    Create dynamic spectrum from folded data cube
    
    Uses average profile as a weight, sums over phase

    Parameters 
    ----------                                                                                                         
    foldspec: [time, frequency, phase] array
    profsig: S/N value, mask all profile below this
    """
    
    # Create profile by summing over time, frequency, normalize peak to 1
    template = foldspec.mean(0).mean(0)
    template /= np.max(template)

    # Noise from bottom 50% of profile
    tnoise = np.std(template[template<np.median(template)])
    template[template < tnoise*profsig] = 0
    profplot2 = np.concatenate((template, template), axis=0)

    # Multiply the profile by the template, sum over phase
    dynspec = (foldspec*template[np.newaxis,np.newaxis,:]).mean(-1)
    dynspec /= np.std(dynspec)

    return dynspec

def create_SS(dynspec, freq, pad=1, taper=1):
    """
    Create secondary spectrum from dynamic spectrum,
    using slow FT for frequency correction
    
    Parameters
    ----------
    dynspec: ndarray of floats [time, freq]
    freq: array of floats
    Frequencies of each channel in MHz    
    mask: ndarray of floats [time, freq]
    pad: Boolean - padding factor of the dynspec in time
    taper: Apply a Tukey Window to the dynspec
    """

    if taper:
        import scipy.signal
        t_window = scipy.signal.windows.tukey(dynspec.shape[0], alpha=0.2, sym=True)
        dynspec = dynspec*t_window[:,np.newaxis]
        f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
        dynspec = dynspec*f_window[np.newaxis,:]

    if pad:
        nt = dynspec.shape[0]
        nf = dynspec.shape[1]
        dspec = np.copy(dynspec)
        dynspec = np.zeros( (nt*2, nf) )
        dynspec[nt//2:nt+nt//2, :] += dspec

    SS = slow_FT(dynspec, freq)
    
    return SS


def create_SSwindow(dynspec, freq, mask, pad=1, taper=1):
    """
    Create secondary spectrum from dynamic spectrum,
    using slow FT for frequency correction
    
    Divides out contribution from window function
    
    Parameters
    ----------
    dynspec: ndarray of floats [time, freq]
    freq: array of floats
    Frequencies of each channel in MHz    
    mask: ndarray of floats [time, freq]
    pad: Boolean - padding factor of the dynspec in time
    taper: Apply a Tukey Window to the dynspec
    """

    if taper:
        import scipy.signal
        t_window = scipy.signal.windows.tukey(dynspec.shape[0], alpha=0.2, sym=True)
        dynspec *= t_window[:,np.newaxis]
        f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
        dynspec *= f_window[np.newaxis,:]

    if pad:
        nt = dynspec.shape[0]
        nf = dynspec.shape[1]
        dspec = np.copy(dynspec)
        dynspec = np.zeros( (nt*2, nf) )
        dynspec[nt//2:nt+nt//2, :] += dspec
        m = np.copy(mask)
        mask = np.zeros( (nt*2, nf) )
        mask[nt//2:nt+nt//2, :] += m

    SS = slow_FT(dynspec, freq)
    SS = np.fft.ifftshift(SS)
    SS_ccorr = np.fft.ifft2(abs(SS)**2.0)

    mask_intrinsic = mask * dynspec.mean(-1, keepdims=True)

    SSm = slow_FT(mask_intrinsic, freq)
    SSm = np.fft.ifftshift(SSm)
    SSm_ccorr = np.fft.ifft2(abs(SSm)**2.0)
    SSm_ccorr /= np.max(SSm_ccorr)

    SS_corrected = np.fft.fft2(abs(SS_ccorr) / abs(SSm_ccorr))
    SS_corrected = np.fft.fftshift(SS_corrected)
    
    return abs(SS)


def parabola(x, xs, a, C):
    return a*(x-xs)**2.0 + C

def Hough(SS, ft, tau, srange, masksize=5.0):
    """ Hough transform of secondary spectrum, 
    calculates and returns summed power along a range of parabolae
    
    Parameters
    ----------
    SS: 2D ndarray of floats
        Secondary Spectrum, scaling must be applied before
    
    ft, tau: array of floats
        fringe rate and time delay of the SS
    srange: array of floats
        parabolic curvatures to sum over
    masksize: float
        thickness of parabola in tau to sum in transform
    """
    
    power = []
    for slope in srange:
        SS_masked = np.zeros_like(SS)
        for i in range(len(ft)):
            p1 = parabola(ft[i], 0, slope, 0)
            SS_masked[i,np.argwhere((tau < p1+masksize) & (tau > p1-masksize))] += 1
        SS_power = (SS*SS_masked).sum()
        mask_power = SS_masked.sum()
        power.append(SS_power/mask_power)
        
    return np.array(power)
