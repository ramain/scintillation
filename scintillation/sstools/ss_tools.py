import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

import matplotlib.pyplot as plt

from slowft import slow_FT

def readpsrfits(fname, undo_scaling=True, dedisperse=False, verbose=True):
   
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
        values of DM, tbin.  Phase 0 point is arbitrary

    Returns: Data cube [time, pol, freq, phase], 
             frequency array [freq], 
             time array [time]
    """

    f = fits.open(fname)
    freq = f['SUBINT'].data['DAT_FREQ'][0]
    F = freq*u.MHz
    
    shape = f['SUBINT'].data[0][-1].shape
    nt = f['SUBINT'].header['NAXIS2']
    nchan = shape[1]

    # Exact start time is encompassed in three header values
    t0 = Time(f['PRIMARY'].header['STT_IMJD'], format='mjd', precision=9)
    t0 += (f['PRIMARY'].header['STT_SMJD'] + f['PRIMARY'].header['STT_OFFS']) *u.s
    # Array of subint lengths, 
    Tint = f['SUBINT'].data['TSUBINT']*u.s
    T = t0 + np.cumsum(Tint)
    
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

    # Print some basic obs. info
    if verbose:
        print("Observed with {0}".format(f['Primary'].header['TELESCOP']) )
        print("Source: {0}".format(f['Primary'].header['SRC_NAME']) )
        print("fref: {0}, chans: {1}".format(fref, (F[1]-F[0]) ) )
        print("t0: {0}, tend: {1} \n \n".format(T[0].isot, T[-1].isot) )      

    return data, F, T


def readpsrarch(fname, dedisperse=True):
    """
    Read pulsar archive directly using psrchive
    
    Parameters
    ----------
    fname: string
    file directory
    dedisperse: Bool
    apply psrchive's by-channel incoherent de-dispersion

    Returns archive data cube, frequency array
    """
    import psrchive
    
    arch = psrchive.Archive_load(fname)
    if dedisperse:
        arch.dedisperse()
    data = arch.get_data()
    freq = arch.get_frequencies()
    return data, freq


def clean_foldspec(I, plots=True, apply_mask=False, rfimethod='var'):
    """
    Clean and rescale folded spectrum
    
    Parameters
    ----------
    I: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots
    apply_mask: Bool
    Multiply dynamic spectrum by mask
    rfimethod: String
    RFI flagging method, currently only supports var

    Returns folded spectrum, after bandpass division, 
    off-gate subtractionand RFI masking
    """

    # Sum to form total intensity, mostly a memory/speed concern
    if len(I.shape) == 4:
        print(I.shape)
        I = I[:,(0,1)].mean(1)

    # Use median over time to not be dominated by outliers
    bpass = np.median( I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass
    
    mask = np.ones_like(I.mean(-1))

    if rfimethod == 'var':
        flag = np.std(foldspec, axis=-1)

        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize//4), int(3*flagsize//4) )
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean+10*flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0)<0.5] = 0
        if apply_mask:
            I = I*mask[...,np.newaxis]

    # determine off_gates as lower 50% of profile
    profile = I.mean(0).mean(0)
    off_gates = np.argwhere(profile<np.median(profile)).squeeze()

    # renormalize, now that RFI are zapped
    bpass = I[...,off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = 0
    foldspec -= np.mean(foldspec[...,off_gates], axis=-1, keepdims=True)
        
    if plots:
        plot_diagnostics(foldspec, flag, mask)

    return foldspec, flag, mask

def plot_diagnostics(foldspec, flag, mask):
    # Need to add labels, documentation
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(231)
    plt.plot(foldspec.mean(0).mean(0))
        
    plt.subplot(232)
    plt.title('RFI flagging parameter (log10)')
    plt.xlabel('time (bins)')
    plt.ylabel('freq (bins)')
    plt.imshow(np.log10(flag).T, aspect='auto')

    plt.subplot(233)
    plt.title('Manual off-gate scaling')
    plt.imshow(mask.T, aspect='auto', cmap='Greys')
    plt.xlabel('time (bins)')
    
    plt.subplot(234)
    plt.imshow(foldspec.mean(0), aspect='auto')
    plt.xlabel('phase')
    plt.ylabel('freq')

    plt.subplot(235)
    plt.imshow(foldspec.mean(1), aspect='auto')
    plt.xlabel('phase')
    plt.ylabel('time')

    plt.subplot(236)
    plt.imshow(foldspec.mean(2).T, aspect='auto')
    plt.xlabel('time')
    plt.ylabel('freq')

def create_dynspec(foldspec, profsig=5., bint=1):
    """
    Create dynamic spectrum from folded data cube
    
    Uses average profile as a weight, sums over phase

    Parameters 
    ----------                                                                                                         
    foldspec: [time, frequency, phase] array
    profsig: S/N value, mask all profile below this
    bint: integer, bin dynspec by this value in time
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
    tbins = int(dynspec.shape[0] // bint)
    dynspec = dynspec[-bint*tbins:].reshape(tbins, bint, -1).mean(1)
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

    SS = slow_FT(dynspec.astype('float64'), freq)
    
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
        dynspec = dynspec*t_window[:,np.newaxis]
        f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
        dynspec = dynspec*f_window[np.newaxis,:]

    if pad:
        nt = dynspec.shape[0]
        nf = dynspec.shape[1]
        dspec = np.copy(dynspec)
        dynspec = np.zeros( (nt*2, nf) )
        dynspec[nt//2:nt+nt//2, :] += dspec
        m = np.copy(mask)
        mask = np.zeros( (nt*2, nf) )
        mask[nt//2:nt+nt//2, :] += m

    SS = slow_FT(dynspec.astype('float64'), freq)
    SS = np.fft.ifftshift(SS)
    SS_ccorr = np.fft.ifft2(abs(SS)**2.0)

    mask_intrinsic = mask * dynspec.mean(-1, keepdims=True)

    SSm = slow_FT(mask_intrinsic.astype('float64'), freq)
    SSm = np.fft.ifftshift(SSm)
    SSm_ccorr = np.fft.ifft2(abs(SSm)**2.0)
    SSm_ccorr /= np.max(SSm_ccorr)

    SS_corrected = np.fft.fft2(abs(SS_ccorr) / abs(SSm_ccorr))
    SS_corrected = np.fft.fftshift(SS_corrected)
    
    return abs(SS_corrected)


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
