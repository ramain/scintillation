import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt

from .slowft import slow_FT

def readpsrfits(fname, undo_scaling=True, dedisperse=False, verbose=True):
   
    """Read folded spectrum from psrfits file, based on code from Dana Simard
    Parameters
    ----------
    fname : string
        Fits file to be opened
    undo_scaling : True or False
        Remove the automatic normalization applied by psrfits.  This renders
        output identical to psrchive within machine precision
        Notes from:
        www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html
    dedisperse : True or False
        Roll in phase by closest incoherent time-shift, using the in-header
        values of DM, tbin.  Phase 0 point is currently arbitrary

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
    npol = shape[0]
    ngate = shape[2]

    # Exact start time is encompassed in three header values
    t0 = Time(f['PRIMARY'].header['STT_IMJD'], format='mjd', precision=9)
    t0 += (f['PRIMARY'].header['STT_SMJD'] + f['PRIMARY'].header['STT_OFFS']) *u.s
    # Array of subint lengths, 
    Tint = f['SUBINT'].data['TSUBINT']*u.s
    T = t0 + np.cumsum(Tint)
    
    # Read and polulate data with each subint, one at a time
    data = np.zeros((nt,npol,nchan,ngate), dtype=np.float32)
    for i in np.arange(nt):
        data[i,:,:,:] = f['SUBINT'].data[i][-1]
    print(data.shape)
    if undo_scaling:
        # Data scale factor (outval=dataval*scl + offs) 
        scl = f['SUBINT'].data['DAT_SCL'].reshape(nt, npol, nchan)
        offs = f['SUBINT'].data['DAT_OFFS'].reshape(nt, npol, nchan)
        data = data * scl[...,np.newaxis] + offs[..., np.newaxis]

    # Remove best integer bin incoherent time shift
    fref = f['HISTORY'].data['CTR_FREQ'][0]
    if dedisperse:
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


def readpsrarch(fname, dedisperse=True, verbose=True):
    """
    Read pulsar archive directly using psrchive
    Requires the python psrchive bindings, only working in python2

    Parameters
    ----------
    fname: string
    file directory
    dedisperse: Bool
    apply psrchive's by-channel incoherent de-dispersion

    Returns archive data cube, frequency array, time(mjd) array, source name
    """
    import psrchive
    
    arch = psrchive.Archive_load(fname)
    source = arch.get_source()
    tel = arch.get_telescope()
    if verbose:
        print("Read archive of {0} from {1}".format(source, fname))

    if dedisperse:
        if verbose:
            print("Dedispersing...")
        arch.dedisperse()
    data = arch.get_data()
    midf = arch.get_centre_frequency()
    bw = arch.get_bandwidth()
    F = np.linspace(midf-bw/2., midf+bw/2., data.shape[2], endpoint=False)
    #F = arch.get_frequencies()

    a = arch.start_time()
    t0 = a.strtempo()
    t0 = Time(float(t0), format='mjd', precision=0)

    # Get frequency and time info for plot axes
    nt = data.shape[0]
    Tobs = arch.integration_length()
    dt = (Tobs / nt)*u.s
    T = t0 + np.arange(nt)*dt
    T = T.mjd
    
    return data, F, T, source, tel


def clean_foldspec(I, plots=True, apply_mask=False, rfimethod='var', flagval=10, offpulse='True', tolerance=0.5):
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

    Returns
    ------- 
    foldspec: folded spectrum, after bandpass division, 
    off-gate subtraction and RFI masking
    flag: std. devs of each subint used for RFI flagging
    mask: boolean RFI mask
    bg: Ibg(t, f) subtracted from foldspec
    bpass: Ibg(f), an estimate of the bandpass
    
    """

    # Sum to form total intensity, mostly a memory/speed concern
    if len(I.shape) == 4:
        print(I.shape)
        I = I[:,(0,1)].mean(1)

    # Use median over time to not be dominated by outliers
    bpass = np.median( I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass
    
    mask = np.ones_like(I.mean(-1))

    prof_dirty = (I - I.mean(-1, keepdims=True)).mean(0).mean(0)
    off_gates = np.argwhere(prof_dirty<np.median(prof_dirty)).squeeze()
    
    if rfimethod == 'var':
        if offpulse:
            flag = np.std(foldspec[..., off_gates], axis=-1)
        else:
            flag = np.std(foldspec, axis=-1)
        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize//4), int(3*flagsize//4) )
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean+flagval*flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0)<tolerance] = 0
        mask[mask.mean(1)<tolerance] = 0
        if apply_mask:
            I[mask < 0.5] = np.mean(I[mask > 0.5])

    # determine off_gates as lower 50% of profile
    profile = I.mean(0).mean(0)
    off_gates = np.argwhere(profile<np.median(profile)).squeeze()

    # renormalize, now that RFI are zapped
    bpass = I[...,off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = np.nanmean(foldspec)
    bg = np.mean(foldspec[...,off_gates], axis=-1, keepdims=True)
    foldspec = foldspec - bg
        
    if plots:
        plot_diagnostics(foldspec, flag, mask)

    return foldspec, flag, mask, bg.squeeze(), bpass.squeeze()


def rfifilter_median(dynspec, xs=20, ys=4, sigma=3., fsigma=5., tsigma=0., iters=3):
    """
    Flag hot pixels, as well as anomalous t,f bins in 
    a dynamic spectrum using a median filter
    
    Parameters
    ----------
    dynspec: ndarray [time, freq]
    xs: Filter size in time
    ys: Filter size in freq
    sigma: threshold for bad pixels in residuals
    fsigma: threshold for bad channels
    tsigma: threshold for bad time bins
    iters: int, number of iterations for median filter

    Returns
    ------- 
    ds_med: median filter of dynspec
    mask_filter: boolean mask of RFI
    
    """

    from scipy.ndimage import median_filter

    ds_med = median_filter(dynspec, size=[xs,ys])
    gfilter = dynspec-ds_med
    mask_filter = np.ones_like(gfilter)

    for i in range(iters):
        if i == 0:
            gfilter_masked = np.copy(gfilter)
        sigmaclip = np.nanstd(gfilter_masked)*sigma
        mask_filter[abs(gfilter)>sigmaclip] = 0
        gfilter_masked[abs(gfilter)>sigmaclip] = np.nan
        
    frac = 100.*(mask_filter.size - 1.*np.sum(mask_filter)) / mask_filter.size
    print('{0}/{1} = {2}% subints flagged '.format(
          int(mask_filter.size - 1.*np.sum(mask_filter)), mask_filter.size, frac))
        
    # Filter channels
    if fsigma > 0:
        nchan = gfilter.shape[1]
        gfilter_freq = np.nanstd(gfilter_masked, axis=0)
        gfilter_freq = gfilter_freq / np.nanmedian(gfilter_freq)
        chanthresh = fsigma*np.nanstd( np.sort(np.ravel(gfilter_freq))[nchan//8:7*nchan//8] )
        badchan = np.argwhere( abs(gfilter_freq-1) > chanthresh).squeeze()
        mask_filter[:,badchan] = 0
        gfilter_masked[:,badchan] = np.nan
        print('{0}/{1} channels flagged'.format(len(badchan), nchan))

    # filter bad time bins
    if tsigma > 0:
        ntime = gfilter.shape[0]
        gfilter_time = np.nanstd(gfilter_masked, axis=1)
        gfilter_time = gfilter_time / np.nanmedian(gfilter_time)
        timethresh = tsigma*np.nanstd( np.sort(np.ravel(gfilter_time))[ntime//8:7*ntime//8] )
        badtbins = np.argwhere( abs(gfilter_time-1) > timethresh).squeeze()
        mask_filter[badtbins] = 0
        gfilter_masked[badtbins] = np.nan
        print('{0}/{1} time bins flagged'.format(len(badtbins), ntime))

    return ds_med, mask_filter


def plot_diagnostics(foldspec, flag, mask):
    """
    Plot the outputs of clean_foldspec, and different axis summations of foldspec

    Parameters
    ----------
    foldspec: ndarray [time, freq, phase]
    flag: ndarray [time, freq], std. dev of each subint
    mask: ndarray [time, freq], boolean mask created from flag thresholds

    """
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(231)
    plt.plot(foldspec.mean(0).mean(0), color='k')
    plt.xlabel('phase (bins)')
    plt.ylabel('I (arb.)')
    plt.title('Pulse Profile')
    plt.xlim(0, foldspec.shape[-1])
    
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
    

def create_dynspec(foldspec, template=[1], profsig=5., bint=1, binf=1):
    """
    Create dynamic spectrum from folded data cube
    
    Uses average profile as a weight, sums over phase

    Returns: dynspec, np array [t, f]

    Parameters 
    ----------                                                                                                         
    foldspec: [time, frequency, phase] array
    template: pulse profile I(phase), phase weights to create dynamic spectrum
    profsig: S/N value, mask all profile below this (only if no template given)
    bint: integer, bin dynspec by this value in time
    binf: integer, bin dynspec by this value in frequency
    """
    
    # If no template provided, create profile by summing over time, frequency
    if len(template) <= 1:
        template = foldspec.mean(0).mean(0)
        template /= np.max(template)

        # Noise estimated from bottom 50% of profile
        tnoise = np.std(template[template<np.median(template)])
        # Template zeroed below threshold
        template[template < tnoise*profsig] = 0

    profplot2 = np.concatenate((template, template), axis=0)

    # Multiply the profile by the template, sum over phase
    dynspec = (foldspec*template[np.newaxis,np.newaxis,:]).mean(-1)

    if bint > 1:
        tbins = int(dynspec.shape[0] // bint)
        dynspec = dynspec[-bint*tbins:].reshape(tbins, bint, -1).mean(1)
    if binf > 1:
        dynspec = dynspec.reshape(dynspec.shape[0], -1, binf).mean(-1)

    return dynspec


def create_secspec(dynspec, freq, pad=1, taper=1):
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

    S = slow_FT(dynspec.astype('float64'), freq)
    
    return S


def create_secspecwindow(dynspec, freq, mask, pad=1, taper=1):
    """
    NOTE: This is a sub-optimal deconvolution of window function
    Leaving here for legacy reasons.

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


def write_psrflux(dynspec, dynspec_errs, F, t, fname, psrname=None, telname=None, note=None):
    """
    Write dynamic spectrum along with column info in 
    psrflux format, compatible with scintools
    
    dynspec: ndarray [time, frequency]
    dynspec_errs: ndarray [time, frequency]
    F: astropy unit, channel frequency
    t: astropy Time values for each subintegration
    fname: filename to write psrflux dynspec to
    psrname: optional, string with source name
    telname: optional, string with telescope
    note: optional, note with additional information
    """
    T_minute = (t.unix - t[0].unix)/60.
    dt = (T_minute[1] - T_minute[0])/2.
    T_minute = T_minute + dt
    F_MHz = F.to(u.MHz).value
    with open(fname, 'w') as fn:
        fn.write("# Dynamic spectrum in psrflux format\n")
        fn.write("# Created using scintillation.dynspectools\n")
        fn.write("# MJD0: {0}\n".format(t[0].mjd))
        if telname:
            fn.write("# telescope: {0}\n".format(telname))
        if psrname:
            fn.write("# source: {0}\n".format(psrname))
        if note:
            fn.write("# {0}\n".format(note))
        fn.write("# Data columns:\n")
        fn.write("# isub ichan time(min) freq(MHz) flux flux_err\n")

        for i in range(len(T_minute)):
            ti = T_minute[i]
            for j in range(len(F)):
                fi = F_MHz[j]
                di = dynspec[i, j]
                di_err = dynspec_errs[i, j]
                fn.write("{0} {1} {2} {3} {4} {5}\n".format(i, j, 
                                            ti, fi, di, di_err) )
    print("Written dynspec to {0}".format(fname))


def read_psrflux(fname):
    """
    Load dynamic spectrum from psrflux file
    
    Skeleton from scintools
    
    Returns: 
    dynspec, dynspec_err, T, F, source
    """

    with open(fname, "r") as file:
        for line in file:
            if line.startswith("#"):
                headline = str.strip(line[1:])
                if str.split(headline)[0] == 'MJD0:':
                    # MJD of start of obs
                    mjd = float(str.split(headline)[1])
                if str.split(headline)[0] == 'source:':
                    # MJD of start of obs
                    source = str.split(headline)[1]
                if str.split(headline)[0] == 'telescope:':
                    # MJD of start of obs
                    telescope = str.split(headline)[1]
       
    try:
        source
    except NameError:
        source = ''
       
    data = np.loadtxt(fname)
    dt = int(np.max(data[:,0])+1)
    df = int(np.max(data[:,1])+1)
    
    t = data[::df,2]*u.min
    F = data[:df,3]*u.MHz
    dynspec = (data[:,4]).reshape(dt,df)
    dynspec_err = (data[:,5]).reshape(dt,df)
    T = Time(float(mjd), format='mjd') + t

    return dynspec, dynspec_err, T, F, source


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
