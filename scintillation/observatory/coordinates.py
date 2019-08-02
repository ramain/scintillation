import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from itertools import combinations

def coordinates(t0, source, observatories, plots=True, dT=24.):
     """
     t0:  astropy Time, starting time of interest
     source: astropy SkyCoord of your source
     observatories:  Observatory dictionary of EarthLocations
     plots:  Whether you want the default rise/set plot
     dT: time in hours beyond t0

     returns points of alt, az, and time
     """
     
     midnight = t0
     delta_midnight = np.linspace(0, dT, 1000)*u.hour
     
     plt.figure(figsize=(10, 5))

     if isinstance(observatories, dict):
         altazs = np.zeros((len(observatories.items()), 2, 1000) )
           
         i = 0
         for name, observatory in observatories.items():
             altaz = source.transform_to(AltAz(obstime=midnight+delta_midnight, location=observatory))
             altazs[i,0,:] = altaz.alt
             altazs[i,1,:] = altaz.az
             i += 1
             if plots:
                 plt.plot(delta_midnight, altaz.alt, label=name)
     else:
         altazs = source.transform_to(AltAz(obstime=midnight+delta_midnight, location=observatories))
         if plots:
             plt.plot(delta_midnight, altazs.alt)
  

     if plots:
         plt.axhline(30, linestyle='--', color='r')
         plt.axhline(10, linestyle='--', color='b')
         plt.axhline(0, linestyle='--', color='k')
         
         plt.xlim(0,dT)
         plt.legend()
         plt.grid()
         
         plt.ylabel('Altitude (degrees)', fontsize=18)
         plt.xlabel('Hours from midnight', fontsize=18)
         plt.show()
         
     return altazs, t0+delta_midnight

 
def UV(T, source, tel1, tel2):
    X1 = np.array([tel1.x.value, tel1.y.value, tel1.z.value])
    X2 = np.array([tel2.x.value, tel2.y.value, tel2.z.value])
    # Baseline center, average of telescope and reference telescope    
    dX = X2 - X1
    X_bl = (X1 + X2) / 2.

    Baseline = EarthLocation(X_bl[0]*u.m, X_bl[1]*u.m, X_bl[2]*u.m)

    ot = Time(T, scale='utc', location=Baseline)
    obst = ot.sidereal_time('mean')
    
    h = obst - Baseline.lon - source.ra
    d = source.dec
    
    # matrix to transform xyz to uvw, doing one step at a time
    xyz_v = np.array([-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)])
    xyz_u = np.array([np.sin(h), np.cos(h), 0.])
    xyz_w = np.array([np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d)])
    
    uvw = np.zeros((len(T), 3))
    uvw[:,0] = (xyz_u * dX).sum()
    uvw[:,1] = (xyz_v * dX).sum()
    uvw[:,2] = (xyz_w * dX).sum()
    
    return uvw


def UV_all(t0, source, observatories, plots=True, dT=24.):
    """
    t0:  start time of your observation
    source:  astropy SkyCoord
    observatories:  dictionary of astropy EarthLocations
    plots:  boolean, whether to create default plots

    returns the UVW matrix of all baselines for duration of obs.
    """
    T = t0 + np.linspace(0, dT, 1000)*u.hour
    nb = len(observatories.items())
    uvw_mat = np.zeros((nb*(nb-1), len(T), 3))
    i = 0
    b = []

    if plots:
         plt.figure(figsize=(6,6))
    
    for tel1,tel2 in combinations(sorted(observatories.items()), 2):
        name1 = tel1[0]
        name2 = tel2[0]
        b.append(name1+" - "+name2)
        loc1 = tel1[1]
        loc2 = tel2[1]
        uvw_mat[i] = UV(T, source, loc1, loc2)

        if plots:
            plt.plot( -uvw_mat[i,:,0]/1000000., -uvw_mat[i,:,1]/1000000., label=b[i], color='k')
            plt.plot( uvw_mat[i,:,0]/1000000., uvw_mat[i,:,1]/1000000., color='k')
            
            plt.xlabel('u (Mm)')
            plt.ylabel('v (Mm)')

        i += 1

    if plots:
        vlim = np.max(uvw_mat.ravel())/1000000.
        plt.xlim(-1.1*vlim, 1.1*vlim)
        plt.ylim(-1.1*vlim, 1.1*vlim)
        plt.show()
        
    return uvw_mat, T

