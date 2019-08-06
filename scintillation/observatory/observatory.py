import numpy as np
import os
import astropy.units as u
from astropy.coordinates import EarthLocation
from collections import defaultdict

class Observatories:

    """
    Loads TEMPO2 observatory.dat file as a python dict
    Observatory locations as astropy EarthLocations

    """
    
    def __init__(self, TEMPO2=False, vlba=True, evn=True, lba=True, leap=True, lofar=True):

        if TEMPO2:
            T2 = os.environ.get('TEMPO2')
            obsfile = T2+'/observatory/observatories.dat'        
        else:
            obsfile = './observatories.dat'

        obsinfo = np.genfromtxt(obsfile, dtype=np.str)
        observatories = defaultdict(dict)
        
        for i in range(obsinfo.shape[0]):
            obsi = obsinfo[i,3]
            obsi_short = obsinfo[i,4]
            xi = float(obsinfo[i,0])*u.m
            yi = float(obsinfo[i,1])*u.m
            zi = float(obsinfo[i,2])*u.m
            obsipos = EarthLocation(xi,yi,zi)
            observatories[obsi] = obsipos

        self.allobs = observatories

        if leap:
            self.LEAP()
        if vlba:
            self.VLBA()
        if lofar:
            self.LOFAR()
        if lba:
            self.LBA()
        if evn:
            self.EVN()

    def subset(self, tels):
        sub = defaultdict(dict)
        for tel in tels:
            sub[tel] = self.allobs[tel]
        return sub
    
    def LEAP(self):
        LEAPtels = ['EFFELSBERG', 'JODRELLM4', 'WSRT', 'NANCAY', 'SRT']
        self.leap = self.subset(LEAPtels)

    def LOFAR(self):
        LOFARtels = ['LOFAR', 'DE601', 'DE602', 'DE603', 'DE604', 'DE605',
                     'FR606', 'SE607', 'UK608', 'DE609', 'FI609']
        self.lofar = self.subset(LOFARtels)
        
    def VLBA(self, VLA=None, HSA=None):
        VLBAtels = ['BREWSTER', 'FORTDAVIS', 'HANCOCK', 'KITTPEAK', 'LOSALAMOS',
                    'MAUNAKEA', 'NORTHLIBERTY', 'OVRO', 'PIETOWN', 'STCROIX']
        if HSA:
            VLBAtels.append('ARECIBO', 'VLA', 'GBT')
        self.vlba = self.subset(VLBAtels)

    def LBA(self):
        LBAtels = ['PARKES', 'Hobart', 'Tidbinbilla', 'Hartebeesthoek', 'ATCA',
                   'Mopra', 'Ceduna']
        self.lba = self.subset(LBAtels)

    def EVN(self):
        EVNtels = ['EFLSBERG', 'JODRELL1', 'WSTRBORK', 'ONSALA85', 'SARDINIA', 'MEDICINA',
                   'TORUN', 'HART', 'URUMQI', 'BADARY', 'SVETLOE', 'ZELENCHK', 'TIANMA65']
        self.evn = self.subset(EVNtels)

