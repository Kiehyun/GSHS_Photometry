# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

"""
#%%
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _astro_utilities
import _Python_utilities

plt.rcParams.update({'figure.max_open_warning': 0})

import warnings
warnings.filterwarnings('ignore')

#%%
#######################################################
# for log file
log_dir = "logs/"
log_file = "{}{}.log".format(log_dir, os.path.basename(__file__)[:-3])
err_log_file = "{}{}_err.log".format(log_dir, os.path.basename(__file__)[:-3])
print ("log_file: {}".format(log_file))
print ("err_log_file: {}".format(err_log_file))
if not os.path.exists('{0}'.format(log_dir)):
    os.makedirs('{0}'.format(log_dir))
#######################################################
#%%
#######################################################
from multiprocessing import Process, Queue
class Multiprocessor():
    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def restart(self):
        self.processes = []
        self.queue = Queue()

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets
########################################################%%
#%%
#######################################################
BASEDIR = Path("/mnt/Rdata/OBS_data")  

PROJECDIR = BASEDIR / "01-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "02-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# # TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "03-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"

# PROJECDIR = BASEDIR / "04-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

# PROJECDIR = BASEDIR / "2024-OA" / "_측광과제-교사참고"
# TODODIR = PROJECDIR / "1반" / "22024문준원" 
# TODODIR = PROJECDIR / "1반" / "22027박성민" 
# TODODIR = PROJECDIR / "1반" / "22028박성환" 
# TODODIR = PROJECDIR / "1반" / "22029박주영"
# TODODIR = PROJECDIR / "1반" / "22030박지원" 
# TODODIR = PROJECDIR / "1반" / "22032박지환" 
# TODODIR = PROJECDIR / "1반" / "22072이재우" 
# TODODIR = PROJECDIR / "1반" / "22080이혁준" 
# TODODIR = PROJECDIR / "1반" / "22098정이찬" 
# TODODIR = PROJECDIR / "1반" / "22106조형준" 
# TODODIR = PROJECDIR / "1반" / "22115최준서" 
# TODODIR = PROJECDIR / "1반" / "23054박하람" 
# TODODIR = PROJECDIR / "1반" / "23067신재헌" 
# TODODIR = PROJECDIR / "1반" / "23097임윤준" 
# TODODIR = PROJECDIR / "1반" / "23116최현준"

# TODODIR = PROJECDIR / "2반" / "23074오서준"
# TODODIR = PROJECDIR / "2반" / "22018김한준"
# TODODIR = PROJECDIR / "2반" / "22004권민우"
# TODODIR = PROJECDIR / "2반" / "22095정은재"
# TODODIR = PROJECDIR / "2반" / "22022노현우"
# TODODIR = PROJECDIR / "2반" / "22073이재욱"
# TODODIR = PROJECDIR / "2반" / "21100정영우"
# TODODIR = PROJECDIR / "2반" / "22045양현서"
# TODODIR = PROJECDIR / "2반" / "22050오태원"
# TODODIR = PROJECDIR / "2반" / "22125홍은찬"
# TODODIR = PROJECDIR / "2반" / "22093정우현"
# TODODIR = PROJECDIR / "2반" / "22110최석원"
# TODODIR = PROJECDIR / "2반" / "22035박홍준"
# TODODIR = PROJECDIR / "2반" / "22053용승주"
# TODODIR = PROJECDIR / "2반" / "22107지민기"
# TODODIR = PROJECDIR / "2반" / "22118최현진"

# TODODIR = PROJECDIR / "3반" / "23075오승민"
# TODODIR = PROJECDIR / "3반" / "23027김재우"
# TODODIR = PROJECDIR / "3반" / "23108조형석"
# TODODIR = PROJECDIR / "3반" / "23069안선우"
# TODODIR = PROJECDIR / "3반" / "22005권순민"
# TODODIR = PROJECDIR / "3반" / "22008김도현"
# TODODIR = PROJECDIR / "3반" / "22039손희원"
# TODODIR = PROJECDIR / "3반" / "22088장태훈"
# TODODIR = PROJECDIR / "3반" / "22012김수아"
# TODODIR = PROJECDIR / "3반" / "22034박현수"
# TODODIR = PROJECDIR / "3반" / "22082임비건"
# TODODIR = PROJECDIR / "3반" / "22048오은총"
# TODODIR = PROJECDIR / "3반" / "22069이은우"
# TODODIR = PROJECDIR / "3반" / "22103조연우"
# TODODIR = PROJECDIR / "3반" / "22121함석규"
# TODODIR = PROJECDIR / "3반" / "22108차무겸"


DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))
print ("DOINGDIRs: ", format(DOINGDIRs))
print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

try : 
    BDFDIR = [x for x in DOINGDIRs if "CAL-BDF" in str(x)]
    print ("BDFDIR: ", format(BDFDIR))
    BDFDIR = Path(BDFDIR[0])    
except : 
    BDFDIR = TODODIR
    pass

DOINGDIRs = sorted([x for x in DOINGDIRs if "_LIGHT_" in str(x)])
# print ("DOINGDIRs: ", format(DOINGDIRs))
# print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

# filter_str = 'IW'
# DOINGDIRs = [x for x in DOINGDIRs if filter_str in str(x)]
# remove = 'BIAS'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'DARK'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'FLAT'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
print ("DOINGDIRs: ", DOINGDIRs)
print ("len(DOINGDIRs): ", len(DOINGDIRs))
#######################################################
#%%
#####################################################################
# Observed location
LOCATION = dict(lon=127.005, lat=37.308889, elevation=101)
Suwon = location = EarthLocation(lon=127.005 * u.deg, 
                                 lat=37.308889 * u.deg, 
                                 height=101 * u.m)
observatory_code = "P64"

# Used for any `astropy.SkyCoord` object:
SKYC_KW = dict(unit=u.deg, frame='icrs')

#######################################################
# Initial guess of FWHM in pixel
FWHM_INIT = 4

# Photometry parameters
# R_AP = 1.5*FWHM_INIT # Aperture radius
# R_IN = 4*FWHM_INIT   # Inner radius of annulus
# R_OUT = 6*FWHM_INIT  # Outer radius of annulus

Mag_Low = 11.5
Mag_High = 15

Mag_target = 12.5
Mag_delta = 2
ERR_Min = 0.5

### TT-ARI
# Mag_Low = 10
# Mag_High = 15

# Mag_target = 11
# Mag_delta = 2
# ERR_Min = 0.5

coord_deltas = np.arange(0.00001, 0.00050, 0.00001)
#######################################################
#%%

#%%
tryagain = False
trynightsky = False
tryASTROMETRYNET=True
downsample = 6

#### combine_BDF
# _astro_utilities.combine_BDF(BDFDIR,
#                 tryagain = tryagain,
#                 )

for DOINGDIR in DOINGDIRs[:] :
    _astro_utilities.solving_fits_file(DOINGDIR,
                downsample = downsample,
                tryagain = tryagain,
                tryASTROMETRYNET = tryASTROMETRYNET,     
                )    
    
    _astro_utilities.ccd_Reducuction (DOINGDIR,
                                      BDFDIR,
                                    tryagain = tryagain,
                                    trynightsky = trynightsky,
                                    )

    _astro_utilities.solving_fits_file(DOINGDIR,
                SOLVINGDIR = _astro_utilities.reduced_dir,
                downsample = downsample,
                tryagain = tryagain,
                tryASTROMETRYNET = tryASTROMETRYNET,     
                )     
    
    _astro_utilities.diff_Photometry_PS1 (DOINGDIR,
                        tryagain = tryagain, 
                        LOCATION = LOCATION,
                        SKYC_KW = SKYC_KW,
                        FWHM_INIT = FWHM_INIT,
                        Mag_Low = Mag_Low,
                        Mag_High = Mag_High,
                        Mag_target = Mag_target,
                        Mag_delta = Mag_delta,
                        ERR_Min = ERR_Min,
                        READINGDIR =  _astro_utilities.reduced_dir,
                        # READINGDIR =  _astro_utilities.reduced_nightsky_dir,
                    ) 

    _astro_utilities.plot_light_curve_variables_using_csv(DOINGDIR,
                                         Mag_target = Mag_target,
                                         FWHM_INIT = FWHM_INIT,
                                         READINGDIR = _astro_utilities.reduced_dir,
                                        #  READINGDIR = _astro_utilities.reduced_nightsky_dir,
                                        )
    
    _astro_utilities.solving_fits_file(DOINGDIR,
                SOLVINGDIR = _astro_utilities.reduced_nightsky_dir,
                downsample = downsample,
                tryagain = tryagain,
                tryASTROMETRYNET = tryASTROMETRYNET,     
                ) 
        
    _astro_utilities.diff_Photometry_PS1 (DOINGDIR,
                        tryagain = tryagain, 
                        LOCATION = LOCATION,
                        SKYC_KW = SKYC_KW,
                        FWHM_INIT = FWHM_INIT,
                        Mag_Low = Mag_Low,
                        Mag_High = Mag_High,
                        Mag_target = Mag_target,
                        Mag_delta = Mag_delta,
                        ERR_Min = ERR_Min,
                        # READINGDIR =  _astro_utilities.reduced_dir,
                        READINGDIR =  _astro_utilities.reduced_nightsky_dir,
                    )   
    
    _astro_utilities.plot_light_curve_variables_using_csv(DOINGDIR,
                                         Mag_target = Mag_target,
                                         FWHM_INIT=FWHM_INIT,
                                        #  READINGDIR = _astro_utilities.reduced_dir,
                                         READINGDIR = _astro_utilities.reduced_nightsky_dir,
                                        )