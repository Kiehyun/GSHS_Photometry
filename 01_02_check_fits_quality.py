# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

"""
#%%
from glob import glob
from pathlib import Path
import shutil
import os
import platform
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
# from photutils import DAOStarFinder

# ...existing code...
# try:
#     from photutils.psf.models import CircularGaussianPRF
# except ImportError:
#     from photutils.psf import CircularGaussianPRF  # 구버전 photutils 호환
# from photutils.psf import PSFPhotometry
# ...existing code...

from astropy.nddata import Cutout2D
import astropy.units as u

from astropy.modeling.fitting import LevMarLSQFitter

import ysfitsutilpy as yfu

import _astro_utilities
import _Python_utilities

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
count_stars = False
verbose = True
tryagain = True
trynightsky = False
file_retry_dt = datetime(2025, 3, 5, 16, 17)
# file_retry_dt = datetime.now()
downsample = 4
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data") 
if platform.system() == "Windows":
    BASEDIR = Path("R:\\")  

# PROJECDIR = BASEDIR / "A3_CCD_obs_raw"

# PROJECDIRs = [ 
#                 # "STX-16803_1bin", 
#                 # "STX-16803_2bin",  
#                 # "STL-11000M_2bin", 
#                 # "STF-8300M_2bin",  
#                 # "QSI683ws_2bin", 
#                 # "STL-11000M_1bin",
#                 "STF-8300M_1bin",
#                 # "QSI683ws_1bin",
#                 # "ASI2600MC_1bin",
#                 "ASI6200MMPro_3bin",
#                 ]
# DOINGDIRs = []
# for DOINGDIR in PROJECDIRs:
#     TODODIR = PROJECDIR / DOINGDIR              
#     DOINGDIRs.extend(sorted(_Python_utilities.getFullnameListOfallsubDirs(str(TODODIR))))
# if verbose == True :
#     print ("DOINGDIRs: ", format(DOINGDIRs))
#     print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))


PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"  #=2
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"  #=1
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin" #=3 
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin" #=4
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin" #-1
TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"  #-2
TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin" #-3

# PROJECDIR = BASEDIR / "C2-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C3-EXO"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin" #-1
TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin" #-=1
TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"    #-=2
TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"    #-=3
TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"    #-=4
TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin" #==1
TODODIR = PROJECDIR / "-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin"  #==2
TODODIR = PROJECDIR / "-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin" #==3
TODODIR = PROJECDIR / "-_-_-_2025-01_-_GSON300_STF-8300M_-_1bin"  #==4
TODODIR = PROJECDIR / "-_-_-_2025-01_-_RiLA600_ASI6200MMPro_-_3bin"
TODODIR = PROJECDIR / "-_-_-_2025-02_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2025-02_-_RiLA600_ASI6200MMPro_-_3bin"
TODODIR = PROJECDIR / "-_-_-_2025-03_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2025-03_-_RiLA600_ASI6200MMPro_-_3bin"

# PROJECDIR = BASEDIR / "C4-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

# PROJECDIR = BASEDIR / "C5-CMD"
# TODODIR = PROJECDIR / "-_-_-_2025-02_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2025-02_-_RiLA600_ASI6200MMPro_-_3bin"
# TODODIR = PROJECDIR / "-_-_-_2025-03_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2025-03_-_RiLA600_ASI6200MMPro_-_3bin"

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))


filter_strs = [
                '2025-0',
                # '2025-02',
                # '2025-03',
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                ]  # Example list of filter strings
DOINGDIRs = [x for x in DOINGDIRs if any(filter_str in str(x) for filter_str in filter_strs)]

rm_filter_strs = [
                # 'GPX-1b',
                # 'HAT',
                'CAL-BDF',
                'DARK',
                'FLAT',
                'BIAS',
                'Bad_fits', 
                # 'Another_bad_string',
                  ]  # Example list of filter strings to remove
DOINGDIRs = [x for x in DOINGDIRs if not any(rm_filter_str in str(x) for rm_filter_str in rm_filter_strs)]

if verbose == True :
    print ("DOINGDIRs: ", DOINGDIRs)
    print ("len(DOINGDIRs): ", len(DOINGDIRs))
#######################################################
#%%
#####################################################################
# Observed location
LOCATION = dict(lon=127.005, lat=37.308889, elevation=101)
# GSHS = EarthLocation(lon=127.005 * u.deg,
#                                  lat=37.308889 * u.deg,
#                                  height=101 * u.m)
MPC_obscode = "P64"
#######################################################
# Used for any `astropy.SkyCoord` object:
SKYC_KW = dict(unit=u.deg, frame='icrs')

# Initial guess of FWHM in pixel
FWHM_INIT = 4
FWHM = FWHM_INIT

# Photometry parameters
R_AP = 1.5 * FWHM_INIT # Aperture radius
R_IN = 4 * FWHM_INIT   # Inner radius of annulus
R_OUT = 6 * FWHM_INIT  # Outer radius of annulus
#######################################################
#%%
def check_Good_FITS(fits_path,  
                          fwhm_thresh=10, 
                          ellipticity_thresh=0.4, 
                          min_stars=5, 
                          verbose=True,
                          plot=True,
                          **kwargs):
    """
    가이드 불량 여부를 판단하는 함수.
    """
    msg = f"check Good FITS {fits_path.name}\n" 
    msg += f"fwhm_thresh : {fwhm_thresh}, " 
    msg += f"ellipticity_thresh : {ellipticity_thresh}, "
    msg += f"min_stars : {min_stars}, \n"

    with fits.open(fits_path) as hdul:
        data = hdul[0].data

        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm_thresh, 
                                threshold=5.0*std)
        sources = daofind(data - median)

        if sources is None :
            msg += f"No star found. \n"
            if verbose:
                print(f"{msg}")
            if plot == True :
                fig, axs = plt.subplots(1,1,
                            sharex=False, sharey=False, gridspec_kw=None)
                im = _astro_utilities.zimshow(axs, hdul[0].data, )
                axs.annotate(f'msg', fontsize=8,
                            xy=(0, 0), xytext=(10, -20), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')
                plt.show()
                plt.close()
            return False, msg
        elif len(sources) < min_stars:
            msg += f"Not enough stars({len(sources)}) to judge. \n"
            if verbose:
                print(f"{msg}")
            if plot == True :   
                fig, axs = plt.subplots(1, 1, 
                            sharex=False, sharey=False, gridspec_kw=None)
                im = _astro_utilities.zimshow(axs, hdul[0].data, )
                axs.annotate(f'msg', fontsize=8,
                            xy=(0, 0), xytext=(10, -20), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')
                plt.show()
                plt.close()
            return False, msg
        else:
            msg += f"Star count: {len(sources)}. \n"
            if verbose:
                print(f"{msg}")
            
            # # Set up PSF model and fitter
            # psf_model = CircularGaussianPRF(x_0=0, y_0=0)
            # fitter = LevMarLSQFitter()
            # phot = PSFPhotometry(psf_model, fitter=fitter)

            # # Fit each detected source
            # for source in sources:
            #     cutout = Cutout2D(hdul[0].data, (source['xcentroid'], source['ycentroid']), size=20)
            #     result = phot(cutout)
            #     fwhm = result['fwhm']  # FWHM in pixels

            if plot == True :   
                fig, axs = plt.subplots(1, 1, 
                            sharex=False, sharey=False, gridspec_kw=None)
                im = _astro_utilities.zimshow(axs, hdul[0].data, )
                axs.annotate(f'msg', fontsize=8,
                            xy=(0, 0), xytext=(10, -20), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')
                plt.show()
                plt.close()
            return True, msg
        # FWHM 계산


    

        

#%%
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True :
        print("DOINGDIR", DOINGDIR)

    # if _Python_utilities.check_string_in_file(log_file, DOINGDIR.name):
    #     print(f"'{DOINGDIR.name}' found in '{log_file}'")
    #     pass
    # else:
    print(f"'{DOINGDIR.name}' not found in '{log_file}' or file does not exist.")

    BADFITSDIR = DOINGDIR / _astro_utilities.Bad_fits_dir
    if not BADFITSDIR.exists():
        os.makedirs("{}".format(str(BADFITSDIR)))
        if verbose == True :
            print("{} is created...".format(str(BADFITSDIR)))

    summary = yfu.make_summary(DOINGDIR/"*.fit*",
                                verify_fix=True,
                                ignore_missing_simple=True,
                                )
    if summary is not None :
        if verbose == True :
            print("len(summary):", len(summary))
            print("summary:", summary)
            #print(summary["file"][0])  
        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)
        if verbose == True :
            print("df_light:\n{}".format(df_light))
    if verbose == True :
        print("df_light :", df_light)

    for _, row  in df_light.iterrows():
        fpath = Path(row["file"])
        # fpath = Path(df_light["file"][1])
        if verbose == True :
            print("fpath :" ,fpath)
        try :
            result, msg = check_Good_FITS(fpath,
                                fwhm_thresh=5, 
                                ellipticity_thresh=0.99, 
                                min_stars=10, 
                                plot=False,
                                verbose=True)
            if verbose == True :
                print("result: ", result)
            if result == False :
                shutil.move(fpath, BADFITSDIR/fpath.name)
                with open(BADFITSDIR/f"{fpath.stem}.log", "w") as f:
                    f.write(msg + "\n")

                if verbose == True :
                    print("#"*60)
                    print("moved to: ", BADFITSDIR)
                    print("#"*80)

        except Exception as err :
            print("X"*60)
            _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
            pass

    _Python_utilities.write_log(log_file, 
                        f"{str(DOINGDIR)} is finighed..", 
                        verbose=verbose)
# %%
