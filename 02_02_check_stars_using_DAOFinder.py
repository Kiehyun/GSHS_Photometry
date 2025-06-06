# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

"""
import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.nddata import CCDData
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, sigma_clipped_stats

from photutils import DAOStarFinder
from photutils import IRAFStarFinder
from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn
from photutils import detect_threshold
from photutils.centroids import centroid_com
#from photutils import aperture_photometry as apphot

import warnings

from ccdproc import CCDData, ccd_process

from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _Python_utilities
import _astro_utilities

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
verbose = True # False
tryagain = False
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data")  

PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C2-Asteroid"
TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C3-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin"

# PROJECDIR = BASEDIR / "C4-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

PROJECDIR = BASEDIR / "C5-Test"
TODODIR = PROJECDIR / "-_-_-_-_GSON300_STF-8300M_-_1bin"

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))
if verbose == True :
    print ("DOINGDIRs: ", format(DOINGDIRs))
    print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

try : 
    BDFDIR = [x for x in DOINGDIRs if "CAL-BDF" in str(x)]
    if verbose == True :
        print ("BDFDIR: ", format(BDFDIR))
    BDFDIR = Path(BDFDIR[0])    
except : 
    BDFDIR = TODODIR
    pass

DOINGDIRs = sorted([x for x in DOINGDIRs if "_LIGHT_" in str(x)])
# print ("DOINGDIRs: ", format(DOINGDIRs))
# print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

# filter_str = '127JOHANNA_LIGHT_-_2023-11-17_-_GSON300_STF-8300M_-_1bin'
# DOINGDIRs = [x for x in DOINGDIRs if filter_str in str(x)]
# remove = 'BIAS'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'DARK'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'FLAT'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
if verbose == True :
    print ("DOINGDIRs: ", DOINGDIRs)
    print ("len(DOINGDIRs): ", len(DOINGDIRs))
#######################################################


#%%
#####################################################################
# Observed location
LOCATION = dict(lon=127.005, lat=37.308889, elevation=101)
GSHS = EarthLocation(lon=127.005 * u.deg,
                                 lat=37.308889 * u.deg,
                                 height=101 * u.m)
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
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True :
        print("DOINGDIR", DOINGDIR)
    DAOFINDERDIR = DOINGDIR / _astro_utilities.DAOfinder_result_dir
    if not DAOFINDERDIR.exists():
        os.makedirs("{}".format(str(DAOFINDERDIR)))
        if verbose == True :
            print("{} is created...".format(str(DAOFINDERDIR)))
    
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
            print("df_light :", df_light)

        for _, row  in df_light.iterrows():
            fpath = Path(row["file"])
            # fpath = Path(df_light["file"][1])
            if verbose == True :
                print("fpath :" ,fpath)
            hdul = fits.open(fpath)

            if hdul[0].header['CCDNAME'] == 'STF-8300M' :
                val_figsize = (12, 9)
                val_fraction = 0.035
                hdul[0].header["GAIN"] = 0.37,
                hdul[0].header["RDNOISE"] = 9.3

            if hdul[0].header['CCDNAME'] == 'STX-16803' :
                val_figsize=(10, 9)
                val_fraction = 0.0455
                hdul[0].header["GAIN"] = 1.27
                hdul[0].header["RDNOISE"] = 9.0    

            ## thres
            # thresh = detect_threshold(data=hdul[0].data, nsigma=3)
            # thresh = thresh[0][0]
            # print('detect_threshold', thresh)

            avg, med, std = sigma_clipped_stats(hdul[0].data)  # by default, 3-sigma 5-iteration.
            thresh = 5. * std
            # print('detect_threshold', thresh)

            DAOfind = DAOStarFinder(
                                    fwhm = FWHM, 
                                    threshold = thresh, 
                                    # sharplo = 0.2, sharphi = 1.0,  # default values: sharplo=0.2, sharphi=1.0,
                                    # roundlo = -1.0, roundhi = 1.0,  # default values -1 and +1
                                    # sigma_radius = 1.5,           # default values 1.5
                                    # ratio = 1.0,                  # 1.0: circular gaussian
                                    exclude_border = True         # To exclude sources near edges
                                    )
            # The DAOStarFinder object ("DAOfind") gets at least one input: the image.
            # Then it returns the astropy table which contains the aperture photometry results:
            DAOfound = DAOfind(hdul[0].data)

            if DAOfound is None :
                if verbose == True :
                    print("DAOfound : No star was found...")    
            else : 
                if verbose == True :
                    print("DAOfound :", DAOfound)
                    print("len(DAOfound) :",len(DAOfound))
                    print(DAOfound.colnames)

                df_DAO = DAOfound.to_pandas()
                if verbose == True :
                    print(type(df_DAO))
                df_DAO.to_csv(f"{DAOFINDERDIR}/{fpath.stem}_DAOfinder_fwhm_{FWHM}.csv")
                if verbose == True :
                    print("df_DAO.describe :", df_DAO.describe)

                #########
                pos = np.transpose((DAOfound['xcentroid'], DAOfound['ycentroid']))
                apert = CAp(pos, r=R_AP)
                annul = CAn(positions=pos, r_in= R_IN, r_out=R_OUT)

                fig, axs = plt.subplots(1, 1, figsize=val_figsize,
                                    # subplot_kw={'projection': wcs},
                                    sharex=False, sharey=False, gridspec_kw=None)

                im = _astro_utilities.zimshow(axs, hdul[0].data, )

                axs.tick_params(labelsize=8)

                annul.plot(axs, color="r")
                for i in range(len(pos)):
                    axs.text(pos[i][0], pos[i][1], f"Star #{str(i)}", fontsize=6, color='w')

                annul.plot(axs, color="r")

                cbar = plt.colorbar(im, ax = axs, fraction=val_fraction, pad=0.04, )
                cbar.ax.tick_params(labelsize=8)

                axs.set_title(f"fname: {fpath.name}\n Result of DAOFinder", fontsize=10,)

                axs.annotate(f'FWHM: {FWHM}', fontsize=8,
                    xy=(0, 0), xytext=(0, -30), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')

                axs.annotate(f'Sky threshold: {thresh:.02f}', fontsize=8,
                    xy=(0, 0), xytext=(0, -40), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')

                axs.annotate(f'Number of star(s): {len(DAOfound)}', fontsize=8,
                    xy=(0, 0), xytext=(0, -50), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')
                
                axs.annotate(f'sharpness: {df_DAO['sharpness'].abs().mean():.03f}', fontsize=8,
                    xy=(0, 0), xytext=(200, -30), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')

                axs.annotate(f'roundness1: {df_DAO['roundness1'].abs().mean():.03f}', fontsize=8,
                    xy=(0, 0), xytext=(200, -40), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')

                axs.annotate(f'roundness2: {df_DAO['roundness2'].abs().mean():.03f}', fontsize=8,
                    xy=(0, 0), xytext=(200, -50), va='top', ha='left',
                    xycoords='axes fraction', textcoords='offset points')

                plt.tight_layout()
                plt.savefig(f"{DAOFINDERDIR}/{fpath.stem}_DAOfinder_fwhm_{FWHM}.png")

                # plt.show()
                plt.close()