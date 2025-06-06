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
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from ccdproc import combine, ccd_process, CCDData

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _astro_utilities
import _Python_utilities

from astropy.nddata import Cutout2D
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils.centroids import centroid_com

from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn
from photutils.aperture import aperture_photometry as apphot

from astroquery.simbad import Simbad
from urllib.parse import urlencode

from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
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
count_stars = False
verbose = True
tryagain = True
trynightsky = False
tryASTROMETRYNET = True
file_age = 0.01
file_retry_dt = datetime(2025, 3, 3, 11)
downsample = 4
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data")  
if platform.system() == "Windows":
    BASEDIR = Path("R:\\ASTRO_data") 

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
# TODODIR = PROJECDIR / "-_-_-_2025-03_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2025-03_-_RiLA600_ASI6200MMPro_-_3bin"

# PROJECDIR = BASEDIR / "C4-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

# PROJECDIR = BASEDIR / "C5-Test"
# TODODIR = PROJECDIR / "-_-_-_-_GSON300_STF-8300M_-_1bin"

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))
if verbose == True :
    print ("DOINGDIRs: ", format(DOINGDIRs))
    print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

try : 
    BDFDIR = [x for x in DOINGDIRs if "CAL-BDF" in str(x)]
    if verbose == True :
        print ("BDFDIR: ", format(BDFDIR))
    MASTERDIR = Path(BDFDIR[0]) / _astro_utilities.master_dir
    if not MASTERDIR.exists():
        os.makedirs("{}".format(str(MASTERDIR)))
        if verbose == True :
            print("{} is created...".format(str(MASTERDIR)))
    if verbose == True :
        print ("MASTERDIR: ", format(MASTERDIR))
except Exception as err :

    print("X"*60)
    _Python_utilities.write_log(err_log_file, f'{str(err)}', verbose=verbose)
    pass

filter_strs = [
                '2025-02-2',
                '2025-03',
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                ]  # Example list of filter strings
DOINGDIRs = [x for x in DOINGDIRs if any(filter_str in str(x) for filter_str in filter_strs)]

rm_filter_strs = [
                'CAL-BDF',
                # 'HAT',
                # 'WASP',
                'Bad_fits', 
                #   'Another_bad_string',
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
GSHS =  EarthLocation(lon=127.005 * u.deg, 
                                 lat=37.308889 * u.deg, 
                                 height=101 * u.m)
observatory_code = "P64"

# Used for any `astropy.SkyCoord` object:
SKYC_KW = dict(unit=u.deg, frame='icrs')

#######################################################
# Initial guess of FWHM in pixel
FWHM_INIT = 4

# Photometry parameters
R_AP = 1.5*FWHM_INIT # Aperture radius
R_IN = 4*FWHM_INIT   # Inner radius of annulus
R_OUT = 6*FWHM_INIT  # Outer radius of annulus

Mag_target = 12.5
Mag_delta_INIT = 2
ERR_Minimum = 0.5

coord_delta = 0.0001
# Mag_target = 11
# Mag_delta = 2
# ERR_Minimum = 0.5
#######################################################

#%%
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True :
        print("DOINGDIR", DOINGDIR)

    foldername_el = DOINGDIR.parts[-1].split("_")
    targ_name = foldername_el[0]
    targ_name = targ_name.replace("-"," ")
    # targ_name = ''.join([i for i in targ_name  if not i.isdigit()])

    obsdate = foldername_el[3]
    bin = foldername_el[-1][0]
    
    if verbose == True :
        print("targ_name :", targ_name)
        print("obsdate :", obsdate)
        print("bin :", bin)

    if DOINGDIR.parts[-3] == "C1-Variable" :
        targ_name = targ_name.replace("-"," ")
        targ_info = _astro_utilities.get_variable_star_info(targ_name, verbose=verbose)
        if verbose == True :
            print(f"targ_info : {targ_info}")
            print(f"type(targ_info) : {type(targ_info)}")

        if targ_info:
            print(f"Variable Star Info for {targ_name}:")
            for key, value in targ_info.items():
                print(f"{key}: {value}")
            star_name = targ_info['Star Name']
            star_info = _astro_utilities.get_star_info(star_name, 
                                                        ra = targ_info['RA'], 
                                                        dec = targ_info['DEC'], 
                                                        verbose=verbose)
            if star_info is None :
                star_name = star_name.replace(" ","-")
                star_info = _astro_utilities.get_star_info(star_name, 
                                                           ra = targ_info['RA'],
                                                           dec = targ_info['DEC'],
                                                           verbose=verbose)
            if star_info is None :
                star_info = targ_info
            if verbose == True :
                print("star_name :", star_name)
                print("star_info :", star_info) 
                print("type(star_info) :", type(star_info))
        else:
            print(f"No information found for {targ_name}.")

    if DOINGDIR.parts[-3] == "C3-EXO" :
        targ_name = targ_name.replace("-"," ")
        nasa_exoplanet_archive = _astro_utilities.NASAExoplanetArchive()
        targ_info = nasa_exoplanet_archive.get_exoplanet_orbital_info(targ_name)
        if verbose == True :
            print(f"targ_name : {targ_name}")
            print(f"targ_info : {targ_info}")
            print(f"type(targ_info) : {type(targ_info)}")

        if targ_info:
            if verbose == True :
                print(f"Orbital information for {targ_name}:")
            for key, value in targ_info.items():
                print(f"{key}: {value}")
            star_name = targ_info['hostname']
            
            if verbose == True :
                print("star_name :", star_name)
            star_info = _astro_utilities.get_star_info(star_name, 
                                                           ra = targ_info['ra'], 
                                                           dec = targ_info['dec'], 
                                                           verbose=verbose)
            if star_info is None :
                star_name = star_name.replace(" ","-")
                star_info = _astro_utilities.get_star_info(star_name, 
                                                           ra = targ_info['ra'], 
                                                           dec = targ_info['dec'], 
                                                           verbose=verbose)

            if star_info is None :
                star_info = targ_info
    
            if verbose == True :
                print("type(star_info) :", type(star_info))
                print("star_info :", star_info)
            # try :
            #     Mag_target = int(star_info['V Magnitude'])  # Convert the average magnitude to an integer
            # except :
            #     pass
            # if Mag_target <= 12 : 
            #     Mag_target = 12
            # elif Mag_target >= 15 :
            #     Mag_target = 15
        else:
            print(f"No orbital information found for {targ_name}.")
    df_targ = pd.DataFrame([star_info])
    if verbose == True:
        print(f"starname: {star_name}")
        print("Mag_target:", Mag_target)
        print("star_info:", star_info)
        print("type(df_targ):", type(df_targ))
    
    READINGDIR = DOINGDIR / _astro_utilities.reduced_dir
    # READINGDIR = DOINGDIR / _astro_utilities.reduced_nightsky_dir

    DIFFPRESULTDIR = DOINGDIR / f"{READINGDIR.parts[-1]}_DPhot_target_Mag{Mag_target}_fw{FWHM_INIT}"
    if not DIFFPRESULTDIR.exists():
        os.makedirs("{}".format(str(DIFFPRESULTDIR)))
        if verbose == True :
            print("{} is created...".format(str(DIFFPRESULTDIR)))

    summary = yfu.make_summary(READINGDIR/"*.fit*",
                                    verify_fix=True,
                                    ignore_missing_simple=True,
                                    verbose = verbose,)
    if summary is not None : 
        if verbose == True :
            print("len(summary):", len(summary))
            print("summary:", summary)

        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)
        if verbose == True :
            print("df_light:\n{}".format(df_light))

        for _, row  in df_light.iterrows():

            fpath = Path(row["file"])

            DOIT = False
            check_fpath = (DIFFPRESULTDIR/f"{fpath.stem}_result_photometry.csv")
            if not check_fpath.exists() :
                DOIT = True
                if verbose == True :
                    print("*"*10)
                    print("*"*10)
                    print(f"{check_fpath} is not exist...")                
            else :
                if tryagain == True and (_Python_utilities.is_file_created_before(check_fpath, file_retry_dt)):
                    DOIT = True
                    if verbose == True :
                        print("*"*10)
                        print("*"*10)
                        print("*"*10)
                        print(f"{check_fpath} is older than {file_retry_dt.strftime('%Y-%m-%d %H:%M:%S')}...")
                else :
                    if verbose == True :
                        print("*"*10)
                        print(f"{check_fpath} is younger than {file_retry_dt.strftime('%Y-%m-%d %H:%M:%S')}...")
                    pass
            if DOIT :   
            
                if verbose == True :
                    print("*"*20)
                    print(f"Starting {fpath.name}...")
                _astro_utilities.GRDFitsUpdater(fpath, verbose=verbose)
                hdul = fits.open(fpath)
                ccd = yfu.load_ccd(fpath)
                flt = hdul[0].header["filter"]

                SOLVE, ASTAP, LOCAL = _astro_utilities.checkPSolve(fpath)
                print(SOLVE, ASTAP, LOCAL)
                
                if SOLVE :
                    try : 
                        wcs = WCS(hdul[0].header)

                        if 'PIXSCALE' in hdul[0].header:
                            PIX2ARCSEC = hdul[0].header['PIXSCALE']
                        else : 
                            PIX2ARCSEC = _astro_utilities.calPixScale(hdul[0].header['FOCALLEN'], 
                                                            hdul[0].header['XPIXSZ'],
                                                            hdul[0].header['XBINNING'])
                            
                        if hdul[0].header['CCDNAME'] == 'STX-16803' :
                            val_figsize=(10, 9)
                            val_fraction = 0.0455
                        else :
                            val_figsize=(12, 9)
                            val_fraction = 0.0035

                        # It is used as a rough estimate, so no need to be accurate:
                        PIX2ARCSEC = hdul[0].header["PIXSCALE"]
                        
                        if "EGAIN" in hdul[0].header :
                            gain = hdul[0].header["EGAIN"]
                        elif "GAIN" in hdul[0].header :
                            gain = hdul[0].header["GAIN"]
                        else :  
                            gain = _astro_utilities.CCDDIC[hdul[0].header["CCDNAME"]]["GAIN"]

                        if "RDNOISE" in hdul[0].header :
                            rdnoise = hdul[0].header["RDNOISE"]
                        else :
                            rdnoise = _astro_utilities.CCDDIC[hdul[0].header["CCDNAME"]]["RDNOISE"]
                        if verbose == True :
                            print(f"gain : {gain},  rdnoise : {rdnoise},  PIX2ARCSEC : {PIX2ARCSEC}")
                        
                        # D.2. Find the observation time and exposure time to set the obs time
                        t_start = Time(hdul[0].header['DATE-OBS'], format='isot')
                        t_expos = hdul[0].header['EXPTIME'] * u.s
                        t_middle = t_start + t_expos / 2 # start time + 0.5 * exposure time
                        if verbose == True :
                            print(f"t_start: {t_start}, t_expos: {t_expos}, t_middle: {t_middle}")
                        
                        # Get the radius of the smallest circle which encloses all the pixels
                        rad = yfu.fov_radius(header=hdul[0].header,
                                            unit=u.deg)
                        cent_coord = yfu.center_radec(ccd_or_header=hdul[0].header, 
                                                            center_of_image=True)
                        pos_sky = SkyCoord(cent_coord, unit='deg')
                        pos_pix = pos_sky.to_pixel(wcs=wcs)
                        
                        if verbose == True :
                            print("rad: {}".format(rad))  # 시야각(FOV)으로 구한 반지름
                            print("cent_coord: {}".format(cent_coord))
                            print("pos_sky: {}".format(pos_sky))
                            print("pos_pix: {}".format(pos_pix))

                        #%%                        
                        try :
                            Mag_delta = Mag_delta_INIT
                            ps1 = ypu.PanSTARRS1(cent_coord.ra, cent_coord.dec, radius=rad,
                                        column_filters={"rmag":f"{Mag_target-Mag_delta}..{Mag_target+Mag_delta}",
                                        "e_rmag":"<0.10", "nr":">5"})
                            PS1_stars_all = ps1.query()

                            if len(PS1_stars_all) < 10 :
                                Mag_delta = Mag_delta_INIT + 2
                                ps1 = ypu.PanSTARRS1(cent_coord.ra, cent_coord.dec, radius=rad,
                                            column_filters={"rmag":f"{Mag_target-Mag_delta}..{Mag_target+Mag_delta}",
                                            "e_rmag":"<0.10", "nr":">5"})
                                PS1_stars_all = ps1.query()
                                
                        except :
                            Mag_delta = Mag_delta_INIT + 2
                            ps1 = ypu.PanSTARRS1(cent_coord.ra, cent_coord.dec, radius=rad,
                                        column_filters={"rmag":f"{Mag_target-Mag_delta}..{Mag_target+Mag_delta}",
                                        "e_rmag":"<0.10", "nr":">5"})
                            PS1_stars_all = ps1.query()

                        if verbose == True :
                            print("type(PS1_stars_all) :", type(PS1_stars_all))
                            print("len(PS1_stars_all) :", len(PS1_stars_all))

                        isnear = ypu.organize_ps1_and_isnear(
                                            ps1, 
                                            # header=ccd.header+ccd.wcs.to_header(), 
                                            ccd.header+ccd.wcs.to_header(), 
                                            # bezel=5*FWHM_INIT*PIX2ARCSEC.value,
                                            # nearby_obj_minsep=5*FWHM_INIT*PIX2ARCSEC.value,
                                            bezel=5*FWHM_INIT*PIX2ARCSEC,
                                            nearby_obj_minsep=5*FWHM_INIT*PIX2ARCSEC,
                                            group_crit_separation=6*FWHM_INIT
                                        )
                        df_stars_all = PS1_stars_all.to_pandas()
                        df_stars = ps1.queried.to_pandas()
                        # print("len(df_stars):", len(df_stars))
                        df_stars = df_stars.dropna(subset=["gmag", "rmag"])
                        # if len(df_stars) > 100 :
                        #     df_stars = df_stars[:100]
                        if verbose == True :
                            print("len(df_stars_all):", len(df_stars_all))
                            print("len(df_stars):", len(df_stars))

                        pos_stars_all = np.array([df_stars_all["RAJ2000"].array, df_stars_all["DEJ2000"].array]).T
                        pos_stars_all = SkyCoord(pos_stars_all, **SKYC_KW).to_pixel(wcs)
                        pos_stars_all = np.transpose(pos_stars_all)
                        # pos_stars_all   # PS1 query 모든 별

                        pos_stars = np.array([df_stars["RAJ2000"].array, df_stars["DEJ2000"].array]).T
                        pos_stars = SkyCoord(pos_stars, **SKYC_KW).to_pixel(wcs)
                        pos_stars = np.transpose(pos_stars)
                        # pos_stars     # PS1 query 중 비교 측광에 사용될 별

                        ap_stars = CAp(positions=pos_stars, r=R_AP)
                        ap_stars_all = CAp(positions=pos_stars_all, r=R_AP)

                        #apert
                        an_stars = CAn(positions=pos_stars, r_in=R_IN, r_out=R_OUT)
                        an_stars_all = CAn(positions=pos_stars_all, r_in=R_IN, r_out=R_OUT)

                        phot_PS1 = ypu.apphot_annulus(hdul[0].data, 
                                                    ap_stars, an_stars, error=yfu.errormap(hdul[0].data))

                        #%%
                        if verbose==True :
                            print("df_targ:", df_targ)
                            print(f"wcs : {wcs}")

                        # pos_pix_targ_init = SkyCoord(df_targ["ra"].values, df_targ["dec"].values, **SKYC_KW).to_pixel(wcs)
                        pos_sky_targ_init = SkyCoord(df_targ["RA"].values, df_targ["DEC"].values, 
                                                    unit=(u.hourangle, u.deg),
                                                    frame='icrs')

                        pos_pix_targ_init = pos_sky_targ_init.to_pixel(wcs)

                        ap_targ = CAp([pos_pix_targ_init[0][0], pos_pix_targ_init[1][0]], r=R_AP)
                        an_targ = CAn([pos_pix_targ_init[0][0], pos_pix_targ_init[1][0]], r_in=R_IN, r_out=R_OUT)

                        phot_targ = ypu.apphot_annulus(hdul[0].data, 
                                                    ap_targ, an_targ, error=yfu.errormap(hdul[0].data))

                        if verbose==True :
                            print("pos_pix_targ_init:", pos_pix_targ_init)
                            print("ap_targ:", ap_targ)
                            print("an_targ:", an_targ)
                            print("phot_PS1:", phot_PS1)
                            print("phot_targ:", phot_targ)
                            print("df_stars:", df_stars)

                        df_phot_PS1 = pd.concat([df_stars, phot_PS1], axis=1)
                        if verbose==True :
                            print("df_phot_PS1:\n", df_phot_PS1)    
                            print("df_phot_PS1.columns:", df_phot_PS1.columns) 

                        #%%
                        #####################################################
                        # Plotting #1 f"{DIFFPRESULTDIR}/{fpath.stem}_PS1_magnitude.png"
                        #####################################################
                        fig, axs = plt.subplots(1, 1, figsize=val_figsize,
                                                subplot_kw={'projection': wcs},
                                                sharex=False, sharey=False, gridspec_kw=None)

                        im = _astro_utilities.zimshow(axs, hdul[0].data, )

                        ap_stars_all.plot(axs, color='w', lw=1)
                        # ap_stars.plot(axs, color='r', lw=1)

                        # ap_targ.plot(axs, color="r")
                        an_stars_all.plot(axs, color="w")
                        an_stars.plot(axs, color="orange")
                        an_targ.plot(axs, color="r")

                        for i, row in df_phot_PS1.iterrows():
                            axs.text(row['xcenter']+10, row['ycenter']+10, f"star {i}: {row[f'{(flt.upper())}mag']:.01f}", fontsize=8, color="w")  
                        
                        try :
                            axs.text(pos_pix_targ_init[0][0]+10, pos_pix_targ_init[1][0]+10, f"{targ_name}", fontsize=8, color="r")
                        except:
                            pass
                        axs.coords.grid(True, color='white', ls=':')
                        axs.coords['ra'].set_axislabel('Right Ascension (J2000)', minpad=0.5, fontsize=8)
                        axs.coords['ra'].set_ticklabel_position('b')
                        axs.coords['dec'].set_axislabel('Declination (J2000)', minpad=0.4, fontsize=8)
                        axs.coords['dec'].set_ticklabel_position('l')
                        axs.coords['ra'].set_major_formatter('hh:mm')
                        axs.coords['dec'].set_major_formatter('dd:mm')

                        axs.coords['ra'].display_minor_ticks(True)
                        axs.coords['dec'].display_minor_ticks(True)
                        axs.coords['ra'].set_minor_frequency(2)
                        axs.coords['dec'].set_minor_frequency(2)
                        axs.tick_params(labelsize=8)

                        cbar = plt.colorbar(im, ax = axs, fraction=val_fraction, pad=0.04, )

                        axs.set_title(f"fname: {fpath.name}\n {targ_name} : {flt.upper()} ${{{flt.upper()}}}_{{PS1}}$ of comparison stars (PS1 query : ${{{Mag_target}}} \pm {{{Mag_delta}}}$)", fontsize=10,)
                        axs.annotate(f'Number of comparison star(s): {len(pos_stars)}\nNumber of all PS1 star(s): {len(pos_stars_all)}', fontsize=8,
                                xy=(0, 0), xytext=(10, -20), va='top', ha='left',
                                xycoords='axes fraction', textcoords='offset points')
                        axs.annotate(f"{targ_name}\n${{{flt.upper()}}}_{{inst}} = {{{phot_targ['mag'][0]:+.03f}}} \pm {{{phot_targ['merr'][0]:.03f}}}$\nsnr: {phot_targ['snr'][0]:.02f}", fontsize=8,
                                xy=(1, 0), xytext=(-100, -20), va='top', ha='left',
                                xycoords='axes fraction', textcoords='offset points')

                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_PS1_magnitude.png")
                        if verbose == True :
                            print(f"{DIFFPRESULTDIR}/{fpath.stem}_PS1_magnitude.png is saved...")	
                        # plt.show()
                        plt.clf()
                        plt.close('all')

                        #%%
                        ######################################################
                        ## Plotting #2 calculate error
                        ######################################################
                        df_phot_stars = df_phot_PS1[df_phot_PS1["merr"] < ERR_Minimum]
                        df_phot_stars_na = df_phot_stars.dropna(subset=[f"{flt.upper()}mag", "mag", f"e_{flt.upper()}mag", "merr", "grcolor", "e_grcolor"])
                        if verbose == True :
                            print("len(df_phot_stars_na):", len(df_phot_stars_na))
                            print("df_phot_stars_na:\n", df_phot_stars_na)    
                        # phot_stars_na = phot_stars_na.set_index('id', drop=True)
                        # df_phot_stars = df_phot_PS1_na.reset_index(drop=True)

                        merr_total1 = np.sqrt((df_phot_stars_na["merr"])**2 + (df_phot_stars_na[f"e_{flt.upper()}mag"])**2)

                        # === Calculate zero point and errors
                        _xx = np.linspace(Mag_target-Mag_delta, Mag_target+Mag_delta)
                        zeropt_med = np.median(df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt.upper()}mag"])
                        # zeropt_avg = np.average(df_phot_stars["mag"] - df_phot_stars[f"{flt.upper()}mag"],
                        #                         weights=1/merr_total1**2)
                        dzeropt = np.max([1/np.sqrt(np.sum(1/(merr_total1)**2)),
                                        np.std((df_phot_stars_na[f"e_{flt.upper()}mag"] - df_phot_stars_na["merr"]), ddof=1)/np.sqrt(len(df_phot_stars[f"{flt.upper()}mag"]))])
                        merr_total2 = np.sqrt(np.sqrt(merr_total1**2 + dzeropt**2))

                        # === Find fitting lines
                        # Search for the usage of scipy.optimize.curve_fit.
                        # poptm, _ = curve_fit(_astro_utilities.linf, df_phot_stars[f"{flt.upper()}mag"],
                        #                     df_phot_stars["mag"],
                        #                     sigma= df_phot_stars["merr"], absolute_sigma=True)
                        # poptc, _ = curve_fit(_astro_utilities.linf, df_phot_stars["grcolor"],
                        #                     df_phot_stars["mag"] - df_phot_stars[f"{flt.upper()}mag"],
                        #                     sigma=merr_total2, absolute_sigma=True)
                        # #%%
                        # df_phot_stars = df_phot_stars.dropna(subset=[f"{flt.upper()}mag", "mag", f"e_{flt.upper()}mag", "merr", "grcolor", "e_grcolor"])
                        # if verbose == True :
                        #     print("len(df_phot_stars):", len(df_phot_stars))
                        #     print("df_phot_stars:\n", df_phot_stars)    
                        #     print("len(df_phot_stars):", len(df_phot_stars))
                        #%%
                        ######################################################
                        ## Plotting #3 f"{DIFFPRESULTDIR}/{fpath.stem}_standardization.png"
                        ######################################################
                        errkw = dict(marker="", ls="", ecolor="gray", elinewidth=0.5)

                        def plot_common(ax, x, y, xerr, yerr, title="", xlabel="", ylabel="", ylim=None):
                            ax.plot(x, y, '+')
                            ax.errorbar(x, y, xerr=xerr, yerr=yerr, **errkw)
                            ax.axhline(zeropt_med, color="r", lw=1, label=f"$Z = {{{zeropt_med:.3f}}} ± {{{dzeropt:.3f}}}$\n(median value)")
                            ax.hlines([zeropt_med + dzeropt, zeropt_med - dzeropt],
                                    *ax.get_xlim(), color=["r","r"], lw=1, ls=":")
                            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
                            # ax.legend(fontsize=8, loc='best')

                        fig, axs = plt.subplots(2, 3, figsize=(15, 6), sharex=False, sharey=False,
                                        gridspec_kw={'height_ratios': [1, 3]})

                        # 상단 행
                        plot_common(axs[0, 0], df_phot_stars_na[f"{flt.upper()}mag"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt.upper()}mag"],
                                        df_phot_stars_na[f"e_{flt.upper()}mag"], df_phot_stars_na["merr"],
                                        ylabel=f"${{{flt.upper()}}}_{{inst}} - {{{flt.upper()}}}_{{PS1}}$",
                                        ylim=(zeropt_med-0.8, zeropt_med+0.8),
                                    )


                        plot_common(axs[0, 1], df_phot_stars_na["grcolor"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt.upper()}mag"],
                                    df_phot_stars_na[f"e_grcolor"], merr_total2,
                                    title=f"${{{flt.upper()}}}_{{inst}} - {{{flt.upper()}}}_{{PS1}} = (z + k'X) + (k''X + k)C$",
                                    ylabel=f"${{{flt.upper()}}}_{{inst}} - {{{flt.upper()}}}_{{PS1}}$",
                                    ylim=(zeropt_med-0.8, zeropt_med+0.8),
                                    )
                            
                        # axs[0, 1].plot(axs[0, 1].get_xlim(), _astro_utilities.linf(np.array(axs[0, 1].get_xlim()), *poptc),
                        #             "g-", lw=1, label=f"$y = {{{poptc[1]:+.3f}}}x {{{poptc[0]:+.3f}}}$\n(curve_fit)")
                        # axs[0, 1].legend(fontsize=8, loc='best')

                        data = df_phot_stars_na[["mag", "merr", "grcolor", "e_grcolor"]]
                        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=axs[0, 2])
                        axs[0, 2].set(title='Correlation Heatmap')

                        # 하단 행
                        plot_common(axs[1, 0], df_phot_stars_na[f"{flt.upper()}mag"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt.upper()}mag"],
                                            df_phot_stars_na[f"e_{flt.upper()}mag"], df_phot_stars_na["merr"],
                                            xlabel=f"${{{flt.upper()}}}_{{PS1}}$ (PS1 to {flt.upper()} filter by Tonry+2012)",
                                            ylabel=f"${{{flt.upper()}}}_{{inst}} - {{{flt.upper()}}}_{{PS1}}$")

                        plot_common(axs[1, 1], df_phot_stars_na["grcolor"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt.upper()}mag"],
                                            df_phot_stars_na[f"e_grcolor"], merr_total2,
                                            xlabel="$g - r$ (PS1)",
                                            ylabel=f"${{{flt.upper()}}}_{{inst}} - {{{flt.upper()}}}_{{PS1}}$")
                        axs[1, 0].legend(fontsize=8, loc='best')

                        # axs[1, 1].plot(axs[1, 1].get_xlim(), _astro_utilities.linf(np.array(axs[1, 1].get_xlim()), *poptc),
                        #             "g-", lw=1, label=f"$y = {{{poptc[1]:+.3f}}}x {{{poptc[0]:+.3f}}}$\n(curve_fit)")
                        axs[1, 1].legend(fontsize=8, loc='best')

                        axs[1, 2].plot(_xx, _xx + zeropt_med,
                                            label=f"${{{flt.upper()}}}_{{inst}} = {{{flt.upper()}}}_{{PS1}}+({{{zeropt_med:.3f}}} \pm {{{dzeropt:.3f}}})$\n(median vlaue)",
                                            color="r", lw=1, ls="-")
                        axs[1, 2].plot(_xx, _xx + zeropt_med+phot_targ['merr'][0],
                                            color="r", lw=1, ls=":")
                        axs[1, 2].plot(_xx, _xx + zeropt_med-phot_targ['merr'][0],
                                            color="r", lw=1, ls=":")
                        # axs[1, 2].plot(axs[1, 2].get_xlim(), _astro_utilities.linf(np.array(axs[1, 2].get_xlim()), *poptm),
                        #             "g-", lw=1, label=f"$y = {{{poptm[1]:.3f}}}x {{{poptm[0]:+.3f}}}$\n(curve_fit)")
                        axs[1, 2].plot(df_phot_stars_na[f"{flt.upper()}mag"], df_phot_stars_na["mag"], '+')
                        axs[1, 2].axhline(phot_targ["mag"].values, label=f"{targ_name}: ${{{flt.upper()}}}_{{inst}} = {{{phot_targ['mag'][0]:+.03f}}} \pm {{{phot_targ['merr'][0]:.03f}}}$")
                        # axs[1, 2].axhline([phot_targ["mag"].values + phot_targ['merr'][0], phot_targ["mag"].values - phot_targ['merr'][0]],
                        #                   *axs[1, 2].get_xlim(), color=["b","b"], lw=1, ls=":")
                        axs[1, 2].axhline(phot_targ["mag"].values + phot_targ['merr'][0],
                                            *axs[1, 2].get_xlim(), color="b", lw=1, ls=":")
                        axs[1, 2].axhline(phot_targ["mag"].values - phot_targ['merr'][0],
                                            *axs[1, 2].get_xlim(), color="b", lw=1, ls=":")
                        axs[1, 2].errorbar(df_phot_stars_na[f"{flt.upper()}mag"],
                                            df_phot_stars_na["mag"],
                                            xerr=df_phot_stars_na[f"e_{flt.upper()}mag"],
                                            yerr=df_phot_stars_na["merr"],
                                            **errkw)
                        axs[1, 2].set(
                                            xlabel=f"${{{flt.upper()}}}_{{PS1}}$ (PS1 to {flt.upper()} filter by Tonry+2012)",
                                            ylabel =f"${{{flt.upper()}}}_{{inst}}$",
                                )
                        axs[1, 2].legend(fontsize=8, loc='best')
                        axs[1, 2].axis('square')

                        # ID 텍스트 추가
                        for _, row in df_phot_stars_na.iterrows():
                            for i in range(2):
                                for j in range(2):
                                    axs[i, j].text(row[f"{flt.upper()}mag" if j == 0 else "grcolor"],
                                                row["mag"] - row[f"{flt.upper()}mag"], int(row["id"]), fontsize=8, clip_on=True)
                            axs[1, 2].text(row[f"{flt.upper()}mag"], row["mag"], int(row["id"]), fontsize=8, clip_on=True)

                        # x축 레이블 숨기기 (상단 행)
                        for ax in axs[0, :2]:
                            ax.tick_params(labelbottom=False)

                        plt.suptitle(f"fname: {fpath.name}\nPS1 check for differential photometry (PS1 query : ${{{Mag_target}}} \pm {{{Mag_delta}}}$), {targ_name}: ${{{flt.upper()}}}_{{result}} = {{{(phot_targ['mag'][0]-zeropt_med):.3f}}} \pm {{{(np.sqrt((dzeropt)**2+(phot_targ['merr'][0])**2)):.03f}}}$", fontsize=10)
                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_standardization_extended.png")
                        if verbose == True :
                            print(f"{DIFFPRESULTDIR}/{fpath.stem}_standardization_extended.png is saved...")
                        # plt.show()
                        plt.clf()
                        plt.close('all')

                        #%%
                        #####################################################
                        # createf"{DIFFPRESULTDIR}/{fpath.stem}_result_photometry.csv"
                        #####################################################
                        df_phot_PS1 = pd.concat([df_phot_PS1, phot_targ], axis=0)
                        df_phot_PS1 = df_phot_PS1.reset_index(drop=True)
                        if verbose==True :
                            print("df_phot_PS1:\n", df_phot_PS1)
                            print("df_phot_PS1.columns:", df_phot_PS1.columns)    
                            print("len(df_phot_PS1):", len(df_phot_PS1))    
                        print(len(df_phot_PS1))
                        df_phot_PS1.at[len(df_phot_PS1)-1, 'OBJ'] = targ_name
                        df_phot_PS1.at[len(df_phot_PS1)-1, 'RAJ2000'] = pos_sky_targ_init.ra.degree
                        df_phot_PS1.at[len(df_phot_PS1)-1, 'DEJ2000'] = pos_sky_targ_init.dec.degree
                        if verbose==True :
                            print("df_phot_PS1:\n", df_phot_PS1)
                            print("df_phot_PS1.columns:", df_phot_PS1.columns)    
                            print("len(df_phot_PS1):", len(df_phot_PS1))  

                        df_phot_PS1['filename'] = fpath.stem
                        df_phot_PS1['t_start'] = t_start
                        df_phot_PS1['t_expos'] = t_expos
                        df_phot_PS1['t_middle'] = t_middle
                        df_phot_PS1['filter'] = flt
                        df_phot_PS1["zeropt_med"] = zeropt_med
                        # df_phot_PS1["zeropt_avg"] = zeropt_avg
                        df_phot_PS1["e_zeropt"] = dzeropt

                        df_phot_PS1[f"{flt.upper()}_magnitude"] = df_phot_PS1["mag"] - df_phot_PS1["zeropt_med"]
                        df_phot_PS1[f"{flt.upper()}_magerr"] = np.sqrt((dzeropt)**2+(df_phot_PS1['merr'])**2)
                        if verbose == True :
                            print("df_phot_PS1:\n", df_phot_PS1)
                            print("len(df_phot_PS1): ", len(df_phot_PS1))

                        df_phot_PS1.to_csv(f"{DIFFPRESULTDIR}/{fpath.stem}_result_photometry.csv")
                        if verbose == True :
                            print(f"{DIFFPRESULTDIR}/{fpath.stem}_result_photometry.csv is saved...")   

                        #%%
                        df_phot_PS1_na = df_phot_PS1.dropna(subset=[f"{flt.upper()}mag", "mag", f"e_{flt.upper()}mag", "merr", "grcolor", "e_grcolor"])
                        if verbose == True :
                            print("len(df_phot_PS1_na):", len(df_phot_PS1_na))
                            print("df_phot_stars:\n", df_phot_PS1_na)    
                            print("len(df_phot_PS1_na):", len(df_phot_PS1_na))
                        #%%
                        ######################################################
                        ## Plotting #5 f"{DIFFPRESULTDIR}/{fpath.stem}_Result_of_differential_photometry.png"
                        ######################################################
                        fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                                                sharex=False, sharey=False, gridspec_kw=None)

                        for idx, row in df_phot_PS1_na.iterrows():
                            im0 = axs[0, 0].errorbar(df_phot_PS1_na["id"],
                                        df_phot_PS1_na[f"{flt.upper()}_magnitude"], yerr=df_phot_PS1_na["merr"],
                                        marker='x',
                                        ls='none',
                                        #ms=10,
                                        capsize=3)

                        axs[0, 0].invert_yaxis()
                        axs[0, 0].set(
                            xlabel='Star ID',
                            ylabel=f"${{{flt.upper()}}}_{{obs}}$"
                            )

                        style = {'edgecolor': 'white', 'linewidth': 3}
                        im1 = axs[0, 1].hist(df_phot_PS1_na[f"{flt.upper()}_magnitude"],
                                    **style)
                        axs[0, 1].set(
                            xlabel=f"${{{flt.upper()}}}_{{obs}}$",
                            ylabel="number of stars"
                            )

                        # 상관관계 계산
                        data =  df_phot_PS1_na[[f"{flt.upper()}_magnitude", "merr"]]
                        corr = data.corr()

                        # 히트맵 그리기
                        im2 = sns.heatmap(corr, annot=True, cmap='coolwarm',
                                            vmin=-1, vmax=1, center=0, ax = axs[1, 0])
                        axs[1, 0].set(
                            title = 'Correlation Heatmap',
                            )

                        axs[1, 1].scatter(df_phot_PS1_na[f"{flt.upper()}_magnitude"], df_phot_PS1_na["merr"], marker='x',)
                        axs[1, 1].errorbar(x=df_phot_PS1_na[f"{flt.upper()}_magnitude"], y=df_phot_PS1_na["merr"],
                                    yerr=None, xerr=df_phot_PS1_na["merr"], fmt="o", color="gray", capsize=3, alpha=0.5)
                        axs[1, 1].set(
                            title = "Correlation between Magnitude and Error",
                            xlabel=f"${{{flt.upper()}}}_{{obs}}$",
                            ylabel="Error",
                            )

                        plt.suptitle(f"fname: {fpath.name}\n Result of differential photometry (Magnitude : {Mag_target}±{Mag_delta})", fontsize=10,)
                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_Result_of_differential_photometry.png")
                        # plt.show()
                        plt.clf()
                        plt.close('all')


                    except Exception as err :
                        print("X"*60)
                        _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                        pass