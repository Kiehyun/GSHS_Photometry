# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

"""
#%%
from glob import glob
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _astro_utilities
import _Python_utilities

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
# read all files in base directory for processing
BASEDIR = Path("/mnt/Rdata/OBS_data") 
DOINGDIR = Path(BASEDIR / "asteroid/RiLA600_STX-16803_-_1bin")
DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(DOINGDIR))

print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))
#######################################################

Suwon = location = EarthLocation(lon=127.005 * u.deg, lat=37.308889 * u.deg, height=101 * u.m) 

#%%
summary_all = pd.DataFrame()

for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)
    
    DOINGDIR = DOINGDIR / _astro_utilities.reduced_dir2
    print("DOINGDIR", DOINGDIR)
    
    fits_in_dir = sorted(list(DOINGDIR.glob('*.fit*')))
    #print("fits_in_dir", fits_in_dir)
    print("len(fits_in_dir)", len(fits_in_dir))

    if len(fits_in_dir) == 0 :
        print(f"There is no fits fils in {DOINGDIR}")
        pass
    else : 
        print(f"Starting: {str(DOINGDIR.parts[-2])}")

        summary = yfu.make_summary(DOINGDIR/"*.fit*")
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])
        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)
        print("df_light:\n{}".format(df_light))

        for i, row  in df_light.iterrows():
            fpath = Path(row["file"])
            hdul = fits.open(fpath)

            SOLVE, ASTAP, LOCAL = _astro_utilities.checkPSolve(fpath)
            print(SOLVE, ASTAP, LOCAL)
            
            if SOLVE :
                w = WCS(hdul[0].header)
                t_obs = Time(hdul[0].header["DATE-OBS"]) + hdul[0].header["EXPOSURE"] * u.s / 2  # middle of observation time

                cent_coord = yfu.center_radec(ccd_or_header=hdul[0].header, center_of_image=True)
                #print(f"calcualted (RA, DEC): ({cent_coord.ra}, {cent_coord.dec})")
                #print(f"in header (RA, DEC): ({hdul[0].header['RA']*u.deg}, {hdul[0].header['DEC']*u.deg})")
                summary.loc[i, "offset_RA(arcmin)"] = (cent_coord.ra - hdul[0].header['RA']*u.deg).to(u.arcmin).value
                summary.loc[i, "offset_DEC(arcmin)"] = (cent_coord.dec - hdul[0].header['DEC']*u.deg).to(u.arcmin).value
                
                altaz = AltAz(obstime=t_obs, location=Suwon)

                cent_aa = cent_coord.transform_to(altaz)
                #print(f"calculated (Az, Alt): ({cent_aa.az}, {cent_aa.alt})")
                #print(f"in header (Az, Alt): ({hdul[0].header['CENTAZ ']*u.deg}, {hdul[0].header['CENTALT']*u.deg})")
                summary.loc[i, "offset_AZ(arcmin)"] = (cent_aa.az - hdul[0].header['CENTAZ']*u.deg).to(u.arcmin).value
                summary.loc[i, "offset_ALT(arcmin)"] = (cent_aa.alt - hdul[0].header['CENTALT']*u.deg).to(u.arcmin).value

        summary_all = pd.concat([summary_all, summary], axis = 0)
        summary.to_csv(f"RADEC/{str(DOINGDIR.parts[-2])}_mount_offset.csv")
summary_all.to_csv(f"RADEC/_mount_offset.csv")