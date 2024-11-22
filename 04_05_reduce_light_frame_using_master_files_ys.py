# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

이 파일은 관측 자료를 전처리 해준다.

"""
#%%
from glob import glob
from pathlib import Path
import os
import shutil
import numpy as np
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

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
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "03-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "04-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))
print ("DOINGDIRs: ", format(DOINGDIRs))
print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

try : 
    BDFDIR = [x for x in DOINGDIRs if "CAL-BDF" in str(x)]
    print ("BDFDIR: ", format(BDFDIR))
    MASTERDIR = Path(BDFDIR[0]) / _astro_utilities.master_dir
    if not MASTERDIR.exists():
        os.makedirs("{}".format(str(MASTERDIR)))
        print("{} is created...".format(str(MASTERDIR)))
    print ("MASTERDIR: ", format(MASTERDIR))
except : 
    pass

DOINGDIRs = sorted([x for x in DOINGDIRs if "_LIGHT_" in str(x)])
# print ("DOINGDIRs: ", format(DOINGDIRs))
# print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

# filter_str = '2017-01-13'
# DOINGDIRs = [x for x in DOINGDIRs if filter_str in str(x)]
# remove = 'BIAS'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'DARK'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'FLAT'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]

print ("DOINGDIRs: ", DOINGDIRs)
print ("len(DOINGDIRs): ", len(DOINGDIRs))
#%%
tryagain = False
trynightsky = True

for DOINGDIR in DOINGDIRs[5:] :
    DOINGDIR = Path(DOINGDIR)
    print(f"Starting: {str(DOINGDIR.parts[-1])}")
    
    sMASTERDIR = DOINGDIR / _astro_utilities.master_dir
    REDUCEDDIR = DOINGDIR / _astro_utilities.reduced_dir

    if not sMASTERDIR.exists():
        os.makedirs(str(sMASTERDIR))
        print("{} is created...".format(str(sMASTERDIR)))

    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        print("{} is created...".format(str(REDUCEDDIR)))

    summary = yfu.make_summary(DOINGDIR/"*.fit*")
    if summary is not None :
        #print(summary)
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])

        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)

        for _, row in df_light.iterrows():

            fpath = Path(row["file"])
            ccd = yfu.load_ccd(fpath)
            filt = ccd.header["FILTER"]
            expt = ccd.header["EXPTIME"]
            if (REDUCEDDIR/ fpath.name).exists() and not tryagain:
                print(f"reduction file already exists...\n{fpath.name}")
                pass
            else :
                try : 
                    red = yfu.ccdred(
                        ccd,
                        output = REDUCEDDIR/ fpath.name,
                        mbiaspath = MASTERDIR / "master_bias.fits",
                        mdarkpath = MASTERDIR / f"master_dark_{expt:.0f}sec.fits",
                        mflatpath = MASTERDIR / f"master_flat_{filt.upper()}_norm.fits",
                        # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                        dark_scale = False,
                        overwrite=True
                        )
                except : 
                    dark_exp = 100
                    red = yfu.ccdred(
                        ccd,
                        output = REDUCEDDIR / fpath.name,
                        mbiaspath = MASTERDIR / "master_bias.fits",
                        mdarkpath = MASTERDIR / f"master_dark_{dark_exp}sec.fits",
                        mflatpath = MASTERDIR / f"master_flat_{filt.upper()}_norm.fits",
                        dark_scale = True,
                        exptime_dark = dark_exp,
                        # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                        overwrite=True,
                        )

    if trynightsky == True : 
        REDUC_nightsky = DOINGDIR / _astro_utilities.reduced_nightsky_dir
        if not REDUC_nightsky.exists():
            os.makedirs("{}".format(str(REDUC_nightsky)))
            print("{} is created...".format(str(REDUC_nightsky)))
    
        summary = yfu.make_summary(REDUCEDDIR /"*.fit*")
        if summary is not None :
            print("len(summary):", len(summary))
            print("summary:", summary)
            #print(summary["file"][0])   

            df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
            df_light = df_light.reset_index(drop=True)

            for filt in ["B", "V", "R", "L", "R", "G", "B", "b", "v", "r"]:
            #for filt in ["V"]:
                df_light_filt = df_light.loc[df_light["FILTER"] == filt].copy()
                
                if df_light_filt.empty:
                    print(f"The dataframe(df_light_filt) {filt} is empty")
                    pass
                else:

                    print("len(df_light_filt):", len(df_light_filt))
                    print("df_light_filt:", df_light_filt)
                
                    if (sMASTERDIR / f"nightskyflat-{filt}.fits").exists() and not tryagain :
                        pass
                    else :      
                        File_Num = 80
                        if len(df_light_filt["file"]) > File_Num :
                            combine_lst = df_light_filt["file"].tolist()[:File_Num]
                        else : 
                            combine_lst = df_light_filt["file"].tolist()
                        if (sMASTERDIR / f"nightskyflat-{filt}_norm.fits").exists():
                            pass
                        else :
                            try : 
                                ccd = yfu.imcombine(
                                                    combine_lst, 
                                                    combine="med",
                                                    scale="avg", 
                                                    scale_to_0th=False, #norm
                                                    reject="sc", 
                                                    sigma=2.5,
                                                    verbose=True,
                                                    memlimit = 2.e+11,
                                                    )
                            except :
                                ccd = yfu.imcombine(
                                                    combine_lst, 
                                                    combine="med",
                                                    scale="avg", 
                                                    scale_to_0th=False, #norm
                                                    reject="sc", 
                                                    # sigma=2.5,
                                                    verbose=True,
                                                    memlimit = 2.e+11,
                                                    )
                            ccd.write(sMASTERDIR / f"nightskyflat-{filt}_norm.fits", overwrite=True)
                            print (f"Create Create nightskyflat-{filt}_norm.fits +++...")

            for _, row in df_light.iterrows():
                fpath = Path(row["file"])
                ccd = yfu.load_ccd(REDUCEDDIR / fpath.name)
                filt = row["FILTER"]
                if (REDUC_nightsky / fpath.name).exists() and not tryagain:
                    print(f"Nightsky reduction file already exists...\n{fpath.name}")
                    pass
                else :
                    try:    
                        ccd = yfu.ccdred(
                                        ccd, 
                                        mflatpath = sMASTERDIR / f"nightskyflat-{filt}_norm.fits",
                                        output = REDUC_nightsky / fpath.name
                                    )
                    except : 
                        # _Python_utilities.write_log(err_log_file, "FileNotFoundError") 
                        pass
