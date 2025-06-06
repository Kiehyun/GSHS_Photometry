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
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
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
tryASTROMETRYNET = True
file_age = 60
file_retry_dt = datetime(2025, 3, 3, 16, 17)
# file_retry_dt = datetime.now()
downsample = 4
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data")  
if platform.system() == "Windows":
    BASEDIR = Path("R:\\ASTRO_data")  

PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C2-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C3-EXO"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"    # finished 
TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin" # No fits nound
TODODIR = PROJECDIR / "-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin" # finished
TODODIR = PROJECDIR / "-_-_-_2025-01_-_GSON300_STF-8300M_-_1bin"  # finished
TODODIR = PROJECDIR / "-_-_-_2025-01_-_RiLA600_ASI6200MMPro_-_3bin"  # -1
TODODIR = PROJECDIR / "-_-_-_2025-02_-_GSON300_STF-8300M_-_1bin"  # -2
TODODIR = PROJECDIR / "-_-_-_2025-02_-_RiLA600_ASI6200MMPro_-_3bin"  # -3

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
    print ("BDFDIR: ", format(BDFDIR))
    MASTERDIR = Path(BDFDIR[0]) / _astro_utilities.master_dir
    if not MASTERDIR.exists():
        os.makedirs("{}".format(str(MASTERDIR)))
        print("{} is created...".format(str(MASTERDIR)))
    print ("MASTERDIR: ", format(MASTERDIR))
except Exception as err :
    print("X"*60)
    _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
    pass

DOINGDIRs = sorted([x for x in DOINGDIRs if "_LIGHT_" in str(x)])
# print ("DOINGDIRs: ", format(DOINGDIRs))
# print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

# filter_str = 'TT-ARI'
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
BDFDIR = Path(BDFDIR[0])
if verbose == True :
    print(f"Starting: {BDFDIR}")

MASTERDIR = BDFDIR / _astro_utilities.master_dir

summary = yfu.make_summary(BDFDIR/"*.fit*", 
                                verify_fix=True,
                                ignore_missing_simple=True,
                                verbose = verbose,
                                )
    
if summary is not None :
    if verbose == True :
        #print(summary)
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])

    if (MASTERDIR / "master_bias.fits").exists() and tryagain == False :
        if verbose == True :
            print("bias file is already exist....")
    else :
        summary_bias = summary.loc[summary["IMAGETYP"] == "BIAS"].copy()
        summary_bias.reset_index(inplace=True)
        if verbose == True :
            print("summary_bias", summary_bias)

        bias_fits = summary_bias["file"]
        if verbose == True :
            # print("type(bias_fits)", type(bias_fits))
            print("len(bias_fits)", len(bias_fits))
            # print("bias_fits", bias_fits)
        DOIT = False
        check_fpath = (MASTERDIR/f"master_bias.fits")
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

            bias_comb = yfu.group_combine(
                            bias_fits.tolist(),
                            type_key = ["IMAGETYP"],
                            type_val = ["BIAS"],
                            group_key = ["EXPTIME"],
                            fmt = "master_bias.fits",  # output file name format
                            outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                            combine = "med",
                            memlimit = 2.e+10,
                            verbose = False,
                        )

    summary_dark = summary.loc[summary["IMAGETYP"] == "DARK"].copy()
    summary_dark.reset_index(inplace=True)
    if verbose == True :
        print("summary_dark", summary_dark)

    if 'EXPTIME' in summary_dark :
        check_exptimes = summary_dark['EXPTIME'].drop_duplicates()
        check_exptimes = check_exptimes.reset_index(drop=True)
        if verbose == True :
            print("check_exptimes", check_exptimes)

        for exptime in check_exptimes :
            DOIT = False
            check_fpath = (MASTERDIR / f"master_dark_{exptime:.0f}sec.fits" )
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
            
                summary_dark_each = summary_dark.loc[summary_dark['EXPTIME'] == exptime]
                dark_fits = summary_dark_each['file']
                if verbose == True :
                    # print("type(dark_fits)", type(dark_fits))
                    print("len(dark_fits)", len(dark_fits))
                    # print("dark_fits", dark_fits)

                dark_comb = yfu.group_combine(
                            dark_fits.tolist(),
                            type_key = ["IMAGETYP"],
                            type_val = ["DARK"],
                            group_key = ["EXPTIME"],
                            fmt = "master_dark_{:.0f}sec.fits",  # output file name format
                            outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                            combine = "med",
                            memlimit = 2.e+10,
                            verbose = False,
                        )
    dark_fpaths = sorted(list((MASTERDIR).glob('*dark*.fit*')))
    if verbose == True : 
        print(f"dark_fpaths: {dark_fpaths}")
        print(f"len(dark_fpaths): {len(dark_fpaths)}")
                
    summary_flat = summary.loc[summary["IMAGETYP"] == "FLAT"].copy()
    summary_flat.reset_index(inplace=True)

    if 'FILTER' in summary_flat :
        check_filters = summary_flat['FILTER'].drop_duplicates()
        check_filters = check_filters.reset_index(drop=True)
        if verbose == True :
            print("check_filters", check_filters)

        for filter in check_filters :
            DOIT = False
            check_fpath = (MASTERDIR / f"master_flat_{filter:s}.fits" )
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
            
                summary_flat_each = summary_flat.loc[summary_flat['FILTER'] == filter]
                flat_fits = summary_flat_each['file']
                if verbose == True :
                    # print("type(flat_fits)", type(flat_fits))
                    print("len(flat_fits)", len(flat_fits))
                    # print("flat_fits", flat_fits)

                try : 
                    flat_comb= yfu.group_combine(
                                    flat_fits.tolist(),
                                    type_key = ["IMAGETYP"],
                                    type_val = ["FLAT"],
                                    group_key = ["FILTER"],
                                    fmt = "master_flat_{:s}_norm.fits",  # output file name format
                                    scale="med_sc", #norm
                                    scale_to_0th=False, #norm
                                    outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                                    combine = "med",
                                    memlimit = 2.e+10,
                                    verbose=verbose,
                                )
                except : 
                    flat_comb = yfu.group_combine(
                                flat_fits.tolist(),
                                type_key = ["IMAGETYP"],
                                type_val = ["FLAT"],
                                group_key = ["FILTER"],
                                fmt = "master_flat_{:s}.fits",  # output file name format
                                #scale="med_sc", #norm
                                #scale_to_0th=False, #norm
                                outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                                combine = "med",
                                memlimit = 2.e+10,
                                verbose=verbose,
                            )
    flat_fpaths = sorted(list((MASTERDIR).glob('*flat*.fit*')))
    if verbose == True : 
        print(f"flat_fpaths: {flat_fpaths}")
        print(f"len(flat_fpaths): {len(flat_fpaths)}")
