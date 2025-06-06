# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

"""
#%%
from glob import glob
from pathlib import Path
import os
import platform
from datetime import datetime
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
count_stars = False
verbose = True
tryagain = True
trynightsky = True
tryASTROMETRYNET = True
file_age = 0.01
file_retry_dt = datetime(2025, 3, 3, 11)
downsample = 4
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data")  
if platform.system() == "Windows":
    BASEDIR = Path("R:\\ASTRO_data") 

PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"  #-1
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"  #-3
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"  
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C2-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C3-EXO"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin"
# TODODIR = PROJECDIR / "-_-_-_2025-01_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2025-01_-_RiLA600_ASI6200MMPro_-_3bin"
# TODODIR = PROJECDIR / "-_-_-_2025-02_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2025-02_-_RiLA600_ASI6200MMPro_-_3bin"

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
    _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
    pass

filter_strs = ['LIGHT',
                # '2025-02',
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                ]  # Example list of filter strings
DOINGDIRs = [x for x in DOINGDIRs if all(filter_str in str(x) for filter_str in filter_strs)]

rm_filter_strs = [
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                # 'Bad_fits', 
                #   'Another_bad_string',
                  ]  # Example list of filter strings to remove
DOINGDIRs = [x for x in DOINGDIRs if not any(rm_filter_str in str(x) for rm_filter_str in rm_filter_strs)]

if verbose == True :
    print ("DOINGDIRs: ", DOINGDIRs)
    print ("len(DOINGDIRs): ", len(DOINGDIRs))
#######################################################
#%%
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True :
        print(f"Starting: {str(DOINGDIR.parts[-1])}")
    
    sMASTERDIR = DOINGDIR / _astro_utilities.master_dir
    REDUCEDDIR = DOINGDIR / _astro_utilities.reduced_dir
    REDUC_nightsky = DOINGDIR / _astro_utilities.reduced_nightsky_dir
    MASTERDIR = Path(BDFDIR[0]) / _astro_utilities.master_dir

    if not sMASTERDIR.exists():
        os.makedirs(str(sMASTERDIR))
        if verbose == True :
            print("{} is created...".format(str(sMASTERDIR)))

    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        if verbose == True :
            print("{} is created...".format(str(REDUCEDDIR)))

    if not REDUC_nightsky.exists():
        os.makedirs("{}".format(str(REDUC_nightsky)))
        if verbose == True :
            print("{} is created...".format(str(REDUC_nightsky)))
    
    BADFITSDIR = DOINGDIR / _astro_utilities.Bad_fits_dir

    summary = yfu.make_summary(BADFITSDIR/"*.fit*",
                                verify_fix=True,
                                ignore_missing_simple=True,
                                )
    if summary is not None :
        if verbose == True :
            print("len(summary):", len(summary))
            print("summary:", summary)
            #print(summary["file"][0])  
        
        for _, row  in summary.iterrows():

            try : 
                fpath = Path(row["file"])
                if verbose == True :
                    print("fpath:", fpath)
                if (DOINGDIR / fpath.name).exists() : 
                    os.remove( DOINGDIR / fpath.name )
                    if verbose == True :
                        print(f"Remove {DOINGDIR / fpath.name}")
                if (REDUCEDDIR / fpath.name).exists() : 
                    os.remove( REDUCEDDIR / fpath.name )
                    if verbose == True :
                        print(f"Remove {REDUCEDDIR / fpath.name}")
                if (REDUC_nightsky / fpath.name).exists() :          
                    os.remove( REDUC_nightsky / fpath.name )
                    if verbose == True :
                        print(f"Remove {REDUC_nightsky / fpath.name}")  
            except Exception as err: 
                if verbose == True :
                    print("X"*60)
                _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                pass

    summary = yfu.make_summary(DOINGDIR/"*.fit*",
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

        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)

        summary_master = yfu.make_summary(MASTERDIR/"*.fit*", 
                           verbose = False,
                           )
        if verbose == True :
            print("summary_master", summary_master)

        summary_master_dark = summary_master.loc[summary_master["IMAGETYP"] == "DARK"].copy()
        summary_master_dark.reset_index(inplace=True)
        if verbose == True :
            print("summary_master_dark", summary_master_dark)

        if 'EXPTIME' in summary_master_dark :
            check_exptimes = summary_master_dark['EXPTIME'].drop_duplicates()
            check_exptimes = check_exptimes.reset_index(drop=True)
            if verbose == True :
                print("check_exptimes", check_exptimes)

        for _, row in df_light.iterrows():

            try : 
                fpath = Path(row["file"])
                ccd = yfu.load_ccd(fpath)
                filt = ccd.header["FILTER"]
                expt = ccd.header["EXPTIME"]

                idx = abs(summary_master_dark['EXPTIME'] - expt).idxmin()
                if verbose == True :
                    print(idx)

                DOIT = False
                check_fpath = (REDUCEDDIR / fpath.name)
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

                    if not (MASTERDIR / f"master_flat_{filt.upper()}_norm.fits").exists() :
                        if verbose == True :
                            print(f"{MASTERDIR}/master_flat_{filt.upper()}_norm.fits is not exists...")
                    else :
                        try :
                            if verbose == True :
                                print(f"Trying Reduction with master_dark_{expt:.0f}sec.fits ...")

                            red = yfu.ccdred(
                                ccd,
                                output=Path(f"{REDUCEDDIR/ fpath.name}"),
                                mdarkpath=str(MASTERDIR / "master_dark_{:.0f}sec.fits".format(expt)),
                                mflatpath=str(MASTERDIR / "master_flat_{}_norm.fits".format(filt.upper())),
                                # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                                overwrite=True,
                                )
                            
                        except : 
                            if verbose == True :
                                print(f"Trying Reduction with master_dark_{summary_master_dark['EXPTIME'][idx]:.0f}sec.fits is not exists...")
                                red = yfu.ccdred(
                                    ccd,
                                    output=Path(f"{REDUCEDDIR/ fpath.name}"),
                                    mdarkpath=str(MASTERDIR / f"master_dark_{summary_master_dark['EXPTIME'][idx]:.0f}sec.fits"),
                                    mflatpath=str(MASTERDIR / f"master_flat_{filt.upper()}_norm.fits"),
                                    dark_scale = True,
                                    exptime_dark = summary_master_dark['EXPTIME'][idx],
                                    # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                                    overwrite=True,
                                    )
                                
                        if verbose == True :
                            print (f"Reduce Reduce {fpath.name} +++...")
                    
            except Exception as err: 
                if verbose == True :
                    print("X"*60)
                _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                pass

    if trynightsky == True : 
        REDUCNSKYDIR = DOINGDIR / _astro_utilities.reduced_nightsky_dir
        if not REDUCNSKYDIR.exists():
            os.makedirs("{}".format(str(REDUCNSKYDIR)))
            if verbose == True :
                print("{} is created...".format(str(REDUCNSKYDIR)))
    
        summary = yfu.make_summary(REDUCEDDIR /"*.fit*")
        if summary is not None :
            if verbose == True :
                print("len(summary):", len(summary))
                print("summary:", summary)
                #print(summary["file"][0])   

            df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
            df_light = df_light.reset_index(drop=True)

            if 'FILTER' in df_light :
                check_filters = df_light['FILTER'].drop_duplicates()
                check_filters = check_filters.reset_index(drop=True)
                if verbose == True :
                    print("check_filters", check_filters)

                for filt in check_filters:
                #for filt in ["V"]:
                    df_light_filt = df_light.loc[df_light["FILTER"] == filt].copy()
                    
                    if df_light_filt.empty:
                        if verbose == True :
                            print(f"The dataframe(df_light_filt) {filt} is empty")
                        pass
                    else:
                        if verbose == True :
                            print("len(df_light_filt):", len(df_light_filt))
                            print("df_light_filt:", df_light_filt)

                        DOIT = False
                        check_fpath = (sMASTERDIR / f"nightskyflat-{filt}_norm.fits")
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

                            File_Num = 80
                            if len(df_light_filt["file"]) > File_Num :
                                combine_lst = df_light_filt["file"].tolist()[:File_Num]
                            else : 
                                combine_lst = df_light_filt["file"].tolist()
                            try : 
                                ccd = yfu.imcombine(
                                                    combine_lst, 
                                                    combine="med",
                                                    scale="avg", 
                                                    scale_to_0th=False, #norm
                                                    reject="sc", 
                                                    sigma=2.5,
                                                    verbose=verbose,
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
                                                    verbose=verbose,
                                                    memlimit = 2.e+11,
                                                    )
                            ccd.write(sMASTERDIR / f"nightskyflat-{filt}_norm.fits", overwrite=True)
                            print (f"nightskyflat-{filt}_nrom.fits is created +++...")

            for _, row in df_light.iterrows():
                try : 
                        
                    fpath = Path(row["file"])
                    ccd = yfu.load_ccd(REDUCEDDIR / fpath.name)
                    filt = row["FILTER"]

                    DOIT = False
                    check_fpath = (REDUCNSKYDIR / fpath.name)
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
 
                        try:    
                            ccd = yfu.ccdred(
                                            ccd, 
                                            mflatpath = sMASTERDIR / f"nightskyflat-{filt}_norm.fits",
                                            output = REDUCNSKYDIR / fpath.name,
                                            # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                                        )
                        except Exception as err: 
                            if verbose == True :
                                print("X"*60)
                            _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                            pass
                except Exception as err: 
                    if verbose == True :
                        print("X"*60)
                    _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                    pass
                    