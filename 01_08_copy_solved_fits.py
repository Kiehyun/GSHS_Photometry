# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user
"""
#%%
from glob import glob
from pathlib import Path
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pathlib import Path

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _astro_utilities
import _Python_utilities

plt.rcParams.update({'figure.max_open_warning': 0})

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
#%%
verbose = True
Overwrite = True
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data")  

PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C2-Asteroid"
TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "C3-EXO"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin"

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
    BDFDIR = Path(BDFDIR[0])    
except : 
    BDFDIR = TODODIR
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
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True : 
        print("DOINGDIR", DOINGDIR)
        print(f"Starting: {str(DOINGDIR.parts[-1])}")

    summary = yfu.make_summary(DOINGDIR/"*.fit*",
                                        verify_fix=True,
                                        ignore_missing_simple=True,
                                        )
    if summary is not None : 

        if verbose == True : 
            print("summary: ", summary)
            print("len(summary)", len(summary))

        for _, row in summary.iterrows():
        
            fpath = Path(row["file"])
            new_fname = ""
            suffix = ".fit"

            for KEY in _astro_utilities.fnameKEYs :
                if KEY in ["OBJECT", "IMAGETYP", "FILTER", 
                    "OPTIC", "CCDNAME"] :
                    new_fname += str(row[KEY])+"_"
                
                if KEY == "DATE-OBS" : 
                    new_fname += row[KEY][:19].replace("T","-").replace(":","-")+"_"


                if KEY == "EXPOSURE" : 
                    new_fname += str(int(row[KEY]))+"sec_"

                if KEY == "CCD-TEMP" : 
                    try:
                        new_fname += str(int(row[KEY]))+"c_"
                    except:
                        new_fname += (row[KEY])+"c_"
                if KEY == "XBINNING" : 
                    new_fname += str(row[KEY])+"bin"+suffix
            if verbose == True :
                print(new_fname)                      
            new_folder = _astro_utilities.get_new_foldername_from_filename(new_fname)
            new_fpath =  BASEDIR /_astro_utilities.CCD_obs_raw_dir / new_folder / new_fname
            
            if verbose == True :
                print("new_folder: ", new_folder)
                print("new_fpath: ", new_fpath)

            if not new_fpath.parents[0].exists():
                os.makedirs(f'{new_fpath.parents[0]}')
                if verbose == True :
                    print(f'{new_fpath.parts[-2]} is created')  
        
            if new_fpath.exists() :
                if verbose == True :
                    print(f'{new_fpath} is already exist')
                duplicate_fpath = BASEDIR / _astro_utilities.CCD_duplicate_dir / new_fpath.name
                if Overwrite == True:
                    shutil.copy2(str(fpath), str(new_fpath))
                    if verbose == True :
                        print(f"copy {str(fpath)} to {str(new_fpath)}")
                else :
                    # shutil.move(fpath, duplicate_fpath)
                    if verbose == True :
                        print(f'{fpath.parts[-1]} is already exist...')
            else : 
                shutil.copy2(str(fpath), str(new_fpath))
                if verbose == True :
                    print(f"copy {str(fpath.name)} to {str(new_fpath)}")
