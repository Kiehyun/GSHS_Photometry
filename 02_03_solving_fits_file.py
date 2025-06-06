# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

"""
#%%
from glob import glob
from pathlib import Path
import os
import platform
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData

import ysfitsutilpy as yfu

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
    BASEDIR = Path("R:\\ASTRO_data")   

PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"      # finished
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"    # finished 
TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"    # finished
TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"    # finished

# PROJECDIR = BASEDIR / "C2-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C3-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-09_-_RiLA600_ASI6200MMPro_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin"

# PROJECDIR = BASEDIR / "C4-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

# DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))

PROJECDIR = BASEDIR / "A3_CCD_obs_raw"
# TODODIR = PROJECDIR / "STX-16803_1bin" 
# TODODIR = PROJECDIR / "STX-16803_2bin" 
# TODODIR = PROJECDIR / "STL-11000M_2bin"  #finished  #check finished
# TODODIR = PROJECDIR / "STF-8300M_2bin"            #check finished
# TODODIR = PROJECDIR / "QSI683ws_2bin"     #finished #check finished
# TODODIR = PROJECDIR / "STL-11000M_1bin"           #check finished
TODODIR = PROJECDIR / "STF-8300M_1bin"            #check finished
# TODODIR = PROJECDIR / "QSI683ws_1bin"             #check finished
# TODODIR = PROJECDIR / "ASI6200MMPro_3bin"             #check finished

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(TODODIR))

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

filter_strs = [
                '2025-01',
                '2025-02',
                '2025-03',
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                ]  # Example list of filter strings
DOINGDIRs = [x for x in DOINGDIRs if any(filter_str in str(x) for filter_str in filter_strs)]

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
    try : 
        DOINGDIR = Path(DOINGDIR)
        if verbose == True :
            print(f"Starting: {str(DOINGDIR.parts[-1])}")
        _astro_utilities.solving_fits_file(DOINGDIR,
                # downsample = downsample,
                # count_stars = count_stars,
                # tryagain = tryagain,
                # tryASTAP = False, # default True 
                tryASTROMETRYNET = True,  # default False
                # makeLOCALsh = True, # default False
                verbose = verbose,
                )      
    except Exception as err :
        print("X"*60)
        _Python_utilities.write_log(err_log_file, DOINGDIR, str(err), verbose=verbose)
        pass

    try :
        _astro_utilities.solving_fits_file(DOINGDIR,
            SOLVINGDIR = _astro_utilities.reduced_dir,
                # downsample = downsample,
                # count_stars = count_stars,
                # tryagain = tryagain,
                # tryASTAP = False, # default True 
                tryASTROMETRYNET = True,  # default False
                # makeLOCALsh = True, # default False
                verbose = verbose,
            ) 
    except Exception as err :
        print("X"*60)
        _Python_utilities.write_log(err_log_file, DOINGDIR, str(err), verbose=verbose)
        pass

    try :
        _astro_utilities.solving_fits_file(DOINGDIR,
            SOLVINGDIR = _astro_utilities.reduced_nightsky_dir,
                # downsample = downsample,
                # count_stars = count_stars,
                # tryagain = tryagain,
                # tryASTAP = False, # default True 
                tryASTROMETRYNET = True,  # default False
                # makeLOCALsh = True, # default False
                verbose = verbose,
            ) 
    except Exception as err :
        print("X"*60)
        _Python_utilities.write_log(err_log_file, DOINGDIR, str(err), verbose=verbose)
        pass

#%%

    _Python_utilities.write_log(log_file, 
                        f"{str(DOINGDIR)} is finighed..", 
                        verbose=verbose)
