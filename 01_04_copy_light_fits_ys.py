# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

#first time
cd ~/Downloads/ && git clone https://github.com/ysBach/ysvisutilpy && cd ysvisutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && git clone https://github.com/ysBach/ysfitsutilpy && cd ysfitsutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && git clone https://github.com/ysBach/ysphotutilpy && cd ysphotutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && git clone https://github.com/ysBach/SNUO1Mpy && cd SNUO1Mpy && git pull && pip install -e . && cd ..

# second time...
cd ~/Downloads/ysvisutilpy && git pull && pip install -e . 
cd ~/Downloads/ysfitsutilpy && git pull && pip install -e . 
cd ~/Downloads/ysphouutilpy && git pull && pip install -e . 
cd ~/Downloads/SNUO1Mpy && git pull && pip install -e . 

"""
#%%
from glob import glob
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd

import numpy as np
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData

import ysfitsutilpy as yfu
import ysphotutilpy as ypu
import ysvisutilpy as yvu

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
BASEDIR = "../RnE_2022/"
BASEDIR = "../RnE_2022/RiLA600_STX-16803_2bin/"
BASEDIR = "../RnE_2022/GSON300_STF-8300M/"
BASEDIR = _astro_utilities.base_dir
OBSRAWDIR = Path(_astro_utilities.CCD_obs_dir)

PostBASEDIR = Path("../Post_process/")

#%%
Object_name = "KLEOPATRA"   #"M45"
Optic_name = "RiLA600" #"GSON300"  #"FSQ106ED" #"FS60-CB" #"TMB130ss" #"FS60-CB"
OptAcc_name = ""    #"x80"
Ccd_name = "STX-16803"  #"STF-8300M"
Bin_name = "2bin"

SearchRAWLIGHT = Path(OBSRAWDIR / f"{Ccd_name}_{Bin_name}")
print("SearchRAWLIGHT:", SearchRAWLIGHT)

#%%
OBSRAWDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(SearchRAWLIGHT))
OBSRAWDIRs = [w for w in OBSRAWDIRs if ((Object_name) in w
                                    or (".tmp") in w)]
print ("OBSRAWDIRs: {}".format(OBSRAWDIRs))

#%%
for BASEDIR in OBSRAWDIRs[:] :
    print ("Starting...\n{}".format(BASEDIR))

    BASEDIR = Path(BASEDIR)
    #print(BASEDIR.parts[-1])
    PostDIR = PostBASEDIR / f"{Optic_name}{OptAcc_name}_{Ccd_name}_{Bin_name}" / BASEDIR.parts[-1]
    #PostDIR = PostBASEDIR / "{Optic_name}-{OptAcc_name}_{Ccd_name}_{Bin_name}" / BASEDIR.parts[-1]
    #print(PostDIR)
    
    if not PostDIR.exists():
        os.makedirs(str(PostDIR))
        print("{} is created...".format(str(PostDIR)))
    #%%
    summary = yfu.make_summary(BASEDIR/"*.fit*")
    if summary.empty:
        pass
    else:
        print(summary)
        print("len(summary):", len(summary))

        for fpath in summary["file"]:
            fpath = Path(fpath)
            print(fpath.name)
            
            shutil.copy(fpath, 
                        r'{0}/{1}'.format(str(PostDIR), fpath.name))
            print(fpath, " --> "
                    r'{0}{1}'.format(str(PostDIR), fpath.name))
            