# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

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
BASEDIR = "../Post_process/GSON300_STF-8300M_1bin/"
OBSRAWDIR = Path(_astro_utilities.CCD_obs_dir)

#%%
BASEDIRs = sorted(_Python_utilities.getFullnameListOfsubDir(BASEDIR))
print ("BASEDIRs: {}".format(BASEDIRs))

#%%
for BASEDIR in BASEDIRs[:] :
    print ("Starting...\n{}".format(BASEDIR))

    BASEDIR = Path(BASEDIR)
    folder_el = BASEDIR.parts[-1].split("_")
    #print(folder_el)    
    Object_name = folder_el[0]
    if folder_el[5].find("-") == -1 :
        Optic_name = folder_el[5]
        OptAcc_name = ""
    else : 
        Optic_name = folder_el[5].split("-")[0]
        OptAcc_name = folder_el[5].split("-")[1]
    #print(Optic_name)
    #print(OptAcc_name)
    Ccd_name = folder_el[6]
    Bin_name = folder_el[8]

    Obs_date = datetime.strptime('2020-07-18', '%Y-%m-%d')
    print(Obs_date)

    #%%
    SearchRAWCAL = OBSRAWDIR / f"{Ccd_name}_{Bin_name}" / "cal"
    bdRAWDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(SearchRAWCAL))
    #print("bdRAWDIRs: ", bdRAWDIRs)
    print("len(bdRAWDIRs): ", len(bdRAWDIRs))

    bRAWDIRs = [w for w in bdRAWDIRs if "bias" in w.lower()]
    print("bRAWDIRs: ", bRAWDIRs)
    print("len(bRAWDIRs): ", len(bRAWDIRs))

    b_date = []
    for bRAWDIR in bRAWDIRs :
        bRAWDIR = Path(bRAWDIR)
        bfolder_el = bRAWDIR.parts[-1].split("_")
        b_date.append(datetime.strptime(bfolder_el[3], '%Y-%m-%d'))
    #%%
    #print(b_date)
    #print(type(Obs_date))
    #print(Obs_date)
    #print(b_date[0]-Obs_date)
    near_idx = _Python_utilities.nearest_ind(b_date, Obs_date)
    print(near_idx)
    
    
        

#%%
OBSRAWDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(SCRAWLIGHT))
OBSRAWDIRs = [w for w in OBSRAWDIRs if ((Object_name) in w
                                    or (".tmp") in w)]

print ("OBSRAWDIRs: {}".format(OBSRAWDIRs))

#%%
for BASEDIR in OBSRAWDIRs[4:5] :
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
            
            # yfu.fits_newpath(Path(fpath), 
            #         #rename_by=["FILTER", "OBJECT", "EXPTIME"], 
            #         #mkdir_by=["FILTER"],
            #         archive_dir = PostDIR,
            #         overwrite=True
            #         )

            # yfu.fitsrenamer(Path(fpath), 
            #         #rename_by=["FILTER", "OBJECT", "EXPTIME"], 
            #         #mkdir_by=["FILTER"],
            #         archive_dir = PostDIR,
            #         overwrite=True
            #         )

#%%
    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        print("{} is created...".format(str(REDUCEDDIR)))


#%%
for BASEDIR in BASEDIRs[4:] :
    print ("Starting...\n{}".format(BASEDIR))

    BASEDIR = Path(BASEDIR)
    
    MASTERDIR = BASEDIR / _astro_utilities.master_dir
    REDUCEDDIR = BASEDIR / _astro_utilities.reduced_dir

    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        print("{} is created...".format(str(REDUCEDDIR)))



BASEDIRs = sorted(_Python_utilities.getFullnameListOfsubDir(BASEDIR))
print ("BASEDIRs: {}".format(BASEDIRs))
print ("len(BASEDIRs): {}".format(len(BASEDIRs)))

   
    OBSRAWDIR = BASEDIR / _astro_utilities.CCD_obs_dir
    MASTERDIR = BASEDIR / _astro_utilities.master_dir
    REDUCEDDIR = BASEDIR / _astro_utilities.reduced_dir
    SOLVEDDIR = BASEDIR / _astro_utilities.solved_dir
    RESULTDIR = BASEDIR / _astro_utilities.DAOfinder_result_dir

    #%%
    summary = yfu.make_summary(BASEDIR/"*.fit*")
    if summary.empty:
        pass
    else:
        print(summary)
        print("len(summary):", len(summary))
        print(summary["file"][0])
        
        #%%
        light_fits = summary[summary["IMAGETYP"] == "LIGHT"]
        if light_fits.empty :
            pass
        else:
            print("len(light_fits):", len(light_fits))
            print("light_fits", light_fits)
            #%%
            light_fits["DATE-LOC-DT"] = pd.to_datetime(light_fits["DATE-LOC"])
            print("light_fits", light_fits)
            obs_date = light_fits["DATE-LOC-DT"].median().to_pydatetime().date()
            obs_date.strftime("%Y-%m-%d")
            print("BASEDIR.parts:", BASEDIR.parts)
            "{}_{}".format(BASEDIR.parts[-1].split("_")[5], BASEDIR.parts[-1].split("_")[6])
            
            calBDDIR = Path("../CCD_obs_raw/") /  \
                                "{}_{}".format(BASEDIR.parts[-1].split("_")[6], 
                                            BASEDIR.parts[-1].split("_")[8]) / "cal" 
            print("calBDDIR: ", calBDDIR)
            calBDDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(calBDDIR))
            print("calBDDIRs: ", calBDDIRs)

            #CCDOBSDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(CCD_obs_dir))


       
