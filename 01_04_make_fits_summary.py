
 """
Created on Thu Nov 22 01:00:19 2018
@author: user
"""
#%%
import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from ccdproc import CCDData

import time
from datetime import datetime, timedelta
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
BASEDIR = Path("/mnt/Rdata/ASTRO_data") 
DOINGDIR = Path('/mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/')

CCDDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(DOINGDIR))
print ("CCDDIRs: ", CCDDIRs)
print ("len(CCDDIRs): ", len(CCDDIRs))

#%%
#######################################################
checkTime_ccd = 1
checkTime_ccdoptic = 1
checkTime_eachdir = -2
for CCDDIR in CCDDIRs[:] :
    doing_status = False
    CCDDIR = Path(CCDDIR)
    ccd_fpath = Path(CCDDIR.parents[0] / f"summary_{CCDDIR.parts[-1]}.csv")
    print("ccd_fpath", ccd_fpath)

    summary_ccd = pd.DataFrame(columns=['file'])
    print(summary_ccd)   

    if not ccd_fpath.exists() :
        print(f"{str(ccd_fpath)} is not exist...")
        doing_status = True
    else : 
        t = os.path.getmtime(ccd_fpath)
        ccd_fpath_dt = datetime.fromtimestamp(t)
        print("ccd_fpath_dt: ", ccd_fpath_dt)
        doing_status = False
        if ccd_fpath_dt < datetime.now() + timedelta(days=checkTime_ccd) :
            print(f"{str(ccd_fpath)} is older more than {checkTime_ccd} days ...")
            doing_status = True
    
    ########################
    #doing_status = True
    ########################

    if doing_status == False :
        summary_ccd = pd.read_csv(str(ccd_fpath))
                
    else : 
        CCDOPTICDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(str(CCDDIR)))
        print ("CCDOPTICDIRs: ", CCDOPTICDIRs)
        print ("len(CCDOPTICDIRs): ", len(CCDOPTICDIRs))

        # remove = 'Cal'
        # CCDOPTICDIRs = [x for x in CCDOPTICDIRs if remove not in x]
        # print ("CCDOPTICDIRs: ", CCDOPTICDIRs)
        # print ("len(CCDOPTICDIRs): ", len(CCDOPTICDIRs))

        for CCDOPTICDIR in CCDOPTICDIRs[:] :
            doing_status = False
            CCDOPTICDIR = Path(CCDOPTICDIR)
            ccdoptic_fpath = Path(CCDOPTICDIR.parents[0] / f"summary_{CCDOPTICDIR.parts[-1]}.csv")
            print("ccdoptic_fpath", ccdoptic_fpath)

            summary_ccdoptic = pd.DataFrame(columns=['file'])
            print(summary_ccdoptic)

            if not ccdoptic_fpath.exists() :
                print(f"{str(ccdoptic_fpath)} is not exist...")
                doing_status = True
            else : 
                t = os.path.getmtime(ccdoptic_fpath)
                ccdoptic_fpath_dt = datetime.fromtimestamp(t)
                print("ccdoptic_fpath_dt: ", ccdoptic_fpath_dt)
                doing_status = False
                if ccdoptic_fpath_dt < datetime.now() + timedelta(days=checkTime_ccdoptic) :
                    print(f"{str(ccdoptic_fpath)} is older more than {checkTime_ccdoptic} days ...")
                    doing_status = True

            if doing_status == False :
                summary_ccdoptic = pd.read_csv(str(ccdoptic_fpath))
            
            else : 
                DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(str(CCDOPTICDIR)))
                # remove = 'Cal'
                # CCDOPTICDIRs = [x for x in CCDOPTICDIRs if remove not in x]
                # print ("DOINGDIRs: ", DOINGDIRs)
                # print ("len(DOINGDIRs): ", len(DOINGDIRs))

                for DOINGSUBDIR in DOINGDIRs[:] :
                    DOINGSUBDIR = Path(DOINGSUBDIR)
                    print("DOINGSUBDIR", DOINGSUBDIR)

                    doing_status = False
                    DOINGSUBDIR = Path(DOINGSUBDIR)
                    save_fpath = DOINGSUBDIR.parents[0] / f"summary_{DOINGSUBDIR.parts[-1]}.csv"
                    print (f"Starting...\n{DOINGSUBDIR.name}")

                    if not save_fpath.exists() :
                        print(f"{str(save_fpath)} is not exist...")
                        doing_status = True
                    else :
                        t = os.path.getmtime(save_fpath)
                        save_fpath_dt = datetime.fromtimestamp(t)
                        #print("save_fpath_dt: ", save_fpath_dt)
                        if save_fpath_dt < datetime.now() + timedelta(days=checkTime_eachdir) :
                            print(f"{str(ccd_fpath)} is older more then {checkTime_eachdir} days ...")
                            doing_status = True
                        else:
                            doing_status = False
                    
                    ########################
                    #doing_status = True
                    ########################
                    if doing_status == False :
                        summary = pd.read_csv(str(save_fpath))

                    else : 
                        fits_in_dir = sorted(list(DOINGSUBDIR.glob('*.fit*')))
                        print("fits_in_dir", fits_in_dir)
                        print("len(fits_in_dir)", len(fits_in_dir))
                        if len(fits_in_dir) == 0 :
                            print(f"There is no fits fils in {DOINGSUBDIR}")
                            summary = pd.DataFrame(columns=['file'])
                            pass
                        else : 
                            summary = yfu.make_summary(DOINGSUBDIR/"*.fit*",
                                    output = save_fpath,
                                    verbose = True,
                                    )
                    summary.set_index('file')
                    summary.to_csv(str(save_fpath))
                    print(f"{save_fpath} is created...")
                    summary_ccdoptic = pd.concat([summary_ccdoptic, summary], axis = 0)
            
            summary_ccdoptic.set_index('file')
            summary_ccdoptic.to_csv(str(ccdoptic_fpath))
            print(f"{ccdoptic_fpath} is created...")
    
        summary_ccd = pd.concat([summary_ccd, summary_ccdoptic], axis = 0)
    summary_ccd.set_index('file')
    summary_ccd.to_csv(str(ccd_fpath))
    print(f"{ccd_fpath} is created...(ccd_fpath)")  