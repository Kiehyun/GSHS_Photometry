# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

"""
#%%
from glob import glob
from pathlib import Path
import os

import ysfitsutilpy as yfu

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
#######################################################
# read all files in base directory for processing
BASEDIR = Path("/mnt/Rdata/OBS_data") 
DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_obs_raw_dir / "STX-16803_1bin" / "LIGHT_RILA600")
DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_obs_raw_dir )
DOINGDIR = ( BASEDIR/ "asteroid")
                
DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(str(DOINGDIR)))
#print ("DOINGDIRs: ", format(DOINGDIRs))
print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

#%%
#######################################################
for DOINGDIR in DOINGDIRs[:] : 
    DOINGDIR = Path(DOINGDIR)
    print(f"Starting: {str(DOINGDIR.parts[-1])}")
    fits_in_dir = sorted(list(DOINGDIR.glob('*.fit*')))
    #print("fits_in_dir", fits_in_dir)
    print("len(fits_in_dir)", len(fits_in_dir))

    if len(fits_in_dir) == 0 :
        print(f"There is no fits fils in {DOINGDIR}")
        pass
    else : 
        summary = None 
        summary = yfu.make_summary(DOINGDIR/"*.fit*",
                    #output = save_fpath,
                    verbose = False
                    )
        print("summary: ", summary)
        print("len(summary)", len(summary))

        for _, row in summary.iterrows():
            fpath = Path(row["file"])
            print (f"starting {fpath.name}...")
            if fpath.suffix == ".fit" and \
                (fpath.parents[0]/f"{fpath.stem}.fits").exists() : 
                os.remove(fpath)
                print(f"{fpath.name} is removed")