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
from pathlib import Path
import os
import numpy as np
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import combine, ccd_process, CCDData
import _Python_utilities
import ysfitsutilpy as yfu
import ysphotutilpy as ypu
# import ysvisutilpy as yvu

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

c_method = 'median'
master_dir = "master_files_ys"
reduce_dir = "reduced_ys"

BASEDIR = "../RnE_2022/KLEOPATRA_Light_-_2022-11-04_-_RiLA600_STX-16803_-_2bin/"
#BASEDIR = "../RnE_2022/"
#BASEDIR = "../CCD_obs_raw/"
#%%
# BASEDIRs = sorted(_Python_utilities.getFullnameListOfsubDir(BASEDIR))
# print ("BASEDIRs1: {}".format(BASEDIRs))
# BASEDIRs = [w for w in BASEDIRs \
#         if not (w.endswith("{}/".format(master_dir)) \
#         or w.endswith("{}/".format(reduce_dir))
#         or w.endswith("fits"))]
# print ("BASEDIRs2: {}".format(BASEDIRs))

#%%
# for BASEDIR in BASEDIRs :
#     print ("Starting...\n{}".format(BASEDIR))
#     ######################################################

#     try : 
#%%
TOPDIR = Path(BASEDIR)
summary = yfu.make_summary("{}/*.fit".format(BASEDIR))
print("summary:\n {}".format(summary))

ARCHIVE = TOPDIR/"archive"
BIASDIR = TOPDIR/"BIAS"
DARKDIR = TOPDIR/"DARK"
FLATDIR = TOPDIR/"FALT"
MASTERDIR = TOPDIR/master_dir

        #yfu.fits_newpath()


    # except Exception as err :
    #     print("X"*60)
    #     print('{0}'.format(err))

    # try: 
#%%
bias_fits = summary[summary["IMAGETYP"] == "BIAS"]["file"]
print("bias_fits\n", bias_fits)

#%%
bias_comb = yfu.group_combine(
                bias_fits,
                type_key = ["IMAGETYP"],
                type_val = ["BIAS"],
                group_key = ["EXPTIME"],
                fmt = "master_bias.fits",  # output file name format
                #outdir = "{}{}".format(BASEDIR, master_dir),  # output directory (will automatically be made if not exist)
                outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                verbose=True
            )
#%%
for fpath in bias_fits:
    bias_move = yfu.fits_newpath(
                #bias_fits.tolist(),
                bias_fits,
                rename_by = ["IMAGETYP", "OBSDATE"],
                #mkdir_by = BIASDIR,
                #header=None,
                delimiter='_',
                fillnan="",
                fileext='.fits'
                )

#%%
for fpath in bias_fits:
    print("fpath:", fpath)
    newpath = yfu.fitsrenamer(
                        bias_fits,
                        rename_by = ["IMAGETYP", "OBSDATE"],
                        delimiter='_',
                        fillnan="",
                        #header = hdr,
                        #add_header = add_hdr,
                        mkdir_by = ["IMAGETYP"],
                        #archive_dir = BIASDIR,
                        key_deprecation = True,
                        #keymap = KEYMAP,
                        verbose = True
                        )
#%%
    """
    except Exception as err :
        print("X"*60)
        print('{0}'.format(err))

    try: 
        dark_fits = summary[summary["IMAGETYP"] == "DARK"]["file"]
        # Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
        dark_comb = yfu.group_combine(
                        dark_fits.tolist(),
                        type_key = ["IMAGETYP"],
                        type_val = ["DARK"],
                        group_key = ["EXPTIME"],
                        fmt = "master_dark_{:.0f}sec.fits",  # output file name format
                        outdir = "{}{}".format(BASEDIR, master_dir)  # output directory (will automatically be made if not exist)
                    )
    except Exception as err :
        print("X"*60)
        print('{0}'.format(err))


    try:
        flat_fits = summary[summary["IMAGETYP"] == "FLAT"]["file"] 
        # Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
        flat_comb_norm = yfu.group_combine(
                        flat_fits.tolist(),
                        type_key = ["IMAGETYP"],
                        type_val = ["FLAT"],
                        group_key = ["FILTER"],
                        fmt = "master_flat_{:s}_norm.fits",  # output file name format
                        scale="med_sc", #norm
                        scale_to_0th=False, #norm
                        outdir = "{}{}".format(BASEDIR, master_dir)  # output directory (will automatically be made if not exist)
                    )

        # Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
        flat_comb_norm = yfu.group_combine(
                        flat_fits.tolist(),
                        type_key = ["IMAGETYP"],
                        type_val = ["FLAT"],
                        group_key = ["FILTER"],
                        fmt = "master_flat_{:s}.fits",  # output file name format
                        #scale="med_sc", #norm
                        #scale_to_0th=False, #norm
                        outdir = "{}{}".format(BASEDIR, master_dir)  # output directory (will automatically be made if not exist)
                    )
    except Exception as err :
        print("X"*60)
        print('{0}'.format(err))
"""
# %%
