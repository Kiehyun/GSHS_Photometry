"""
# -*- coding: utf-8 -*-

Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

이 파일은 BASEDIR 폴더 안에 있는 모든 fit 파일에 대해서 
fits header에 있는 정보를 바탕으로  
destination_BASEDIR_name 안에 규칙적으로 폴더를 만들어서 저장합니다. 

"""
#%%
from glob import glob
from pathlib import Path, PosixPath, WindowsPath
import os
from astropy.time import Time
from datetime import datetime, timedelta
from astropy.io import fits
import shutil 

import ysfitsutilpy as yfu
import ysphotutilpy as ypu
#import ysvisutilpy as yvu

import _astro_utilities
import _Python_utilities

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
BASEDIR = Path(r"r:\CCD_obs") 
BASEDIR = Path("/mnt/Rdata/OBS_data") 
#BASEDIR = Path("/Volumes/OBS_data")

DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_NEW_dir)
DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_NEWUP_dir)
                
DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(str(DOINGDIR)))
#print ("DOINGDIRs: ", format(DOINGDIRs))
print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

fnameKEYs = ["OBJECT", "IMAGETYP", "FILTER", "DATE-OBS", 
            "EXPOSURE", "OPTIC", "CCDNAME", "CCD-TEMP", "XBINNING"]

#%%
for DOINGDIR in DOINGDIRs[:] : 
    DOINGDIR = Path(DOINGDIR)
    print(f"Starting: {str(DOINGDIR.parts[-1])}")
    try: 
        summary = yfu.make_summary(DOINGDIR/"*.fit*",)
        if summary is not None : 
            print("summary: ", summary)
            print("len(summary)", len(summary))
        
            for _, row in summary.iterrows():
                fpath = Path(row["file"])
                print (f"starting {fpath.name}...")
                new_fname = ""
                suffix = ".fit"
                try:
                    for KEY in fnameKEYs :
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
                
                    print(new_fname)
                        
                    new_folder = _astro_utilities.get_new_foldername_from_filename(new_fname)
                    #print("new_folder: ", new_folder)
                    new_fpath =  BASEDIR /_astro_utilities.CCD_obs_raw_dir / new_folder / new_fname
                    print("new_fpath: ", new_fpath)

                    if not new_fpath.parents[0].exists():
                        os.makedirs('{0}'.format(str(new_fpath.parents[0])))
                        print('{0} is created'.format(str(new_fpath.parts[-2])))  
                
                    if new_fpath.exists() :
                        duplicate_fpath = BASEDIR / _astro_utilities.CCD_duplicate_dir / new_fpath.name
                        #os.rename(str(new_fpath), str(duplicate_fpath))
                        shutil.move(str(new_fpath), str(duplicate_fpath))
                        print (f"move duplicate file to {str(duplicate_fpath)}")
                    shutil.move(str(fpath), str(new_fpath))
                    print(f"move {str(fpath.name)} to {str(new_fpath)}")


                    # new_fpath =  BASEDIR /_astro_utilities.CCD_obs_raw_dir / new_folder / new_fname
                    # new_fpath_fit =  BASEDIR /_astro_utilities.CCD_obs_raw_dir / new_folder / f'{new_fpath.stem}.fit'
                    # new_fpath_fits =  BASEDIR /_astro_utilities.CCD_obs_raw_dir / new_folder / f'{new_fpath.stem}.fits'
                    # print("new_fpath: ", new_fpath)

                    # if not new_fpath_fit.parents[0].exists():
                    #     os.makedirs('{0}'.format(str(new_fpath_fit.parents[0])))
                    #     print('{0} is created'.format(str(new_fpath_fit.parts[-2])))  
                
                    # if new_fpath_fit.exists() and new_fpath.suffix == '.fits':
                    #     duplicate_fpath = BASEDIR / _astro_utilities.CCD_duplicate_dir / new_fpath_fit.name
                    #     #os.rename(str(new_fpath), str(duplicate_fpath))
                    #     shutil.move(str(new_fpath_fit), str(duplicate_fpath))
                    #     print (f"move duplicate file to {str(duplicate_fpath)}")
                    # if new_fpath_fits.exists() and new_fpath.suffix == '.fits':
                    #     duplicate_fpath = BASEDIR / _astro_utilities.CCD_duplicate_dir / new_fpath.name
                    #     #os.rename(str(new_fpath), str(duplicate_fpath))
                    #     shutil.move(str(fpath), str(duplicate_fpath))
                    #     print (f"move duplicate file to {str(duplicate_fpath)}")
                    # else : 
                    #     #os.rename(str(fpath), str(new_fpath))
                    #     shutil.move(str(fpath), str(new_fpath))
                    #     print(f"move {str(fpath.name)} to {str(new_fpath)}")
                    # shutil.move(str(fpath), str(new_fpath))
                    # print(f"move {str(fpath.name)} to {str(new_fpath)}")
                    
                except Exception as err:
                    print("X"*30, f'\n{err}')
                    #_Python_utilities.write_log(err_log_file, err)
                    pass
    except Exception as err:
        print("X"*30, f'\n{err}')
        #_Python_utilities.write_log(err_log_file, err)
        pass
#%%   
#############################################################################
#Check and delete empty folder....
#############################################################################
for i in range(4):
    DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_NEWUP_dir)         
    DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(DOINGDIR))
    #print ("DOINGDIRs: ", format(DOINGDIRs))
    print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

    for DOINGDIR in DOINGDIRs :    
        if len(os.listdir(str(DOINGDIR))) == 0 :
            shutil.rmtree(f"{str(DOINGDIR)}") # Delete..
            print (f"rmtree {str(DOINGDIR)}")
        else : 
            fpaths = _Python_utilities.getFullnameListOfallFiles(str(DOINGDIR))
            print("fpaths", fpaths)

            for fpath in fpaths[:]:
                print("fpath", fpath)

                if fpath[-4:].lower() in [".txt", "xisf", ".zip", ".png", ".log",
                                            "seal", "tiff", ".axy", "atch", "lved",
                                            "rdls", "xyls", "corr", "xosm", ".ini",
                                            ".wcs", ".csv"] \
                                        and os.path.isfile(fpath):
                    os.remove("{}".format(fpath))
                    print("{} is removed...".format(fpath)) 