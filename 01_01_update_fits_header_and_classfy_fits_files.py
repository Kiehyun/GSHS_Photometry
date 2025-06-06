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

import ysfitsutilpy as yfu

import _Python_utilities
import _astro_utilities
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
verbose = True # False     
Overwrite = False   
#######################################################
# Set directory variables.
#######################################################
BASEDIR = Path("/mnt/Rdata/ASTRO_data") 
if platform.system() == "Windows":
    BASEDIR = Path("R:\\")  
 
DOINGDIR = BASEDIR/ _astro_utilities.CCD_NEW_dir

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(DOINGDIR))
if verbose == True :
    print ("DOINGDIRs: ", format(DOINGDIRs))
    print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))     
 
#######################################################  
#%%
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True : 
        print("DOINGDIR", DOINGDIR)
        print(f"Starting: {str(DOINGDIR.parts[-1])}")

    try :
        summary = yfu.make_summary(DOINGDIR/"*.fit*",
                                verify_fix=True,
                                ignore_missing_simple=True,
                                verbose = verbose,
                                )
        if summary is None :
            print("summary is None")
            continue
        else : 

            if verbose == True : 
                print("summary: ", summary)
                print("len(summary)", len(summary))

            for _, row in summary.iterrows():
            
                fpath = Path(row["file"])
                if verbose == True : 
                    print (row["file"])   # 파일명 출력
                
                hdul = _astro_utilities.KevinFitsUpdater(fpath,
                                                # imgtype_update=True, # default False
                                                # fil_update=True, # default False
                                                verbose = verbose, 
                                                )
                if verbose == True :
                    print("hdul: ", hdul)
            
    except Exception as err :
        print("X"*60)
        _Python_utilities.write_log(err_log_file, f'{str(err)}', verbose=verbose)
        pass

    _Python_utilities.write_log(log_file, 
                                f"Fits headers are updated in {DOINGDIR}",
                                verbose=verbose)

for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    if verbose == True : 
        print("DOINGDIR", DOINGDIR)
        print(f"Starting: {str(DOINGDIR.parts[-1])}")

    try :
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
                        shutil.move(str(fpath), str(new_fpath))
                        if verbose == True :
                            print(f"move {str(fpath.name)} to {str(new_fpath)}")
                    else :
                        shutil.move(fpath, duplicate_fpath)
                        if verbose == True :
                            print(f'{fpath.parts[-1]} is move to duplicate folder...')
                else : 
                    shutil.move(str(fpath), str(new_fpath))
                    if verbose == True :
                        print(f"move {str(fpath.name)} to {str(new_fpath)}")

    except Exception as err :
        print("X"*60)
        _Python_utilities.write_log(err_log_file, f'{str(err)}', verbose=verbose)
        pass

    _Python_utilities.write_log(log_file, 
                        f"All fits files are moved in {DOINGDIR}",
                        verbose=verbose)

#%%   
#############################################################################
#Check and delete empty folder....
#############################################################################
for i in range(4):
    DOINGDIR = ( BASEDIR/ _astro_utilities.CCD_NEW_dir)         
    DOINGDIRs = sorted(_Python_utilities.getFullnameListOfallsubDirs(DOINGDIR))
    if verbose == True :
        print ("DOINGDIRs: ", format(DOINGDIRs))
        print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

    for DOINGDIR in DOINGDIRs :    
        if len(os.listdir(str(DOINGDIR))) == 0 :
            shutil.rmtree(f"{str(DOINGDIR)}") # Delete..
            if verbose == True :
                print (f"rmtree {str(DOINGDIR)}")
        else : 
            fpaths = _Python_utilities.getFullnameListOfallFiles(str(DOINGDIR))
            if verbose == True :
                print("fpaths", fpaths)

            for fpath in fpaths[:]:
                if verbose == True :
                    print("fpath", fpath)

                if fpath[-4:].lower() in [".txt", "xisf", ".zip", ".png", ".log",
                                            "seal", "tiff", ".axy", "atch", "lved",
                                            "rdls", "xyls", "corr", "xosm", ".ini",
                                            ".wcs", ".csv"] \
                                        and os.path.isfile(fpath):
                    os.remove("{}".format(fpath))
                    if verbose == True :
                        print("{} is removed...".format(fpath)) 