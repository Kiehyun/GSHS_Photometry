# -*- coding: utf-8 -*-
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
from astropy.io import fits
import subprocess
from datetime import datetime, timedelta

import ysfitsutilpy as yfu

import _astro_utilities
import _Python_utilities

import warnings
warnings.filterwarnings('ignore')

#%%
#######################################################
# for log file
log_dir = "logs/"
log_file = "{}{}.log".format(log_dir, os.path.basename(__file__)[:-5])
err_log_file = "{}{}_err.log".format(log_dir, os.path.basename(__file__)[:-3])
print ("log_file: {}".format(log_file))
print ("err_log_file: {}".format(err_log_file))
if not os.path.exists('{0}'.format(log_dir)):
    os.makedirs('{0}'.format(log_dir))
#######################################################
#%%
#######################################################
from multiprocessing import Process, Queue
class Multiprocessor():
    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def restart(self):
        self.processes = []
        self.queue = Queue()

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets
########################################################

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

PROJECDIR = BASEDIR / "A3_CCD_obs_raw"

PROJECDIRs = [ 
                # "STX-16803_1bin", 
                # "STX-16803_2bin",  
                # "STL-11000M_2bin", 
                # "STF-8300M_2bin",  
                # "QSI683ws_2bin", 
                # "STL-11000M_1bin",
                "STF-8300M_1bin",
                # "QSI683ws_1bin",
                # "ASI2600MC_1bin",
                "ASI6200MMPro_3bin",
                ]
DOINGDIRs = []
for DOINGDIR in PROJECDIRs:
    TODODIR = PROJECDIR / DOINGDIR              
    DOINGDIRs.extend(sorted(_Python_utilities.getFullnameListOfallsubDirs(str(TODODIR))))
if verbose == True :
    print ("DOINGDIRs: ", format(DOINGDIRs))
    print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

filter_strs = [
                '2025-0',
                # '2025-02',
                # '2025-03',
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                ]  # Example list of filter strings
DOINGDIRs = [x for x in DOINGDIRs if any(filter_str in str(x) for filter_str in filter_strs)]

rm_filter_strs = [
                # 'GPX-1b',
                # 'HAT',
                # 'WASP',
                'DARK',
                'FLAT',
                'BIAS',
                'Bad_fits', 
                # 'Another_bad_string',
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
        print("DOINGDIR", DOINGDIR)

    if _Python_utilities.check_string_in_file(log_file, DOINGDIR.name):
        print(f"'{DOINGDIR.name}' found in '{log_file}'")
        pass
    
    else:
        SOLVINGDIR = DOINGDIR

        summary = yfu.make_summary(SOLVINGDIR/"*.fit*")
        if summary is not None :
            if verbose == True :
                print("len(summary):", len(summary))
                print("summary:", summary)
                #print(summary["file"][0])  
            df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
            df_light = df_light.reset_index(drop=True)
            if verbose == True :
                print("df_light:\n{}".format(df_light))

        # for _, row  in df_light.iterrows():

        myMP = Multiprocessor()
        num_cpu = 14
        values = []
        fullnames = df_light["file"].tolist()
        num_batches = len(fullnames) // num_cpu + 1

        for batch in range(num_batches):
            myMP.restart()
            for fullname in fullnames[batch*num_batches:(batch+1)*num_batches]:
                fpath = Path(fullname)
                if verbose == True :
                    print("fpath :" ,fpath)
                try :
                    myMP.run(_astro_utilities.KevinSolver, 
                                                fpath, 
                                                #str(SOLVEDDIR), 
                                                # downsample = 2,   #default is 4
                                                # pixscale = PIXc,
                                                cpulimit = 15,  #default is 30
                                                # nside = 20,  #default is 15
                                                # tryASTAP = False,   #default is True  
                                                # tryLOCAL = False,   #default is True
                                                # tryASTROMETRYNET = True,  #default is False
                                                # makeLOCALsh = True, #default is False
                                                verbose = verbose,
                                                )
                except Exception as err :
                    print("X"*60)
                    _Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                    pass
            if verbose == True :
                print(("+"*10 + "\n")*2)
                print(f"Start batch {str(batch)}/{num_batches} {str(DOINGDIR.stem)}")

            values.append(myMP.wait())
            if verbose == True :
                print(("+"*20 + "\n")*3)
                print(f"OK batch {str(batch)}/{num_batches} {str(DOINGDIR.stem)}")

        _Python_utilities.write_log(log_file, 
                            f"{str(DOINGDIR)} is finighed..", 
                            verbose=verbose)