# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com

"""
#%%
from glob import glob
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

import astropy.units as u

import _astro_utilities
import _Python_utilities

from astroquery.simbad import Simbad

from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
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
BASEDIR = Path("/mnt/Rdata/OBS_data")  

PROJECDIR = BASEDIR / "01-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "02-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "03-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "04-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

DOINGDIRs = sorted(_Python_utilities.getFullnameListOfsubDirs(TODODIR))
print ("DOINGDIRs: ", format(DOINGDIRs))
print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

try : 
    BDFDIR = [x for x in DOINGDIRs if "CAL-BDF" in str(x)]
    print ("BDFDIR: ", format(BDFDIR))
    MASTERDIR = Path(BDFDIR[0]) / _astro_utilities.master_dir
    if not MASTERDIR.exists():
        os.makedirs("{}".format(str(MASTERDIR)))
        print("{} is created...".format(str(MASTERDIR)))
    print ("MASTERDIR: ", format(MASTERDIR))
except : 
    pass

DOINGDIRs = sorted([x for x in DOINGDIRs if "_LIGHT_" in str(x)])
# print ("DOINGDIRs: ", format(DOINGDIRs))
# print ("len(DOINGDIRs): ", format(len(DOINGDIRs)))

# filter_str = 'BL'
# DOINGDIRs = [x for x in DOINGDIRs if filter_str in str(x)]
# remove = 'BIAS'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'DARK'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
# remove = 'FLAT'
# DOINGDIRs = [x for x in DOINGDIRs if remove not in x]
print ("DOINGDIRs: ", DOINGDIRs)
print ("len(DOINGDIRs): ", len(DOINGDIRs))
#%%
#####################################################################
# Observed location
LOCATION = dict(lon=127.005, lat=37.308889, elevation=101)
Suwon = location = EarthLocation(lon=127.005 * u.deg, 
                                 lat=37.308889 * u.deg, 
                                 height=101 * u.m)
observatory_code = "P64"

# Used for any `astropy.SkyCoord` object:
SKYC_KW = dict(unit=u.deg, frame='icrs')

#######################################################
# Initial guess of FWHM in pixel
FWHM_INIT = 4

# Photometry parameters
R_AP = 1.5*FWHM_INIT # Aperture radius
R_IN = 4*FWHM_INIT   # Inner radius of annulus
R_OUT = 6*FWHM_INIT  # Outer radius of annulus

Mag_Low = 11.5
Mag_High = 15

Mag_target = 12.5
Mag_delta = 2
ERR_Min = 0.5

coord_delta = 0.00001
coord_delta = 0.00003
# coord_delta = 0.00005
# coord_delta = 0.00006
coord_delta = 0.000075
coord_delta = 0.00008
# coord_delta = 0.00009
coord_delta = 0.00010
# coord_delta = 0.0002
# coord_delta = 0.0005
coord_deltas = np.arange(0.00001, 0.00050, 0.00001)
#######################################################

#%%
for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)

    READINGDIR = DOINGDIR / _astro_utilities.reduced_dir
    # READINGDIR = DOINGDIR / _astro_utilities.reduced_nightsky_dir

    # DIFFPRESULTDIR = DOINGDIR / f"{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}"
    DIFFPRESULTDIR = DOINGDIR / f"{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}_fw{FWHM_INIT}"
    LIGHTCUEVEDIR = DOINGDIR / "LightCurve"
    if not LIGHTCUEVEDIR .exists():
        os.makedirs("{}".format(str(LIGHTCUEVEDIR )))
        print("{} is created...".format(str(LIGHTCUEVEDIR )))

    csv_in_dir = sorted(list(DIFFPRESULTDIR.glob('*result_photometry.csv')))
    if len(csv_in_dir) == 0 : 
        print("len(csv_in_dir):", len(csv_in_dir))
    else : 
        df = pd.DataFrame()
        for fpath in csv_in_dir[:]:
            fpath = Path(fpath)
            print(f"starting... {fpath}")
            df_csv = pd.read_csv(fpath)
            df = pd.concat([df, df_csv], axis=0)

        print("len(df):", len(df))
        print("df.columns:", df.columns)

        df['t_middle_dt'] = pd.to_datetime(df['t_middle'])
        df = df.drop(columns=['Unnamed: 0'], axis=0)
        df = df.reset_index(drop=True)
        targ_name = DOINGDIR.parts[-1].split("_")[0]
        targ_name = targ_name.replace("-"," ")
        print("targ_name :", targ_name)

        from astroquery.simbad import Simbad
        result_table = Simbad.query_object(targ_name)
        
        if not result_table :
            print("there is no result...")
        else : 
            # print(result_table.columns)
            print("result_table :", result_table)

            # type(result_table['RA'][0])
            # result_table['RA'][0].split(" ")
            targ_sky = SkyCoord(ra=f"{result_table['RA'][0].split(' ')[0]}h{result_table['RA'][0].split(' ')[1]}m{result_table['RA'][0].split(" ")[2]}s",
                        dec=f"{result_table['DEC'][0].split(' ')[0]}d{result_table['DEC'][0].split(' ')[1]}m{result_table['DEC'][0].split(" ")[2]}s", frame='icrs')
            print("targ_sky :", targ_sky)
            for coord_delta in coord_deltas :
                df_targ = df.loc[(df["RAJ2000"] > targ_sky.ra.value*(1-coord_delta)) \
                                & (df["RAJ2000"] < targ_sky.ra.value*(1+coord_delta)) \
                                & (df["DEJ2000"] > targ_sky.dec.value*(1-coord_delta))\
                                & (df["DEJ2000"] < targ_sky.dec.value*(1+coord_delta))\
                                & (df["merr_ann"] < ERR_Min)]

                if df_targ.empty :
                    print("df_targ is empty")
                else : 
                    df_targ.to_csv(f"{LIGHTCUEVEDIR}/{READINGDIR.parts[-2]}_{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}_fw{FWHM_INIT}_light_curve_{coord_delta:.05f}.csv")
                    print(f"{READINGDIR.parts[-2]}_{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}_fw{FWHM_INIT}_light_curve_{coord_delta:.05f}.csv is created...")

                    ttime = Time(df_targ["t_middle_dt"])

                    fig, axs = plt.subplots(1, 2, figsize=(16, 10), 
                                sharex=False, sharey=False, gridspec_kw=None)

                    chls = ['B', 'V', 'R']
                    for chl in chls :
                        # if f'{chl}_magnitude' in df_targ:    
                        df_targ_chl = df_targ.loc[df_targ["filter"] == chl].copy()
                        if not df_targ_chl.empty :
                            # print(df_targ_chl)
                            # ttime = Time(df_targ_chl["t_middle_dt"])
                            im0 = axs[0].errorbar(Time(df_targ_chl["t_middle_dt"]).mjd, 
                                    df_targ_chl[f'{chl}_magnitude'], yerr=abs(df_targ_chl["merr_ann"]),
                                    marker='x',
                                    ls='none',
                                    #ms=10,
                                    capsize=3,
                                    label=f'{chl}_magnitude')
                            axs[1].errorbar(df_targ_chl["t_middle_dt"], 
                                    df_targ_chl['flux_star'], yerr=abs(df_targ_chl["flux_err"]),
                                    marker='x',
                                    ls='none',
                                    #ms=10,
                                    capsize=3,
                                    label=f'flux_star at {chl}')

                    axs[0].invert_yaxis()

                    axs[0].set(
                        xlabel='Time (MJD)',
                        ylabel="Magnitude",
                    )
                    axs[0].legend()
                    axs[0].grid(linestyle=':')

                    axs[0].set_title(f"light curve of {targ_name}", fontsize=12,)
                    axs[0].annotate(f'Coord: {targ_sky} +-{coord_delta}', fontsize=8,
                                xy=(0, 0), xytext=(-10, -30), va='top', ha='left',
                                xycoords='axes fraction', textcoords='offset points')

                    axs[1].set(
                        xlabel='Time (date)',
                        ylabel="flux",
                    )
                    axs[1].legend()
                    axs[1].grid(linestyle=':')

                    axs[1].set_title(f"light curve of {targ_name}", fontsize=12,)
                    axs[1].annotate(f'Coord: {targ_sky} +-{coord_delta}', fontsize=8,
                                xy=(0, 0), xytext=(-10, -30), va='top', ha='left',
                                xycoords='axes fraction', textcoords='offset points')

                    plt.tight_layout()
                    plt.savefig(f"{LIGHTCUEVEDIR}/{READINGDIR.parts[-2]}_{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}_fw{FWHM_INIT}_light_curve_{coord_delta:.05f}.png")

                    plt.show()
                    # plt.close()
        

# except Exception as err: 
#     print ('Error messgae .......')
#     #_Python_utilities.write_log(err_log_file, err)
# %%
