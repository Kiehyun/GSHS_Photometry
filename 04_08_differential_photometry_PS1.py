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
import seaborn as sns
from ccdproc import combine, ccd_process, CCDData

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import astropy.units as u

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _astro_utilities
import _Python_utilities

from astropy.nddata import Cutout2D
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils.centroids import centroid_com

from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn
from photutils.aperture import aperture_photometry as apphot

from astroquery.simbad import Simbad
from urllib.parse import urlencode

from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
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
#######################################################
BASEDIR = Path("/mnt/Rdata/OBS_data")  

PROJECDIR = BASEDIR / "01-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "02-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

PROJECDIR = BASEDIR / "03-EXO"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"

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

# filter_str = 'Kepler-17b_LIGHT_-_2024-06-26_-_RiLA600_STX-16803_-_2bin'
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

### TT-ARI
# Mag_Low = 10
# Mag_High = 15

# Mag_target = 11
# Mag_delta = 2
# ERR_Min = 0.5
#######################################################

#%%
tryagain = False

for DOINGDIR in DOINGDIRs[:] :
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)

    READINGDIR = DOINGDIR / _astro_utilities.reduced_dir
    # READINGDIR = DOINGDIR / _astro_utilities.reduced_nightsky_dir

    DIFFPRESULTDIR = DOINGDIR / f"{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}_fw{FWHM_INIT}"
    if not DIFFPRESULTDIR.exists():
        os.makedirs("{}".format(str(DIFFPRESULTDIR)))
        print("{} is created...".format(str(DIFFPRESULTDIR)))

    summary = yfu.make_summary(READINGDIR/"*.fit*")
    if summary is not None : 
        print("len(summary):", len(summary))
        #print("summary:", summary)
        #print(summary["file"][0])
        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)
        #print("df_light:\n{}".format(df_light))

        for _, row  in df_light.iterrows():
            try:
                fpath = Path(row["file"])

                if (DIFFPRESULTDIR/f"{fpath.stem}_result_photometry.csv").exists() and not tryagain:
                    print("*"*10)
                    print(f"{fpath.stem}_result_photometry.csv is already exist...")
                else :
                    print("*"*20)
                    print(f"Starting {fpath.name}...")
                    hdul = fits.open(fpath)
                    ccd = yfu.load_ccd(fpath)
                    flt = hdul[0].header["filter"]

                    SOLVE, ASTAP, LOCAL = _astro_utilities.checkPSolve(fpath)
                    print(SOLVE, ASTAP, LOCAL)
                    
                    if SOLVE :
                        wcs = WCS(hdul[0].header)
                        # It is used as a rough estimate, so no need to be accurate:
                        #PIX2ARCSEC = 0.62*u.arcsec
                        if 'PIXSCALE' in hdul[0].header:
                            PIX2ARCSEC = hdul[0].header['PIXSCALE']
                        else : 
                            PIX2ARCSEC = _astro_utilities.calPixScale(hdul[0].header['FOCALLEN'], 
                                                            hdul[0].header['XPIXSZ'],
                                                            hdul[0].header['XBINNING'])
                            
                        if hdul[0].header['CCDNAME'] == 'STF-8300M' :
                            val_figsize = (12, 9)
                            val_fraction = 0.035
                            hdul[0].header["GAIN"] = 0.37,
                            hdul[0].header["RDNOISE"] = 9.3

                        if hdul[0].header['CCDNAME'] == 'STX-16803' :
                            val_figsize=(10, 9)
                            val_fraction = 0.0455
                            hdul[0].header["GAIN"] = 1.27
                            hdul[0].header["RDNOISE"] = 9.0    

                        # It is used as a rough estimate, so no need to be accurate:
                        PIX2ARCSEC = hdul[0].header["PIXSCALE"]
                        rdnoise = hdul[0].header["RDNOISE"]
                        gain    = hdul[0].header["GAIN"]
                        print(f"rdnoise : {rdnoise}, gain : {gain}, PIX2ARCSEC : {PIX2ARCSEC}")
                        
                        # D.2. Find the observation time and exposure time to set the obs time
                        t_start = Time(hdul[0].header['DATE-OBS'], format='isot')
                        t_expos = hdul[0].header['EXPTIME'] * u.s
                        t_middle = t_start + t_expos / 2 # start time + 0.5 * exposure time
                        print(f"t_start: {t_start}, t_expos: {t_expos}, t_middle: {t_middle}")
                        
                        # Get the radius of the smallest circle which encloses all the pixels
                        rad = yfu.fov_radius(header=hdul[0].header,
                                            unit=u.deg)
                        print("rad: {}".format(rad))  # 시야각(FOV)으로 구한 반지름

                        cent_coord = yfu.center_radec(ccd_or_header=hdul[0].header, 
                                                            center_of_image=True)
                        print("cent_coord: {}".format(cent_coord))

                        pos_sky = SkyCoord(cent_coord, unit='deg')
                        pos_pix = pos_sky.to_pixel(wcs=wcs)

                        print("pos_sky: {}".format(pos_sky))
                        print("pos_pix: {}".format(pos_pix))

                        #%%
                        ps1 = ypu.PanSTARRS1(cent_coord.ra, cent_coord.dec, radius=rad,
                        column_filters={"rmag":f"{Mag_target-Mag_delta}..{Mag_target+Mag_delta}",
                                                "e_rmag":"<0.10", "nr":">5"}
                                                )
                        PS1_stars_all = ps1.query()
                        print("type(PS1_stars_all) :", type(PS1_stars_all))
                        print("len(PS1_stars_all) :", len(PS1_stars_all))

                        df_stars_all = PS1_stars_all.to_pandas()

                        isnear = ypu.organize_ps1_and_isnear(
                                            ps1, 
                                            # header=ccd.header+ccd.wcs.to_header(), 
                                            ccd.header+ccd.wcs.to_header(), 
                                            # bezel=5*FWHM_INIT*PIX2ARCSEC.value,
                                            # nearby_obj_minsep=5*FWHM_INIT*PIX2ARCSEC.value,
                                            bezel=5*FWHM_INIT*PIX2ARCSEC,
                                            nearby_obj_minsep=5*FWHM_INIT*PIX2ARCSEC,
                                            group_crit_separation=6*FWHM_INIT
                                        )
                        df_stars = ps1.queried.to_pandas()
                        # print("len(df_stars):", len(df_stars))
                        df_stars = df_stars.dropna(subset=["gmag", "rmag"])
                        print("len(df_stars):", len(df_stars))

                        pos_stars_all = np.array([df_stars_all["RAJ2000"].array, df_stars_all["DEJ2000"].array]).T
                        pos_stars_all = SkyCoord(pos_stars_all, **SKYC_KW).to_pixel(wcs)
                        pos_stars_all = np.transpose(pos_stars_all)
                        # pos_stars_all   # PS1 query 모든 별

                        pos_stars = np.array([df_stars["RAJ2000"].array, df_stars["DEJ2000"].array]).T
                        pos_stars = SkyCoord(pos_stars, **SKYC_KW).to_pixel(wcs)
                        pos_stars = np.transpose(pos_stars)
                        # pos_stars     # PS1 query 중 비교 측광에 사용될 별

                        ap_stars = CAp(positions=pos_stars, r=R_IN)
                        ap_stars_all = CAp(positions=pos_stars_all, r=R_IN)
                        #apert
                        an_stars = CAn(positions=pos_stars, r_in=R_IN, r_out=R_OUT)
                        an_stars_all = CAn(positions=pos_stars_all, r_in=R_IN, r_out=R_OUT)
                        
                        #%%
                        fig, axs = plt.subplots(1, 1, figsize=val_figsize,
                        subplot_kw={'projection': wcs},
                        sharex=False, sharey=False, gridspec_kw=None)

                        im = _astro_utilities.zimshow(axs, hdul[0].data, )
                        axs.coords.grid(True, color='white', ls=':')
                        axs.coords['ra'].set_axislabel('Right Ascension (J2000)', minpad=0.5, fontsize=8)
                        axs.coords['ra'].set_ticklabel_position('bl')
                        axs.coords['dec'].set_axislabel('Declination (J2000)', minpad=0.4, fontsize=8)
                        axs.coords['dec'].set_ticklabel_position('bl')
                        axs.coords['ra'].set_major_formatter('hh:mm')
                        axs.coords['dec'].set_major_formatter('dd:mm')
                        axs.coords['ra'].display_minor_ticks(True)
                        axs.coords['dec'].display_minor_ticks(True)
                        axs.coords['ra'].set_minor_frequency(2)
                        axs.coords['dec'].set_minor_frequency(2)
                        axs.tick_params(labelsize=8)

                        for i in range(len(pos_stars)):
                            axs.text(pos_stars[i][0], pos_stars[i][1], f"Star #{str(i)}", fontsize=6, color='w')

                        ap_stars_all.plot(axs, color='w', lw=1)
                        ap_stars.plot(axs, color='r', lw=1)

                        axs.set_title(f"fname: {fpath.name}\n Comparison Stars of PS1 (red tag, Magnitude : {Mag_target}+-{Mag_delta})", fontsize=10,)

                        cbar = plt.colorbar(im, ax = axs, fraction=0.035, pad=0.04, )
                        cbar.ax.tick_params(labelsize=8)

                        axs.annotate(f'Number of star(s): {len(pos_stars)}', fontsize=8,
                            xy=(0, 0), xytext=(-10, -50), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')

                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR/fpath.stem}_PS1_comparison.png")

                        # plt.show()
                        plt.close()

                        #%%
                        fig, axs = plt.subplots(1, 1, figsize=val_figsize,
                                                subplot_kw={'projection': wcs},
                                                sharex=False, sharey=False, gridspec_kw=None)

                        im = _astro_utilities.zimshow(axs, hdul[0].data, )

                        _phot_stars = []

                        for i, row in df_stars.iterrows():
                            pos_star = SkyCoord(row["RAJ2000"], row["DEJ2000"],
                                                **SKYC_KW).to_pixel(wcs)
                            ap = CAp([pos_star[0], pos_star[1]],
                                    r=R_AP)
                            an = CAn([pos_star[0], pos_star[1]],
                                    r_in=R_IN, r_out=R_OUT)
                            _phot_star = ypu.apphot_annulus(hdul[0].data,
                                                            ap, an,
                                                            error=yfu.errormap(hdul[0].data))
                            _phot_star[f"{flt}mag"] = row[f"{flt}mag"]
                            _phot_star[f"e_{flt}mag"] = row[f"e_{flt}mag"]
                            _phot_star["gmag"] = row["gmag"]
                            _phot_star["e_gmag"] = row["e_gmag"]
                            _phot_star["rmag"] = row["rmag"]
                            _phot_star["e_rmag"] = row["e_rmag"]
                            _phot_star["grcolor"] = row["grcolor"]
                            _phot_star["e_grcolor"] = row["e_grcolor"]
                            _phot_star["id"] = i
                            _phot_star["objID"] = int(row["objID"])
                            _phot_stars.append(_phot_star)
                            axs.text(pos_star[0]+10, pos_star[1]+10, f"star {i}:{row[f'{flt}mag']:.01f}",
                                    fontsize=8, color="w")
                            ap.plot(axs, color="orange")
                            # an.plot(axs, color="w")

                        axs.coords.grid(True, color='white', ls=':')
                        axs.coords['ra'].set_axislabel('Right Ascension (J2000)', minpad=0.5, fontsize=8)
                        axs.coords['ra'].set_ticklabel_position('b')
                        axs.coords['dec'].set_axislabel('Declination (J2000)', minpad=0.4, fontsize=8)
                        axs.coords['dec'].set_ticklabel_position('l')
                        axs.coords['ra'].set_major_formatter('hh:mm')
                        axs.coords['dec'].set_major_formatter('dd:mm')

                        axs.coords['ra'].display_minor_ticks(True)
                        axs.coords['dec'].display_minor_ticks(True)
                        axs.coords['ra'].set_minor_frequency(2)
                        axs.coords['dec'].set_minor_frequency(2)
                        axs.tick_params(labelsize=8)

                        cbar = plt.colorbar(im, ax = axs, fraction=0.035, pad=0.04, )

                        axs.set_title(f"fname: {fpath.name}\n {flt} magnitude of PS1 comparison stars (Magnitude : {Mag_target}+-{Mag_delta})", fontsize=10,)
                        axs.annotate(f'Number of star(s): {len(pos_stars)}', fontsize=8,
                                xy=(0, 0), xytext=(-10, -50), va='top', ha='left',
                                xycoords='axes fraction', textcoords='offset points')

                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_PS1_magnitude.png")

                        # plt.show()
                        plt.close()

                        #%%
                        df_phot_stars = pd.concat(_phot_stars)
                        df_phot_stars_na = df_phot_stars.dropna()
                        print(len(df_phot_stars_na))

                        df_phot_stars_na = df_phot_stars[df_phot_stars["merr"] < ERR_Min]
                        # phot_stars_na = phot_stars_na.set_index('id', drop=True)
                        df_phot_stars_na = df_phot_stars_na.reset_index(drop=True)
                        print(len(df_phot_stars_na))
                        # print(df_phot_stars_na)
                        df_phot_stars_na

                        #%%
                        merr_total1 = np.sqrt((df_phot_stars_na["merr"])**2 + (df_phot_stars_na[f"e_{flt}mag"])**2)

                        # === Calculate zero point and errors
                        _xx = np.linspace(Mag_Low, Mag_High)
                        zeropt_med = np.median(df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"])
                        zeropt_avg = np.average(df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                                weights=1/merr_total1**2)
                        dzeropt = np.max([1/np.sqrt(np.sum(1/(merr_total1)**2)),
                                        np.std((df_phot_stars_na[f"e_{flt}mag"] - df_phot_stars_na["merr"]), ddof=1)/np.sqrt(len(df_phot_stars_na[f"{flt}mag"]))])
                        merr_total2 = np.sqrt(np.sqrt(merr_total1**2 + dzeropt**2))

                        # === Find fitting lines
                        # Search for the usage of scipy.optimize.curve_fit.
                        poptm, _ = curve_fit(_astro_utilities.linf, df_phot_stars_na[f"{flt}mag"],
                                            df_phot_stars_na["mag"],
                                            sigma= df_phot_stars_na["merr"], absolute_sigma=True)
                        poptc, _ = curve_fit(_astro_utilities.linf, df_phot_stars_na["grcolor"],
                                            df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                            sigma=merr_total2, absolute_sigma=True)

                        #%%
                        fig, axs = plt.subplots(2, 3, figsize=(15, 6), sharex=False, sharey=False,
                                        gridspec_kw={'height_ratios': [1, 3]})
                        
                        errkw = dict(marker="", ls="", ecolor="gray", elinewidth=0.5)

                        def plot_common(ax, x, y, xerr, yerr, title="", xlabel="", ylabel="", ylim=None):
                            ax.plot(x, y, '+')
                            ax.errorbar(x, y, xerr=xerr, yerr=yerr, **errkw)
                            ax.axhline(zeropt_med, color="r", lw=1, label=f"$Z = {{{zeropt_med:.3f}}} ± {{{dzeropt:.3f}}}$\n(median value)")
                            ax.axhline(zeropt_avg, color="b", lw=1, label=f"$Z = {{{zeropt_avg:.3f}}} ± {{{dzeropt:.3f}}}$\n(average value)")
                            ax.hlines([zeropt_med + dzeropt, zeropt_med - dzeropt, zeropt_avg + dzeropt, zeropt_avg - dzeropt],
                                    *ax.get_xlim(), color=["r","r","b","b"], lw=1, ls=":")
                            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
                            # ax.legend(fontsize=8, loc='best')

                        # 상단 행
                        plot_common(axs[0, 0], df_phot_stars_na[f"{flt}mag"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                    df_phot_stars_na[f"e_{flt}mag"], df_phot_stars_na["merr"],
                                    ylabel=f"${{{flt}}}_{{inst}} - {{{flt}}}_{{PS1}}$",
                                    ylim=(zeropt_med-0.8, zeropt_med+0.8))

                        plot_common(axs[0, 1], df_phot_stars_na["grcolor"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                    df_phot_stars_na[f"e_grcolor"], merr_total2,
                                    title=f"${{{flt}}}_{{inst}} - {{{flt}}}_{{PS1}} = (z + k'X) + (k''X + k)C$",
                                    ylabel=f"${{{flt}}}_{{inst}} - {{{flt}}}_{{PS1}}$",
                                    ylim=(zeropt_med-0.8, zeropt_med+0.8))
                        axs[0, 1].plot(axs[0, 1].get_xlim(), _astro_utilities.linf(np.array(axs[0, 1].get_xlim()), *poptc),
                                    "g-", lw=1, label=f"$y = {{{poptc[1]:+.3f}}}x {{{poptc[0]:+.3f}}}$\n(curve_fit)")
                        # axs[0, 1].legend(fontsize=8, loc='best')

                        data = df_phot_stars_na[["mag", "merr", "grcolor", "e_grcolor"]]
                        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=axs[0, 2])
                        axs[0, 2].set(title='Correlation Heatmap')

                        # 하단 행
                        plot_common(axs[1, 0], df_phot_stars_na[f"{flt}mag"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                    df_phot_stars_na[f"e_{flt}mag"], df_phot_stars_na["merr"],
                                    xlabel=f"${{{flt}}}_{{PS1}}$ (PS1 to {flt} filter by Tonry+2012)",
                                    ylabel=f"${{{flt}}}_{{inst}} - {{{flt}}}_{{PS1}}$")

                        plot_common(axs[1, 1], df_phot_stars_na["grcolor"], df_phot_stars_na["mag"] - df_phot_stars_na[f"{flt}mag"],
                                    df_phot_stars_na[f"e_grcolor"], merr_total2,
                                    xlabel="$g - r$ (PS1)",
                                    ylabel=f"${{{flt}}}_{{inst}} - {{{flt}}}_{{PS1}}$")
                        axs[1, 0].legend(fontsize=8, loc='best')

                        axs[1, 1].plot(axs[1, 1].get_xlim(), _astro_utilities.linf(np.array(axs[1, 1].get_xlim()), *poptc),
                                    "g-", lw=1, label=f"$y = {{{poptc[1]:+.3f}}}x {{{poptc[0]:+.3f}}}$\n(curve_fit)")
                        axs[1, 1].legend(fontsize=8, loc='best')

                        axs[1, 2].plot(_xx, _xx + zeropt_med,
                                    label=f"${{{flt}}}_{{inst}} = {{{flt}}}_{{PS1}}$ {zeropt_med:+.03f}\n(median vlaue)",
                                    color="r", lw=1, ls="-")
                        axs[1, 2].plot(axs[1, 2].get_xlim(), _astro_utilities.linf(np.array(axs[1, 2].get_xlim()), *poptm),
                                    "g-", lw=1, label=f"$y = {{{poptm[1]:.3f}}}x {{{poptm[0]:+.3f}}}$\n(curve_fit)")
                        axs[1, 2].plot(df_phot_stars_na[f"{flt}mag"], df_phot_stars_na["mag"], '+')
                        axs[1, 2].errorbar(df_phot_stars_na[f"{flt}mag"],
                                    df_phot_stars_na["mag"],
                                    xerr=df_phot_stars_na[f"e_{flt}mag"],
                                    yerr=df_phot_stars_na["merr"],
                                    **errkw)
                        axs[1, 2].set(
                                    xlabel=f"${{{flt}}}_{{PS1}}$ (PS1 to {flt} filter by Tonry+2012)",
                                    ylabel =f"${{{flt}}}_{{inst}}$",
                                )
                        axs[1, 2].legend(fontsize=8, loc='best')
                        axs[1, 2].axis('square')

                        # ID 텍스트 추가
                        for _, row in df_phot_stars_na.iterrows():
                            for i in range(2):
                                for j in range(2):
                                    axs[i, j].text(row[f"{flt}mag" if j == 0 else "grcolor"],
                                                row["mag"] - row[f"{flt}mag"], int(row["id"]), fontsize=8, clip_on=True)
                            axs[1, 2].text(row[f"{flt}mag"], row["mag"], int(row["id"]), fontsize=8, clip_on=True)

                        # x축 레이블 숨기기 (상단 행)
                        for ax in axs[0, :2]:
                            ax.tick_params(labelbottom=False)

                        plt.suptitle(f"fname: {fpath.name}\n PS1 check for differential photometry (Magnitude : {Mag_target}+-{Mag_delta})", fontsize=10)
                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_standardization_extended.png")

                        # plt.show()
                        plt.close()


                        #%%
                        FWHM = FWHM_INIT
                        avg, med, std = sigma_clipped_stats(hdul[0].data)  # by default, 3-sigma 5-iteration.
                        thresh = 5. * std

                        DAOfind = DAOStarFinder(
                                                fwhm = FWHM,
                                                threshold=thresh,   # In reality, FWHM must be measured a priori using, e.g., ``ginga``
                                                # sharplo=0.2, sharphi=1.0,   # default values 0.2 and 1.0
                                                # roundlo=-1.0, roundhi=1.0,  # default values -1 and +1
                                                # sigma_radius=1.5,           # default values 1.5
                                                # ratio=1.0,                  # 1.0: circular gaussian
                                                exclude_border=True         # To exclude sources near edges
                                                )

                        DAOfound = DAOfind(hdul[0].data)
                        if len(DAOfound) > 2000 :
                            from photutils import detect_threshold
                            thresh_snr = detect_threshold(data=hdul[0].data, nsigma=3,)
                            print('type(thresh_snr) :', type(thresh_snr))
                            print('thresh_snr.shape :', thresh_snr.shape)
                            print('detect_threshold', thresh_snr)
                            thresh = thresh_snr[0][0]

                            DAOfind = DAOStarFinder(
                                                fwhm = FWHM,
                                                threshold=thresh,   # In reality, FWHM must be measured a priori using, e.g., ``ginga``
                                                # sharplo=0.2, sharphi=1.0,   # default values 0.2 and 1.0
                                                # roundlo=-1.0, roundhi=1.0,  # default values -1 and +1
                                                # sigma_radius=1.5,           # default values 1.5
                                                # ratio=1.0,                  # 1.0: circular gaussian
                                                exclude_border=True         # To exclude sources near edges
                                                )
                            DAOfound = DAOfind(hdul[0].data)

                        print("len(DAOfound) :",len(DAOfound))
                        print(DAOfound.colnames)

                        # DAOfound.write(f"{DIFFPRESULTDIR/fpath.stem}_DAOStarfinder_fwhm_{FWHM}.csv",
                        #                             overwrite = True,
                        #                             format='ascii.fast_csv')
                        df_DAO = DAOfound.to_pandas()
                        print(type(df_DAO))
                        df_DAO

                        pos = np.transpose((DAOfound['xcentroid'], DAOfound['ycentroid']))
                        apert = CAp(pos, r=R_AP)
                        annul = CAn(positions=pos, r_in= R_IN, r_out=R_OUT)
                        #%%
                        fig, axs = plt.subplots(1, 1, figsize=val_figsize,
                        subplot_kw={'projection': wcs},
                        sharex=False, sharey=False, gridspec_kw=None)

                        im = _astro_utilities.zimshow(axs, hdul[0].data, )
                        axs.set_title('World coordinate system', fontsize=9)
                        axs.coords.grid(True, color='white', ls=':')
                        axs.coords['ra'].set_axislabel('Right Ascension (J2000)', minpad=0.5, fontsize=8)
                        axs.coords['ra'].set_ticklabel_position('bl')
                        axs.coords['dec'].set_axislabel('Declination (J2000)', minpad=0.4, fontsize=8)
                        axs.coords['dec'].set_ticklabel_position('bl')
                        axs.coords['ra'].set_major_formatter('hh:mm')
                        axs.coords['dec'].set_major_formatter('dd:mm')
                        axs.coords['ra'].display_minor_ticks(True)
                        axs.coords['dec'].display_minor_ticks(True)
                        axs.coords['ra'].set_minor_frequency(2)
                        axs.coords['dec'].set_minor_frequency(2)
                        axs.tick_params(labelsize=8)

                        annul.plot(axs, color="r")
                        for i in range(len(pos)):
                            axs.text(pos[i][0], pos[i][1], f"Star #{str(i)}", fontsize=6, color='w')

                        annul.plot(axs, color="r")

                        cbar = plt.colorbar(im, ax = axs, fraction=0.035, pad=0.04, )
                        cbar.ax.tick_params(labelsize=8)

                        axs.set_title(f"fname: {fpath.name}\n Result of DAOFinder", fontsize=10,)

                        axs.annotate(f'FWHM: {FWHM}', fontsize=8,
                            xy=(0, 0), xytext=(-10, -30), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')

                        axs.annotate(f'Sky threshold: {thresh:.02f}', fontsize=8,
                            xy=(0, 0), xytext=(-10, -40), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')

                        axs.annotate(f'Number of star(s): {len(DAOfound)}', fontsize=8,
                            xy=(0, 0), xytext=(-10, -50), va='top', ha='left',
                            xycoords='axes fraction', textcoords='offset points')

                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_DAOStarfinder_fwhm_{FWHM}.png")

                        # plt.show()
                        plt.close()

                        #%%
                        apphot_result = apphot(hdul[0].data, apert, method='center')
                        print(type(apphot_result))
                        # df_apphot = pd.DataFrame()
                        # apphot_result
                        df_apphot = apphot_result.to_pandas()
                        print(type(df_apphot))
                        df_apphot

                        ap_area  = apert.area
                        ap_area

                        # since our `annul` has many elements,
                        mask_apert = (apert.to_mask(method='center'))
                        mask_annul = (annul.to_mask(method='center'))

                        mag_ann  = np.zeros(len(apphot_result))
                        merr_ann = np.zeros(len(apphot_result))

                        #%%
                        for i in range(len(apphot_result)):
                            annul_weighted = mask_annul[i].multiply(hdul[0].data)
                            sky_non0   = np.nonzero(annul_weighted)
                            sky_pixel  = annul_weighted[sky_non0]

                            msky, sky_std, nsky, nrej = _astro_utilities.sky_fit(sky_pixel, method='mode',
                                                                                mode_option='sex')


                            flux_star = apphot_result['aperture_sum'][i] - msky * ap_area  # total - sky

                            flux_err  = np.sqrt(apphot_result['aperture_sum'][i] * gain    # Poissonian (star + sky)
                                                + ap_area * rdnoise**2 # Gaussian
                                                + (ap_area * (gain * sky_std))**2 / nsky )

                            mag_ann[i], merr_ann[i] = _astro_utilities.mag_inst(flux_star, flux_err)
                            df_apphot.at[i, 'msky'] = msky
                            df_apphot.at[i, 'sky_std'] = sky_std
                            df_apphot.at[i, 'nsky'] = nsky
                            df_apphot.at[i, 'nrej'] = nrej
                            df_apphot.at[i, 'flux_star'] = flux_star
                            df_apphot.at[i, 'flux_err'] = flux_err
                            df_apphot.at[i, 'mag_ann'] = mag_ann[i]
                            df_apphot.at[i, 'merr_ann'] = merr_ann[i]

                        df_apphot['filename'] = fpath.stem
                        df_apphot['t_start'] = t_start
                        df_apphot['t_expos'] = t_expos
                        df_apphot['t_middle'] = t_middle
                        df_apphot['filter'] = flt
                        df_apphot["zeropt_med"] = zeropt_med
                        df_apphot["zeropt_avg"] = zeropt_avg
                        df_apphot["e_zeropt"] = dzeropt

                        df_apphot[f"{flt}_magnitude"] = df_apphot["mag_ann"] - df_apphot["zeropt_med"]

                        df_apphot['filename'] = fpath.stem
                        df_apphot['t_start'] = t_start
                        df_apphot['t_expos'] = t_expos
                        df_apphot['t_middle'] = t_middle
                        df_apphot['filter'] = flt
                        df_apphot["zeropt_med"] = zeropt_med
                        df_apphot["zeropt_avg"] = zeropt_avg
                        df_apphot["e_zeropt"] = dzeropt

                        df_apphot[f"{flt}_magnitude"] = df_apphot["mag_ann"] - df_apphot["zeropt_med"]

                        sky_coord = wcs.pixel_to_world(df_apphot['xcenter'], df_apphot['ycenter'])
                        sky_coord
                        print(type(sky_coord))

                        # df_apphot["RA2000"] = sky_coord.ra
                        # df_apphot["RA2000"]
                        df_RADEC = pd.DataFrame({"RAJ2000": sky_coord.ra.degree, "DEJ2000": sky_coord.dec.degree})
                        # df_RADEC
                        #type(df_RADEC["RA2000"][0])
                        df_apphot = pd.concat([df_apphot, df_RADEC], axis=1,)

                        df_apphot.to_csv(f"{DIFFPRESULTDIR}/{fpath.stem}_result_photometry.csv")
                        
                        df_apphot_sub = df_apphot.dropna()
                        print(len(df_apphot_sub))
                        df_apphot_sub = df_apphot_sub.loc[(df_apphot_sub["merr_ann"] < ERR_Min)]
                        df_apphot_sub

                        #%%
                        fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                                                sharex=False, sharey=False, gridspec_kw=None)

                        for idx, row in df_apphot_sub.iterrows():
                            im0 = axs[0, 0].errorbar(df_apphot_sub["id"],
                                        df_apphot_sub[f"{flt}_magnitude"], yerr=df_apphot_sub["merr_ann"],
                                        marker='x',
                                        ls='none',
                                        #ms=10,
                                        capsize=3)

                        axs[0, 0].invert_yaxis()
                        axs[0, 0].set(
                            xlabel='Star ID',
                            ylabel=f"${{{flt}}}_{{obs}}$"
                            )

                        style = {'edgecolor': 'white', 'linewidth': 3}
                        im1 = axs[0, 1].hist(df_apphot_sub[f"{flt}_magnitude"],
                                    **style)
                        axs[0, 1].set(
                            xlabel=f"${{{flt}}}_{{obs}}$",
                            ylabel="number of stars"
                            )

                        # 상관관계 계산
                        data =  df_apphot_sub[[f"{flt}_magnitude", "merr_ann"]]
                        corr = data.corr()

                        # 히트맵 그리기
                        im2 = sns.heatmap(corr, annot=True, cmap='coolwarm',
                                            vmin=-1, vmax=1, center=0, ax = axs[1, 0])
                        axs[1, 0].set(
                            title = 'Correlation Heatmap',
                            )

                        axs[1, 1].scatter(df_apphot_sub[f"{flt}_magnitude"], df_apphot_sub["merr_ann"], marker='x',)
                        axs[1, 1].errorbar(x=df_apphot_sub[f"{flt}_magnitude"], y=df_apphot_sub["merr_ann"],
                                    yerr=None, xerr=df_apphot_sub["merr_ann"], fmt="o", color="gray", capsize=3, alpha=0.5)
                        axs[1, 1].set(
                            title = "Correlation between Magnitude and Error",
                            xlabel=f"${{{flt}}}_{{obs}}$",
                            ylabel="Error",
                            )

                        plt.suptitle(f"fname: {fpath.name}\n Result of differential photometry (Magnitude : {Mag_target}+-{Mag_delta})", fontsize=10,)

                        plt.tight_layout()
                        plt.savefig(f"{DIFFPRESULTDIR}/{fpath.stem}_Result_of_differential_photometry.png")

                        # plt.show()
                        plt.close()

            except Exception as err: 
                print ('Error messgae .......')
                #_Python_utilities.write_log(err_log_file, err)