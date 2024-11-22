# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: user

"""
#%%
from pathlib import Path
from astropy.io import fits
import subprocess
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ccdproc import combine
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroquery.astrometry_net import AstrometryNet

from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils.centroids import centroid_com

from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn

from astroquery.jplhorizons import Horizons
from astroquery.imcce import Skybot

import ysfitsutilpy as yfu
import ysphotutilpy as ypu

import _Python_utilities

#%%
#########################################
#directory variables
#########################################

c_method = "median"

CCD_obs_raw_dir = "CCD_obs_raw"
CCD_NEW_dir = "CCD_new_files"
CCD_NEWUP_dir = "CCD_newUpdated_files"
CCD_duplicate_dir = "CCD_duplicate_files"

master_dir = "master_files_ys"
reduced_dir = "reduced"
reduced_dir2 = "reduced2"
reduced_nightsky_dir = "reduced_nightsky"
solved_dir = "solved"
solved_dir2 = "solved2"
DAOfinder_result_dir = "DAOfinder_result"
IRAFfinder_result_dir = "IRAFfinder_result"
APh_result_dir = "APh_result"
Asteroid_result_dir = "Asteroid_result"
Asteroid_diff_Phot_dir = "Asteroid_diff_Phot"
Inst_Mag_dir = "Inst_Mag_result"
Diff_Phot_dir = "Diff_Phot_result"
Exoplanet_diff_Phot_dir = "Exoplanet_diff_Phot"

master_file_dir = 'master_file_Python/'
processing_dir = 'processing_Python/'
integration_dir = 'integration_Python/'
alignment_dir = 'alignment_Python/'

#######################################################
# OBS instruments information 
#######################################################

#CCDNAME, PIXSIZE, GAIN, RENOISE    
CCDDIC = {"ST-8300M": {"PIXSIZE":5.4, 
                        "GAIN":0.37,
                        "RDNOISE":9.3}, 
        "STF-8300M": {"PIXSIZE":5.4, 
                        "GAIN":0.37,
                        "RDNOISE":9.3}, 
        "QSI683ws": {"PIXSIZE":5.4, 
                        "GAIN":0.13,
                        "RDNOISE":8.0},
        "STL-11000M": {"PIXSIZE":9.0, 
                        "GAIN":0.8,
                        "RDNOISE":9.6},
        "STX-16803": {"PIXSIZE":9.0, 
                        "GAIN":1.27,
                        "RDNOISE":9.0},
        "QHY8": {"PIXSIZE":5.4, 
                "GAIN": "-",
                "RDNOISE":"-"},
        "ATR3CMOS26000KPA": {"PIXSIZE":3.76, 
                "GAIN": "-",
                "RDNOISE":"-"},
        "TT-2600CP": {"PIXSIZE":3.76, 
                "GAIN": "-",
                "RDNOISE":"-"},
        "ASI183MMPro": {"PIXSIZE":2.4, 
                "GAIN": "-",
                "RDNOISE":"-"},
        "ASI6200MMPro": {"PIXSIZE":3.76, 
                "GAIN": "-",
                "RDNOISE":"-"},
                        }

OPTICDIC = {"TMB130ss": {"APATURE" : 130, 
                         "FOCALLEN" : 910},
            "TMB130ss-x75": {"APATURE" : 130, 
                         "FOCALLEN" : 910*0.75}, 
            "RiLA600": {"APATURE" : 600, 
                        "FOCALLEN" : 3000}, 
            "RILA600": {"APATURE" : 600, 
                        "FOCALLEN" : 3000}, 
            "GSON300": {"APATURE" : 300, 
                        "FOCALLEN" : 1200}, 
            "OON300": {"APATURE" : 300, 
                        "FOCALLEN" : 1200}, 
            "FS60CB": {"APATURE" : 60, 
                       "FOCALLEN" : 355}, 
            "SVX80T": {"APATURE": 80,
                        "FOCALLEN": 480},
            "SVX80T-x80": {"APATURE":80,
                        "FOCALLEN": 480*0.8},
            "FSQ106ED": {"APATURE": 106,
                         "FOCALLEN": 530},
            "FSQ106ED-x73": {"APATURE": 106,
                         "FOCALLEN": 530*0.73},
            "FSQ106ED-x72": {"APATURE": 106,
                         "FOCALLEN": 530*0.72},
            "TEC140": {"APATURE": 140,
                       "FOCALLEN": 980},
            "TEC140-x75": {"APATURE": 140,
                       "FOCALLEN": 980*0.75},
            "TEC140-x72": {"APATURE": 140,
                       "FOCALLEN": 980*0.72},
                       }


#######################################################

#%%
#########################################
# Visiulization
#########################################
from warnings import warn
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    ZScaleInterval,
    simple_norm,
)

def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs), stretch=stretch)

def zimshow(
        ax,
        image,
        stretch=LinearStretch(),
        cmap=None,
        origin="lower",
        zscale_kw={},
        **kwargs
    ):
    im = ax.imshow(
        image,
        norm=znorm(image, stretch=stretch, **zscale_kw),
        origin=origin,
        cmap=cmap,
        **kwargs
    )
    return im

def norm_imshow(
    ax,
    data,
    origin="lower",
    stretch="linear",
    power=1.0,
    asinh_a=0.1,
    min_cut=None,
    max_cut=None,
    min_percent=None,
    max_percent=None,
    percent=None,
    clip=True,
    log_a=1000,
    invalid=-1.0,
    zscale=False,
    vmin=None,
    vmax=None,
    **kwargs
):
    """Do normalization and do imshow"""
    if vmin is not None and min_cut is not None:
        warn("vmin will override min_cut.")

    if vmax is not None and max_cut is not None:
        warn("vmax will override max_cut.")

    if zscale:
        zs = ImageNormalize(data, interval=ZScaleInterval())
        min_cut = vmin = zs.vmin
        max_cut = vmax = zs.vmax

    if vmin is not None or vmax is not None:
        im = ax.imshow(data, origin=origin, vmin=vmin, vmax=vmax, **kwargs)
    else:
        im = ax.imshow(
            data,
            origin=origin,
            norm=simple_norm(
                data=data,
                stretch=stretch,
                power=power,
                asinh_a=asinh_a,
                min_cut=min_cut,
                max_cut=max_cut,
                min_percent=min_percent,
                max_percent=max_percent,
                percent=percent,
                clip=clip,
                log_a=log_a,
                invalid=invalid
            ),
            **kwargs)
    return im

def sky_fit(all_sky, method='mode', sky_nsigma=3, sky_iter=5, \
            mode_option='sex', med_factor=2.5, mean_factor=1.5):
    '''
    Estimate sky from given sky values.
    Parameters
    ----------
    all_sky : ~numpy.ndarray
        The sky values as numpy ndarray format. It MUST be 1-d for proper use.
    method : {"mean", "median", "mode"}, optional
        The method to estimate sky value. You can give options to "mode"
        case; see mode_option.
        "mode" is analogous to Mode Estimator Background of photutils.
    sky_nsigma : float, optinal
        The input parameter for sky sigma clipping.
    sky_iter : float, optinal
        The input parameter for sky sigma clipping.
    mode_option : {"sex", "IRAF", "MMM"}, optional.
        sex  == (med_factor, mean_factor) = (2.5, 1.5)
        IRAF == (med_factor, mean_factor) = (3, 2)
        MMM  == (med_factor, mean_factor) = (3, 2)
    Returns
    -------
    sky : float
        The estimated sky value within the all_sky data, after sigma clipping.
    std : float
        The sample standard deviation of sky value within the all_sky data,
        after sigma clipping.
    nsky : int
        The number of pixels which were used for sky estimation after the
        sigma clipping.
    nrej : int
        The number of pixels which are rejected after sigma clipping.
    -------
    '''
    sky = all_sky.copy()
    if method == 'mean':
        return np.mean(sky), np.std(sky, ddof=1)

    elif method == 'median':
        return np.median(sky), np.std(sky, ddof=1)

    elif method == 'mode':
        sky_clip   = sigma_clip(sky, sigma=sky_nsigma,
                                maxiters=sky_iter, #iters=sky_iter,
                                )
        sky_clipped= sky[np.invert(sky_clip.mask)]
        nsky       = np.count_nonzero(sky_clipped)
        mean       = np.mean(sky_clipped)
        med        = np.median(sky_clipped)
        std        = np.std(sky_clipped, ddof=1)
        nrej       = len(all_sky) - len(sky_clipped)

        if nrej < 0:
            raise ValueError('nrej < 0: check the code')

        if nrej > nsky: # rejected > survived
            raise Warning('More than half of the pixels rejected.')

        if mode_option == 'IRAF':
            if (mean < med):
                sky = mean
            else:
                sky = 3 * med - 2 * mean

        elif mode_option == 'MMM':
            sky = 3 * med - 2 * mean

        elif mode_option == 'sex':
            if (mean - med) / std > 0.3:
                sky = med
            else:
                sky = (2.5 * med) - (1.5 * mean)
        else:
            raise ValueError('mode_option not understood')

        return sky, std, nsky, nrej
#%%
#########################################
#calPixScale
#########################################
def calPixScale (
    F_length,
    Pix_size,
    binn) :
    '''
        Parameters
        ----------
        F_length : float or int
            Focal Length of Telescope with out accesery (mm)
        
        Pix_Size : float
            pixel size of detector (um), 
        
        binn : int
            binning number, 
    
        Pixel scale : Pix_Size  /   Telescope Focal Length   )   X 206.265  
            (arcsec / pixel)        
    '''

    PIXScale = Pix_size * binn / (F_length ) *  206.265
    return PIXScale
 
#%%
#########################################
#KvinFitsMover
#########################################
def KevinFitsNewFname(
    fpath,
    fnameKEYs = ["OBJECT", "IMAGETYP", "FILTER", "DATE-OBS", 
            "EXPOSURE", "OPTIC", "CCDNAME", "CCD-TEMP", "XBINNING"],
    ): 
    
    '''
        Parameters
        ----------
        fpath : string
            The fullname of input file...
        
        fnameKEYs : list
            KEY of fits file header for update
    '''
    
    fpath = Path(fpath)
    hdul = fits.open(str(fpath))
    print("fpath: ", fpath)
    for fnameKEY in fnameKEYs: 
        print(f"{fnameKEY}: ", hdul[0].header[fnameKEY])
        try :
            ccdtemp = str(int(hdul[0].header["CCD-TEMP"]))
        except : 
            ccdtemp = "N"
        print("ccdtemp: ", ccdtemp)

        new_fname = hdul[0].header["OBJECT"]+"_"+hdul[0].header["IMAGETYP"]+"_"+hdul[0].header["FILTER"]+"_"
        new_fname += hdul[0].header["DATE-OBS"][:19].replace("T","-").replace(":","-")+"_"
        new_fname += str(int(hdul[0].header["EXPOSURE"]))+"sec_"
        new_fname += hdul[0].header["OPTIC"]+"_"+hdul[0].header["CCDNAME"]+"_"       
        new_fname += ccdtemp+"c_"+str(int(hdul[0].header["XBINNING"]))+"bin.fit"
        #new_fname += fpath.ext
        print("new_fname: ", new_fname)
        hdul.close()
    return new_fname


#%%
#########################################
#KvinFitsUpdater
#########################################
def KevinFitsUpdater(
    fpath,
    checkKEYs = ["OBJECT", "TELESCOP", "OPTIC", "CCDNAME", 'FILTER',
            #"GAIN", "EGAIN", "RDNOISE", 
            "PIXSCALE", "FOCALLEN", "APATURE", "CCD-TEMP",
            'XPIXSZ', 'YPIXSZ',
            "XBINNING", "YBINNING", "FLIPSTAT", "EXPTIME", "EXPOSURE"],
    imgtype_update=False,
    fil_update=True,
    **kwargs
    ):
    '''
        Parameters
        ----------
        fpath : string
            The fullname of input file...
        checkKEYs : dictionary
            KEY of fits file header for update
    '''
    
    fpath = Path(fpath)

    foldername_el = fpath.parts[-2].split('_')
    fname_el = fpath.parts[-1].split('_')
    print("foldername_el", foldername_el)
    print("fname_el", fname_el)
    object_name = foldername_el[0].replace(" ","")
    print("object_name", object_name)
    if imgtype_update == True :
        image_type = foldername_el[1]
        print("image_type", image_type)
    if fil_update == True :
        filter_name = fname_el[2].upper()
        print("filter_name", filter_name)
    optic_name = foldername_el[5]
    print("optic_name", optic_name)
    ccd_name = foldername_el[6]
    print("ccd_name", ccd_name)
    with fits.open(str(fpath), mode="append") as hdul :
        for checkKEY in checkKEYs: 
            if not checkKEY in hdul[0].header :
                hdul[0].header.append(checkKEY, 
                                '', 
                                f"The keyword '{checkKEY}' is added.") 
            print(f"{checkKEY}: ", hdul[0].header[checkKEY])

        hdul.flush()  # changes are written back to original.fits

    # Change something in hdul.
    with fits.open(str(fpath), mode="update") as hdul :
        
        ###########################
        #### "OBJECT"
        hdul[0].header["OBJECT"] = object_name   #delete upper()
        print(f"The 'OBJECT' is set {object_name}")

        ###########################
        #### "FILTER"
        # hdul[0].header["FILTER"] = object_name.upper()
        # print(f"The 'OBJECT' is set {object_name.upper()}")

        ###########################
        #### "CCDNAME"
        if 'INSTRUME' in hdul[0].header :
            if 'qsi' in hdul[0].header['INSTRUME'].lower() :     
                CCDNAME = 'QSI683ws'
            elif 'st-8300' in hdul[0].header['INSTRUME'].lower() : 
                CCDNAME = 'ST-8300M'
            elif 'qhy8' in hdul[0].header['INSTRUME'].lower() : 
                CCDNAME = 'QHY8'
            elif 'stf-8300' in hdul[0].header['INSTRUME'].lower() : 
                CCDNAME = 'STF-8300M'
            elif '11000' in hdul[0].header['INSTRUME'] : 
                CCDNAME = 'STL-11000M'
            elif '16803' in hdul[0].header['INSTRUME'] : 
                CCDNAME = 'STX-16803'
            elif 'ASI183MM Pro' in hdul[0].header['INSTRUME'] : 
                CCDNAME = 'ASI183MMPro'
            elif 'ASI6200MM Pro' in hdul[0].header['INSTRUME'] : 
                CCDNAME = 'ASI6200MMPro'
            elif "SBIG" in hdul[0].header['INSTRUME'] :
                if hdul[0].header['XPIXSZ'] == 5.4 \
                        or hdul[0].header['XPIXSZ'] == 10.8 :
                    CCDNAME = 'STF-8300M'
                elif hdul[0].header['XPIXSZ'] == 9.0 \
                        or hdul[0].header['XPIXSZ'] == 18.0 :
                    if hdul[0].header['NAXIS1'] == 2048 \
                            or  hdul[0].header['NAXIS1'] == 4096 :
                        CCDNAME = 'STX-16803'
                    elif hdul[0].header['NAXIS1'] == 4008 \
                            or  hdul[0].header['NAXIS1'] == 2672 \
                            or  hdul[0].header['NAXIS1'] == 2004 \
                            or  hdul[0].header['NAXIS1'] == 1336 :
                        CCDNAME = 'STL-11000M'
                    else:
                        CCDNAME = ccd_name
            elif "CMOS26000" in hdul[0].header['INSTRUME'] or \
                "ToupTek" in hdul[0].header['INSTRUME'] :
                CCDNAME = "TT-2600CP"
                hdul[0].header["FILTER"] = "-"

            else :
                #CDNAME = hdul[0].header['INSTRUME']
                CCDNAME = ccd_name
        else :
            CCDNAME = ccd_name
        print("CCDNAME", CCDNAME)

        hdul[0].header["CCDNAME"] = CCDNAME
        print(f"The 'CCDNAME' is set {hdul[0].header['CCDNAME']}...")

        ###########################
        #### 'DATE-OBS'
        if len(hdul[0].header['DATE-OBS']) == 10 \
            and 'TIME-OBS' in hdul[0].header : 
            hdul[0].header['DATE-OBS'] += 'T' + hdul[0].header['TIME-OBS']
            print(f"The 'DATE-OBS' is set {hdul[0].header['DATE-OBS']}")
        
        ###########################
        #### 'IMAGETYP'
        if imgtype_update == True :
            hdul[0].header["IMAGETYP"] = image_type.upper()
            print(f"The 'IMAGETYP' is set {hdul[0].header['IMAGETYP']}")
        if not "IMAGETYP" in hdul[0].header :
            hdul[0].header["IMAGETYP"] = image_type  
        elif "ze" in hdul[0].header["IMAGETYP"].lower() \
                or "bi" in hdul[0].header["IMAGETYP"].lower() :
            hdul[0].header["IMAGETYP"] = "BIAS"
            print(f"The 'IMAGETYP' is set {hdul[0].header['IMAGETYP']}")
            hdul[0].header["OBJECT"] = "-"
            print(f"The 'OBJECT' is set {hdul[0].header['OBJECT']}")
        elif "da" in hdul[0].header["IMAGETYP"].lower() :
            hdul[0].header["IMAGETYP"] = "DARK"
            print(f"The 'IMAGETYP' is set {hdul[0].header['IMAGETYP']}")
            hdul[0].header["OBJECT"] = "-"
            print(f"The 'OBJECT' is set {hdul[0].header['OBJECT']}")
        elif "fl" in hdul[0].header["IMAGETYP"].lower() :
            hdul[0].header["IMAGETYP"] = "FLAT"
            print(f"The 'IMAGETYP' is set {hdul[0].header['IMAGETYP']}")
            hdul[0].header["OBJECT"] = "-"
            print(f"The 'OBJECT' is set {hdul[0].header['OBJECT']}")
        elif "obj" in hdul[0].header["IMAGETYP"].lower() \
                or "lig" in hdul[0].header["IMAGETYP"].lower() :
            hdul[0].header["IMAGETYP"] = "LIGHT"
            print(f"The 'IMAGETYP' is set {hdul[0].header['IMAGETYP']}")

        if "BIAS" in hdul[0].header["IMAGETYP"] \
            or "DARK" in hdul[0].header["IMAGETYP"] :
            for _KEY in ['FILTER', 'OPTIC', 'FOCALLEN', 'APATURE', 'PIXSCALE',] :
                hdul[0].header[_KEY] = "-"
                print(f"The '{_KEY}' is set {hdul[0].header[_KEY]}")

        if "FLAT" in hdul[0].header["IMAGETYP"] \
            or "LIGHT" in hdul[0].header["IMAGETYP"] :
            if not "FILTER" in hdul[0].header :
                hdul[0].header["FILTER"] = filter_name
            if hdul[0].header["FILTER"] != filter_name \
                and fil_update==True :
                hdul[0].header["FILTER"] = filter_name
            print(f"FILTER is set {hdul[0].header['FILTER']}")
            if not "OPTIC" in hdul[0].header :
                hdul[0].header["OPTIC"] = optic_name
                print(f"The 'OPTIC' is set {hdul[0].header['OPTIC']}")
            elif  hdul[0].header["OPTIC"] != optic_name :
                hdul[0].header["OPTIC"] = optic_name
                print(f"The 'OPTIC' is set {hdul[0].header['OPTIC']}")

            hdul[0].header['FOCALLEN'] = OPTICDIC[hdul[0].header['OPTIC']]['FOCALLEN']
            print(f"The 'FOCALLEN' is set {hdul[0].header['FOCALLEN']}...")
            hdul[0].header['FOCRATIO'] = OPTICDIC[hdul[0].header['OPTIC']]["FOCALLEN"]/OPTICDIC[hdul[0].header['OPTIC']]["APATURE"]
            print(f"The 'FOCRATIO' is set {hdul[0].header['FOCRATIO']}...")
        
        ###########################
        #### 
        if (not 'TELESCOP' in hdul[0].header):
            hdul[0].header['TELESCOP'] = "-"
            print(f"The 'TELESCOP' is set {hdul[0].header['TELESCOP']}...")
        ###########################
        #### 
        if (not 'XBINNING' in hdul[0].header)\
            and (hdul[0].header["CCDNAME"] == "STX-16803") :
            if hdul[0].header['NAXIS1'] == 4096 \
                or  hdul[0].header['NAXIS2'] == 4096 :
                hdul[0].header['XBINNING'] = 1
                hdul[0].header['YBINNING'] = 1
                hdul[0].header['TELESCOP'] = "-"   
        
            elif hdul[0].header['NAXIS1'] == 2048 \
                or  hdul[0].header['NAXIS2'] == 2048 :
                hdul[0].header['XBINNING'] = 2
                hdul[0].header['YBINNING'] = 2
                hdul[0].header['TELESCOP'] = "-"
        
            elif hdul[0].header['NAXIS1'] == 1024 \
                or  hdul[0].header['NAXIS2'] == 1024 :
                hdul[0].header['XBINNING'] = 3
                hdul[0].header['YBINNING'] = 3
                hdul[0].header['TELESCOP'] = "-"
        hdul[0].header['XBINNING'] = int(hdul[0].header['XBINNING'])
        hdul[0].header['YBINNING'] = int(hdul[0].header['YBINNING'])
        print(f"The 'XBINNING', 'YBINNING' are set {hdul[0].header['XBINNING']}, \
                {hdul[0].header['YBINNING']},...")

        ###########################
        ####
        if (not 'XPIXSZ' in hdul[0].header) \
                and CCDNAME == 'STX-16803' :
            hdul[0].header['XPIXSZ'] = 9 * hdul[0].header['XBINNING']
            hdul[0].header['YPIXSZ'] = 9 * hdul[0].header['YBINNING']
            print(f"The 'XPIXSZ' and 'YPIXSZ' are set {9 * hdul[0].header['XBINNING']} \
                and {9 * hdul[0].header['YBINNING']}...")
        #hdul[0].header['GAIN'] = GAINDIC[CCDNAME]
        #hdul[0].header['GAIN'] = CCDDIC[hdul[0].header['CCDNAME']]['GAIN']
        #print(f"The 'GAIN' is set {hdul[0].header['GAIN']}...")
        #hdul[0].header['EGAIN'] = GAINDIC[CCDNAME]
        #hdul[0].header['EGAIN'] = CCDDIC[hdul[0].header['CCDNAME']]['GAIN']
        #print(f"The 'EGAIN' is set {hdul[0].header['EGAIN']}...")
        #hdul[0].header['RDNOISE'] = RDNOISEDIC[CCDNAME]
        #hdul[0].header['RDNOISE'] = CCDDIC[hdul[0].header['CCDNAME']]['RDNOISE']
        #print(f"The 'RDNOISE' is set {hdul[0].header['RDNOISE']}...")
        
        ###########################
        ####     
        if not "CCD-TEMP" in hdul[0].header :
            hdul[0].header['CCD-TEMP'] = 'N'
            print(f"The 'CCD-TEMP' is set {hdul[0].header['CCD-TEMP']}...")

        ###########################
        #### 
        if "EXPOSURE" in hdul[0].header :
            if not "EXPTIME" in hdul[0].header :
                hdul[0].header["EXPTIME"] = hdul[0].header["EXPOSURE"]
                print(f"The 'EXPTIME' is set {hdul[0].header['EXPOSURE']}...")
        elif "EXPTIME" in hdul[0].header :
            hdul[0].header["EXPOSURE"] = hdul[0].header["EXPTIME"]
            print(f"The 'EXPOSURE' is set {hdul[0].header['EXPTIME']}...")
        else :
            hdul[0].header["EXPTIME"] = 'N'
            hdul[0].header["EXPOSURE"] = 'N'
            print(f"The 'EXPTIME' and 'EXPOSURE' are set 'N'...")

        ###########################
        #### 
        if not "OPTIC" in hdul[0].header :
            hdul[0].header["OPTIC"] = optic_name
        if not "FOCALLEN" in hdul[0].header :
            hdul[0].header["FOCALLEN"] = OPTICDIC[hdul[0].header['OPTIC']]['FOCALLEN']
        if not "APATURE" in hdul[0].header :
            hdul[0].header["APATURE"] = OPTICDIC[hdul[0].header['OPTIC']]["APATURE"]

        print(hdul[0].header['OPTIC']+'_'+hdul[0].header['CCDNAME'])
        if "FLAT" in hdul[0].header["IMAGETYP"] or \
            "LIGHT" in hdul[0].header["IMAGETYP"] :
            if not "PIXSCALE" in hdul[0].header :
                hdul[0].header["PIXSCALE"] = calPixScale(hdul[0].header['FOCALLEN'], 
                                                            hdul[0].header['XPIXSZ'],
                                                            hdul[0].header['XBINNING'],)
            hdul[0].header["PIXSCALE"] = calPixScale(hdul[0].header['FOCALLEN'], 
                                                            hdul[0].header['XPIXSZ'],
                                                            hdul[0].header['XBINNING'],)
        hdul[0].header['FLIPSTAT'] = " "
        print(f"The 'FLIPSTAT' is set {hdul[0].header['FLIPSTAT']}...")
        
        for checkKEY in checkKEYs: 
            print(f"{checkKEY}: ", hdul[0].header[checkKEY])

        hdul.flush()  # changes are written back to original.fits
        print('*'*30)
        print(f"The header of {fpath.name} is updated..")

    return hdul

#%%
class KevinFitsHeader():
    def __init__(self, fpath):
        self.fpath = Path(fpath)
        self.checkKEYs = ["OBJECT", "TELESCOP", "OPTIC", "CCDNAME", 'FILTER',
            "GAIN", "EGAIN", "RDNOISE", "FOCALLEN", "PIXSCALE",
            'XPIXSZ', 'YPIXSZ', "XBINNING", "YBINNING", "FLIPSTAT"]
        '''
        Parameters
        ----------
        fpath : string
            The fullname of input file...

        '''
    def append_header(self):
        with fits.open(str(self.fpath), mode="append") as self.hdul :
            for self.checkKEY in self.checkKEYs: 
                if not self.checkKEY in self.hdul[0].header :
                    self.hdul[0].header.append(self.checkKEY, 
                                    '', 
                                    f"The keyword '{self.checkKEY}' is added.") 
                print(f"{self.checkKEY}: ", self.hdul[0].header[self.checkKEY])

            self.hdul.flush()  # changes are written back to original.fits
        return self.hdul

    def update_header(self): 
        self.foldername_el = self.fpath.parts[-2].split('_')
        self.fname_el = self.fpath.parts[-1].split('_')
        print("foldername_el", self.foldername_el)
        print("fname_el", self.fname_el)
        self.object_name = self.foldername_el[0]
        self.filter_name = self.fname_el[2]
        self.optic_name = self.foldername_el[5]
        self.ccd_name = self.foldername_el[6]
        print("object_name", self.object_name)
        print("filter_name", self.filter_name)
        print("optic_name", self.optic_name)
        print("ccd_name", self.ccd_name)
   
        # Change something in hdul.
        with fits.open(str(self.fpath), mode="update") as self.hdul :
            
            #if object_name != hdul[0].header["OBJECT"] : 
            self.hdul[0].header["OBJECT"] = self.object_name.upper()
            print(f"The 'OBJECT' is set {self.object_name.upper()}")
        
            if len(self.hdul[0].header['DATE-OBS']) == 10 \
                and 'TIME-OBS' in self.hdul[0].header : 
                self.hdul[0].header['DATE-OBS'] += 'T' + self.hdul[0].header['TIME-OBS']
                print(f"The 'DATE-OBS' is set {self.hdul[0].header['DATE-OBS']}")

            if "ze" in self.hdul[0].header["IMAGETYP"].lower() \
                    or "bi" in self.hdul[0].header["IMAGETYP"].lower() :
                self.hdul[0].header["IMAGETYP"] = "BIAS"
                print(f"The 'IMAGETYP' is set {self.hdul[0].header['IMAGETYP']}")
            elif "da" in self.hdul[0].header["IMAGETYP"].lower() :
                self.hdul[0].header["IMAGETYP"] = "DARK"
                print(f"The 'IMAGETYP' is set {self.hdul[0].header['IMAGETYP']}")
            elif "fl" in self.hdul[0].header["IMAGETYP"].lower() :
                self.hdul[0].header["IMAGETYP"] = "FLAT"
                print(f"The 'IMAGETYP' is set {self.hdul[0].header['IMAGETYP']}")
            elif "da" in self.hdul[0].header["IMAGETYP"].lower() \
                    or "lig" in self.hdul[0].header["IMAGETYP"].lower() :
                self.hdul[0].header["IMAGETYP"] = "LIGHT"
                print(f"The 'IMAGETYP' is set {self.hdul[0].header['IMAGETYP']}")
            
            if "BIAS" in self.hdul[0].header["IMAGETYP"] \
                or "DARK" in self.hdul[0].header["IMAGETYP"] :
                self.hdul[0].header["FILTER"] = "-"
                print(f"The 'FILTER' is set {self.hdul[0].header['FILTER']}")
                self.hdul[0].header['OPTIC'] = "-"
                print(f"The 'OPTIC' is set {self.hdul[0].header['OPTIC']}")

            if "FLAT" in self.hdul[0].header["IMAGETYP"] \
                or "LIGHT" in self.hdul[0].header["IMAGETYP"] :
                self.hdul[0].header["FILTER"] = self.filter_name.upper()
                print(f"The 'FILTER' is set {self.hdul[0].header['FILTER']}")
                self.hdul[0].header["OPTIC"] = self.optic_name
                print(f"The 'OPTIC' is set {self.hdul[0].header['OPTIC']}")

            try : 
                if 'qsi' in self.hdul[0].header['INSTRUME'] : 
                    self.CCDNAME = 'QSI683ws'
                elif 'st-8300' in self.hdul[0].header['INSTRUME'] : 
                    self.CCDNAME = 'ST-8300M'
                elif 'stf-8300' in self.hdul[0].header['INSTRUME'] : 
                    self.CCDNAME = 'STF-8300M'
                elif '11000' in self.hdul[0].header['INSTRUME'] : 
                    self.CCDNAME = 'STL-11000M'
                elif '16803' in self.hdul[0].header['INSTRUME'] : 
                    self.CCDNAME = 'STX-16803'
                elif "SBIG" in self.hdul[0].header['INSTRUME'] :
                    if self.hdul[0].header['XPIXSZ'] == 5.4 \
                            or self.hdul[0].header['XPIXSZ'] == 10.8 :
                        self.CCDNAME = 'STF-8300M'
                    elif self.hdul[0].header['XPIXSZ'] == 9.0 \
                            or self.hdul[0].header['XPIXSZ'] == 18.0 :
                        if self.hdul[0].header['NAXIS1'] == 2048 \
                                or self.hdul[0].header['NAXIS1'] == 4096 :
                            self.CCDNAME = 'STX-16803'
                        elif self.hdul[0].header['NAXIS1'] == 4008 \
                                or  self.hdul[0].header['NAXIS1'] == 2672 \
                                or  self.hdul[0].header['NAXIS1'] == 2004 \
                                or  self.hdul[0].header['NAXIS1'] == 1336 :
                            self.CCDNAME = 'STL-11000M'
                else :
                    self.CCDNAME = self.ccd_name    
            except :
                self.CCDNAME = self.ccd_name
            print("CCDNAME", self.CCDNAME)

            #self.hdul[0].header["CCDNAME"] = self.CCDNAME
            #print(f"The 'CCDNAME' is set {self.CCDNAME}...")

            if (not 'XPIXSZ' in self.hdul[0].header) \
                and self.CCDNAME == 'STX-16803' :
                self.hdul[0].header['XPIXSZ'] = 9 * self.hdul[0].header['XBINNING']
                self.hdul[0].header['YPIXSZ'] = 9 * self.hdul[0].header['YBINNING']
                print(f"The 'XPIXSZ' and 'YPIXSZ' are set {9 * self.hdul[0].header['XBINNING']} and {9 * self.hdul[0].header['YBINNING']}...")

            if not "CCD-TEMP" in self.hdul[0].header :
                self.hdul[0].header['CCD-TEMP'] = 'N'
                print(f"The 'CCD-TEMP' is set {self.hdul[0].header['CCD-TEMP']}...")

            if not "EXPOSURE" in self.hdul[0].header :
                self.hdul[0].header["EXPOSURE"] = self.hdul[0].header["EXPTIME"]
                print(f"The 'EXPOSURE' is set {self.hdul[0].header['EXPTIME']}...")

            self.hdul[0].header['GAIN'] = CCDDIC[self.CCDNAME]["GAIN"]
            print(f"The 'GAIN' is set {self.hdul[0].header['GAIN']}...")
            self.hdul[0].header['EGAIN'] = CCDDIC[self.CCDNAME]["EGAIN"]
            print(f"The 'EGAIN' is set {self.hdul[0].header['EGAIN']}...")
            self.hdul[0].header['RDNOISE'] = CCDDIC[self.CCDNAME]["RDNOISE"]
            print(f"The 'RDNOISE' is set {self.hdul[0].header['RDNOISE']}...")
            self.hdul.flush()  # changes are written back to original.fits
            print('*'*30)
            print(f"The header of {self.fpath.name} is updated..")
        return self.hdul



#%%
def fits_newpath(
        fpath,
        rename_by,
        mkdir_by=None,
        header=None,
        delimiter='_',
        fillnan="",
        fileext='.fit',
        **kwargs
):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.
    rename_by : list of str, optional
        The keywords of the FITS header to rename by.
    mkdir_by : list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the
        keys.
    header : Header object, optional
        The header to extract `rename_by` and `mkdir_by`. If `None`, the
        function will do ``header = fits.getheader(fpath)``.
    delimiter : str, optional
        The delimiter for the renaming.
    fillnan : str, optional
        The string that will be inserted if the keyword is not found from the
        header.
    fileext : str, optional
        The extension of the file name to be returned. Normally it should be
        ``'.fits'`` since this function is `fits_newname`, but you may prefer,
        e.g., ``'.fit'`` for some reason. If `fileext` does not start with a
        period (``"."``), it is automatically added to the final file name in
        front of the ``fileext``.
    Returns
    -------
    newpath : path
        The new path.
    '''

    if header is None:
        hdr = fits.getheader(fpath)
    else:
        hdr = header.copy()

    # First make file name without parent path
    hdrvals = []
    for k in rename_by:
        try:
            hdrvals.append(str(hdr[k]))
        except KeyError:
            hdrvals.append(fillnan)

    if not fileext.startswith('.'):
        fileext = f".{fileext}"

    newname = delimiter.join(list(hdrvals))  # just in case, re-listify...
    newname = newname + fileext
    newpath = Path(fpath.parent)

    if mkdir_by is not None:
        for k in mkdir_by:
            newpath = newpath / hdr[k]

    newpath = newpath / newname

    return newpath

#%%
#########################################
#KevinPSolver
#########################################
def KevinSolver(fpath, 
                    solved_dir = None,
                    downsample = 4,
                    pixscale = None,
                    cpulimit = 15,
                    SOLVE = False, 
                    nsigma = 15,
                    tryASTAP = True, 
                    tryLOCAL = True,
                    tryASTROMETRYNET = True,
                    makeLOCALsh = False,
                    **kwargs
                    ):
    """
    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.

    solved dir: string
        The directory where the output file

    pixscale : int

    """
    from astropy.coordinates import SkyCoord
    fpath = Path(fpath)
    hdul = fits.open(fpath)
    
    if downsample == None :
        downsample = 4    
    
    if pixscale is None :
        if 'PIXSCALE' in hdul[0].header:
            pixscale = hdul[0].header['PIXSCALE']
        else : 
            pixscale = calPixScale(hdul[0].header['FOCALLEN'], 
                                        hdul[0].header['XPIXSZ'],
                                        hdul[0].header['XBINNING'])

    # print(f"pixscale: {pixscale:.03f}, L: {pixscale*0.97:.03f}, U: {pixscale*1.03:.03f}")
    
    if cpulimit == None :
        cpulimit = 15  
    if nsigma == None :
        nsigma = 15  

    vfov = pixscale * hdul[0].header['NAXIS1']/3600    
    hfov = pixscale * hdul[0].header['NAXIS2']/3600
    print("vfov , hfov  :", vfov , hfov )
    
    try : 
        spd = 180 + hdul[0].header['DEC']
        ra = hdul[0].header['RA']
        dec = hdul[0].header['DEC']
    except :
        ra_h, ra_m, ra_s = hdul[0].header['RA'].split(':')
        dec_d, dec_m, dec_s = hdul[0].header['DEC'].split(':')
        ra_new = f"{ra_h}h{ra_m}m{ra_s}s"
        dec_new = f"{dec_d}d{dec_m}m{dec_s}s"
        c = SkyCoord(ra=ra_new, dec=dec_new)
        spd = 180 + c.dec.deg
        ra = c.ra.deg
        dec = c.dec.deg
    hdul.close()
    # try :
    SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
    print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
    if not SOLVE and tryASTAP == True : 
        print(f"Trying to solve using ASTAP:\n   {fpath.parent/fpath} ")      
        with subprocess.Popen(['astap', 
                    '-f', str(fpath), 
                    #'-o', 
                    # '-r', f'{ra}',          
                    # '-spd', f'{spd}',
                    # '-fov', f'{hfov}',
                    # '-z', f'{str(downsample)}',
                    '-wcs',
                    '-analyse2',
                    '-update',],
                    stdout=subprocess.PIPE) as proc :
            print(proc.stdout.read())

    SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
    print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
    if not SOLVE and tryASTAP == True :
        print(f"Trying to solve using ASTAP with variables:\n   {fpath.parent/fpath} ")
        with subprocess.Popen(['astap', 
                    '-f', str(fpath), 
                    # '-o', 
                    # '-spd ', f'{spd}',
                    '-fov', f'{hfov}',
                    '-z', f'{str(downsample)}',
                    '-wcs',
                    '-analyse2',
                    '-update',],
                    stdout=subprocess.PIPE) as proc :
            print(proc.stdout.read())

    SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
    print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
    if not SOLVE and tryLOCAL == True : 
        print(f"Trying to solve using LOCAL:\n   {fpath.parent/fpath} ")
        # solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
        with subprocess.Popen(['solve-field', 
                                '-O', #--overwrite: overwrite output files if they already exist
                                # '-g', #--guess-scale: try to guess the image scale from the FITS headers
                                '--cpulimit', f'{cpulimit}',  #will make it give up after 30 seconds.
                                # '--nsigma', f'{nsigma}',
                                # '--downsample', f'{str(downsample)}',
                                # '-u', 'app', #'--scale-units', 'arcsecperpix', #pixel scale
                                # '-L', f'{pixscale*0.95:.03f}', 
                                # '-U', f'{pixscale*1.05:.03f}',   
                                # '-N', f'{fpath.parent / fpath.stem}.new', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
                                # '-N', f'{fpath}', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
                                #-p', 
                                '--no-plots',#: don't create any plots of the results
                                #'-D', str(SOLVEDDIR),
                                str(fpath)
                                ], 
                                stdout=subprocess.PIPE) as proc :
            print(proc.stdout.read())
        
        if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
            #print(str(fpath))
            shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
            print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
            #print(f"{str(fpath)} is removed...")
      
        # SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
        # print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
        # if not SOLVE and tryLOCAL == True : 
        #     print(f"Trying to solve using LOCAL with variables:\n  {fpath.parent/fpath.name} ")
        #     # solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
        #     with subprocess.Popen(['solve-field', 
        #                         '-O', #--overwrite: overwrite output files if they already exist
        #                         '-g', #--guess-scale: try to guess the image scale from the FITS headers
        #                         '--cpulimit', f'{cpulimit}',  #will make it give up after 30 seconds.
        #                         '--nsigma', f'{nsigma}',
        #                         '--downsample', f'{str(downsample)}',
        #                         '-u', 'app', #'--scale-units', 'arcsecperpix', #pixel scale
        #                         '-L', f'{pixscale*0.95:.03f}', '-H', f'{pixscale*1.05:.03f}',   
        #                         # '--ra', f'{ra}', '--dec', f'{dec}',
        #                         # '-N', f'{fpath.parent / fpath.stem}.new', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
        #                         # '-N', f'{fpath}', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
        #                         #-p', 
        #                         '--no-plots',#: don't create any plots of the results
        #                         #'-D', str(SOLVEDDIR),
        #                         str(fpath)
        #                         ], 
        #                         stdout=subprocess.PIPE) as proc :
        #         print(proc.stdout.read())

        # if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
        #     #print(str(fpath))
        #     shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        #     print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        #     #print(f"{str(fpath)} is removed...")

        SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
        print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
        if not SOLVE and tryLOCAL == True : 
            print(f"Trying to solve using LOCAL with variables again :\n  {fpath.parent/fpath.name} ")
            # solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
            with subprocess.Popen(['solve-field', 
                                '-O', #--overwrite: overwrite output files if they already exist
                                '-g', #--guess-scale: try to guess the image scale from the FITS headers
                                '--cpulimit', f'{cpulimit}',  #will make it give up after 30 seconds.
                                '--nsigma', f'{nsigma}',
                                '--downsample', f'{str(downsample)}',
                                '-u', 'app', #'--scale-units', 'arcsecperpix', #pixel scale
                                '-L', f'{pixscale*0.95:.03f}', '-H', f'{pixscale*1.05:.03f}',   
                                '--ra', f'{ra}', '--dec', f'{dec}',
                                # '-N', f'{fpath.parent / fpath.stem}.new', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
                                # '-N', f'{fpath}', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
                                #-p', 
                                '--no-plots',#: don't create any plots of the results
                                #'-D', str(SOLVEDDIR),
                                str(fpath)
                                ], 
                                stdout=subprocess.PIPE) as proc :
                print(proc.stdout.read())

        if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
            #print(str(fpath))
            shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
            print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
            #print(f"{str(fpath)} is removed...")
            
        SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
        print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)

        if not SOLVE and tryASTROMETRYNET == True : 
            print(f"Trying to solve using astrometry:\n  {fpath.parent/fpath.name} ")
            from astroquery.astrometry_net import AstrometryNet
            ast = AstrometryNet()
            # ger from nova.astrometry.net
            ast.api_key = 'bldvwzzuvktnwfph' #must changed...
            
            submission_id = None
            solve_timeout = 600
            try_again = True                

            try :                
                while try_again:
                    try:
                        if not submission_id:
                            wcs_header = ast.solve_from_image(str(fpath),
                                                force_image_upload=True,
                                                solve_timeout = solve_timeout,
                                                submission_id=submission_id)
                        else:
                            wcs_header = ast.monitor_submission(submission_id,
                                                                solve_timeout = solve_timeout)
                    except TimeoutError as e:
                        submission_id = e.args[1]
                    else:
                        # got a result, so terminate
                        try_again = False

                if not wcs_header:
                    # Code to execute when solve fails
                    print("fits file solving failure...")

                else:
                    # Code to execute when solve succeeds
                    print("fits file solved successfully...")

                    with fits.open(str(fpath), mode='update') as hdul:
                        for card in wcs_header :
                            try: 
                                print(card, wcs_header[card], wcs_header.comments[card])
                                hdul[0].header.set(card, wcs_header[card], wcs_header.comments[card])
                            except : 
                                print(card)
                        hdul.flush

                    print(str(fpath)+" is created...")
            
            except Exception as err: 
                print("Err :", err)
                pass 

    SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
    print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
    
    # if not SOLVE and makeLOCALsh == True : 
    #     print(f"Adding to sh:\n   {fpath.name} ")
    #     makingAstrometrySH(fpath, 
    #                     # solved_dir = None,
    #                     downsample = 1,
    #                     pixscale = pixscale,
    #                     # SOLVE = False, 
    #                     cpulimit = 30,
    #                     )

    if not SOLVE and makeLOCALsh == True :
        #solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
        solve_sh += f"solve-field -O -g --cpulimit {cpulimit} --nsigma {nsigma} --downsample {downsample} -u app -L {pixscale*0.95:.03f} -U {pixscale*1.05:.03f} --no-plots {str(fpath)}\n"
        solve_sh += f"mv {fpath.parent/fpath.stem}.new {str(fpath)}\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}-indx.xyls\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}-indx.axy\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.rdls\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.corr\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.solved\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.match\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.axy\n"
        # solve_sh += f"rm {fpath.parent/fpath.stem}.wcs\n"
        # print("result:", solve_sh)

        with open(f"__{datetime.now().strftime('%Y%m%d')}_todo_astrometry_solve.sh", 'a') as f:
            f.write(solve_sh) 

    return 0
        


# #%%
# #########################################
# # LOCALPSolver
# #########################################
# def LOCALPSolver(fpath, 
#                     solved_dir = None,
#                     downsample = 4,
#                     pixscale = None,
#                     cpulimit = 30,
#                     nsigma = 15,
#                     **kwargs
#                     ):
#     """
#     Parameters
#     ----------
#     fpath : path-like
#         The path to the original FITS file.

#     solved dir: string
#         The directory where the output file

#     pixscale : int

#     """
#     fpath = Path(fpath)

#     if pixscale is None :
#         hdul = fits.open(fpath)
#         if 'PIXSCALE' in hdul[0].header:
#             pixscale = hdul[0].header['PIXSCALE']
#         else : 
#             pixscale = calPixScale(hdul[0].header['FOCALLEN'], hdul[0].header['XPIXSZ'])
#         hdul.close()
#     print(f"pixscale: {pixscale:.03f}, L: {pixscale*0.97:.03f}, U: {pixscale*1.03:.03f}")
    
#     # solve command.
#     # solve-field fullname.fit -O --cpulimit 120 --nsigma 15 -u app -L 1.2 -U 1.3 -N new_filename.fits -p --no-plots -D output_directory {0}
#     with subprocess.Popen(['solve-field', 
#                                 '-O', #--overwrite: overwrite output files if they already exist
#                                 '-g', #--guess-scale: try to guess the image scale from the FITS headers
#                                 '--cpulimit', f'{cpulimit}',  #will make it give up after 30 seconds.
#                                 '--nsigma', f'{nsigma}',
#                                 '--downsample', f'{str(downsample)}',
#                                 '-u', 'app', #'--scale-units', 'arcsecperpix', #pixel scale
#                                 '-L', f'{pixscale*0.95:.03f}', 
#                                 '-U', f'{pixscale*1.05:.03f}',   
#                                 # '-N', f'{fpath.parent / fpath.stem}.new', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
#                                 # '-N', f'{fpath}', #--new-fits <filename>: output filename of the new FITS file containingthe WCS header; "none" to not create this file
#                                 #-p', 
#                                 '--no-plots',#: don't create any plots of the results
#                                 #'-D', str(SOLVEDDIR),
#                                 str(fpath)
#                                 ], 
#                                 stdout=subprocess.PIPE) as proc :
#         print(proc.stdout.read())

#     if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
#         #print(str(fpath))
#         shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
#         print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
#         #print(f"{str(fpath)} is removed...")

# #%%
# #########################################
# #ASTAPPSolver
# #########################################
# def ASTAPSolver(fpath, 
#                     solved_dir = None,
#                     downsample = 4,
#                     pixscale = None,
#                     **kwargs
#                     ):
#     """
#     Parameters
#     ----------
#     fpath : path-like
#         The path to the original FITS file.

#     solved dir: string
#         The directory where the output file

#     pixscale : int

#     """
#     fpath = Path(fpath)

#     if pixscale is None :
#         hdul = fits.open(fpath)
#         if 'PIXSCALE' in hdul[0].header:
#             pixscale = hdul[0].header['PIXSCALE']
#         else : 
#             pixscale = calPixScale(hdul[0].header['FOCALLEN'], hdul[0].header['XPIXSZ'])
#         hdul.close()
#     print(f"pixscale: {pixscale:.03f}, L: {pixscale*0.97:.03f}, U: {pixscale*1.03:.03f}")

    
#     #try :
#     #https://www.hnsky.org/astap.htm#astap_command_line
#     with subprocess.Popen(['astap', 
#                         '-f', str(fpath), 
#                         # '-o', 
#                         # '-fov',
#                         '-z', f'{str(downsample)}',
#                         # '-wcs',
#                         '-analyse2',
#                         '-update',],
#                         stdout=subprocess.PIPE) as proc :
#         print(proc.stdout.read())
    


# #%%
# #########################################
# #AstrometrynetSolver
# #########################################
# def AstrometrynetSolver(fpath, 
#                     solved_dir = None,
#                     downsample = 4,
#                     pixscale = None,
#                     solve_timeout = 600,
#                     submission_id = None,
#                     ast = AstrometryNet(),
#                     ast_api_key = 'bldvwzzuvktnwfph',
#                     **kwargs
#                     ):
#     ''' Gives the new path of the FITS file from header.
#     Parameters
#     ----------
#     DOINGDIR: pathlike
#         The path to the original .
#     summary : dataframe
        
#     Returns
#     -------
    
#     '''
#     ast = ast

#     # ger from nova.astrometry.net
#     ast.api_key = ast_api_key #must changed...

#     fpath = Path(fpath)
#     print(fpath)
#     hdul = fits.open(fpath)

#     if 'PIXSCALE' in hdul[0].header:
#         PIXc = hdul[0].header['PIXSCALE']
#     else : 
#         PIXc = calPixScale(hdul[0].header['FOCALLEN'], 
#                                             hdul[0].header['XPIXSZ'],
#                                             hdul[0].header['XBINNING'])
#     print("PIXc : ", PIXc)
#     hdul.close()

#     SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
#     print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)

#     if SOLVE :
#         print(f"{fpath.name} is already solved...")
#     else :             
#         try_again = True                
            
#         while try_again:
#             try:
#                 if not submission_id:
#                     wcs_header = ast.solve_from_image(str(fpath),
#                                         force_image_upload=True,
#                                         solve_timeout = solve_timeout,
#                                         submission_id=submission_id)
#                 else:
#                     wcs_header = ast.monitor_submission(submission_id,
#                                                         solve_timeout = solve_timeout)
#             except TimeoutError as e:
#                 submission_id = e.args[1]
#             else:
#                 # got a result, so terminate
#                 try_again = False

#         if not wcs_header:
#             # Code to execute when solve fails
#             print("fits file solving failure...")

#         else:
#             # Code to execute when solve succeeds
#             print("fits file solved successfully...")

#             with fits.open(str(fpath), mode='update') as hdul:
#                 for card in wcs_header :
#                     try: 
#                         print(card, wcs_header[card], wcs_header.comments[card])
#                         hdul[0].header.set(card, wcs_header[card], wcs_header.comments[card])
#                     except : 
#                         print(card)
#                 hdul.flush

#             print(str(fpath)+" is created...")
#     return 0

#%%
#########################################
# checkPSolve
#########################################
def checkPSolve(fpath, 
                    **kwargs,
                    ):
    """
    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.

    return
    ----------
    SOLVE, ASTAP, LOCAL :  bool, bool, bool

    """

    fpath = Path(fpath)

    if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
        #print(str(fpath))
        shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        #print(f"{str(fpath)} is removed...")

    hdul = fits.open(fpath)
    PSKeys = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2", 
              "A_0_0", "A_0_1", "A_1_0","A_1_1",
              "PC1_1", "PC1_2", "PC2_1", "PC2_2", 
              ]
    
    chk = 0
    SOLVE = False
    ASTAP = False
    LOCAL = False

    for PSKey in PSKeys :
        if PSKey in hdul[0].header : 
            chk += 1
    if chk > 5 : 
        SOLVE = True
        LOCAL = False
        ASTAP = False
        try : 
            for comment in hdul[0].header["COMMENT"]:
                if "scale:" in comment :
                    LOCAL = True
        except :
            LOCAL = False
                        
        if "PLTSOLVD" in hdul[0].header:
            try : 
                ASTAP = hdul[0].header["PLTSOLVD"]
            except : 
                ASTAP = False
    else : 
        SOLVE = False
        ASTAP = False
        LOCAL = False
    hdul.close()
    remove_ext  = [".ini", ".axy", ".corr", ".match", ".rdls", 
                   ".solved", "-indx.xyls", ".solved", "-objs.png", "-ngc.png", "-indx.png", 
                   ".wcs"]
    for ext in remove_ext : 
        if (fpath.parent / f"{fpath.stem}{ext}").exists() :
            os.remove(fpath.parent / f"{fpath.stem}{ext}")
            print(f"{fpath.parent}/{fpath.stem}{ext} is removed...")
    
    if fpath.exists() and (fpath.parent/f'{fpath.stem}.new').exists():
        #print(str(fpath))
        shutil.move(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        print(str(fpath.parent/f'{fpath.stem}.new'), str(fpath))
        #print(f"{str(fpath)} is removed...")

    return SOLVE, ASTAP, LOCAL


# #%%
# #########################################
# # checkSolve
# #########################################
# def checkSolve(fpath, 
#                     **kwargs,
#                     ):
#     """
#     Parameters
#     ----------
#     fpath : path-like
#         The path to the original FITS file.

#     return
#     ----------
#     SOLVE, ASTAP, LOCAL :  bool, bool, bool

#     """

#     fpath = Path(fpath)
#     hdul = fits.open(fpath)
#     PSKeys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2", 
#               "A_0_0", "A_0_1", "A_1_0","A_1_1",
#               "PC1_1", "PC1_2", "PC2_1", "PC2_2", ]
    
#     chk = 0
#     SOLVE = False

#     for PSKey in PSKeys :
#         if PSKey in hdul[0].header : 
#             chk += 1
#     if chk > 3 : 
#         SOLVE = True
        
#     hdul.close()
#     remove_ext  = [".ini", ".axy", ".corr", ".match", ".rdls", 
#                    ".solved", "-indx.xyls", ".solved"]
#     for ext in remove_ext : 
#         if (fpath.parent / f"{fpath.stem}{ext}").exists() :
#             os.remove(fpath.parent / f"{fpath.stem}{ext}")
#             print(f"{fpath.parent}/{fpath.stem}{ext} is removed...")

#     return SOLVE

# #%%
# #########################################
# # makingAstrometrySH
# #########################################
# def makingAstrometrySH(fpath, 
#                         solved_dir = None,
#                         downsample = 4,
#                         pixscale = None,
#                         SOLVE = False, 
#                         cpulimit = 30,
#                         nsigma = 15,
#                         **kwargs
#                         ): 
#     """
#     Parameters
#     ----------
#     fpath : path-like
#         The path to the original FITS file.

#     solved dir: string
#         The directory where the output file

#     pixscale : int

#     """
#     fpath = Path(fpath)
#     solve_sh = ""
    
#     if pixscale is None :
#         hdul = fits.open(fpath)
#         if 'PIXSCALE' in hdul[0].header:
#             pixscale = hdul[0].header['PIXSCALE']
#         else : 
#             pixscale = calPixScale(hdul[0].header['FOCALLEN'], 
#                                         hdul[0].header['XPIXSZ'],
#                                         hdul[0].header['XBINNING'])
#         hdul.close()
#     print(f"pixscale: {pixscale:.03f}, L: {pixscale*0.97:.03f}, U: {pixscale*1.03:.03f}")
    
#     # try :
#     SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
#     print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)
#     if not SOLVE : 
#         #solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
#         #solve-field -O -g --cpulimit 15 --nsigma 15 --downsample 4 -u app -L 0.6 -U 0.63 --no-plots
#         solve_sh += f"solve-field -O -g --cpulimit {cpulimit} --nsigma {nsigma} --downsample {downsample} -u app -L {pixscale*0.95:.03f} -U {pixscale*1.05:.03f} --no-plots {str(fpath)}\n"
#         solve_sh += f"mv {fpath.parent/fpath.stem}.new {str(fpath)}\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}-indx.xyls\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}-indx.axy\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.rdls\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.corr\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.solved\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.match\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.axy\n"
#         solve_sh += f"rm {fpath.parent/fpath.stem}.wcs\n"
#         print("result:", solve_sh)

#         with open(f"__{datetime.now().strftime('%Y%m%d')}_todo_astrometry_solve.sh", 'a') as f:
#             f.write(solve_sh) 
#     return solve_sh

#%%        
# =============================================================================
def get_RADEC_offset(hdul,
                     **kwarg
                     ):
    """
    Parameters
    ----------
    hdul : fits file
    """
    Suwon = location = EarthLocation(lon=127.005 * u.deg, lat=37.308889 * u.deg, height=101 * u.m)
    #print('Starting get_new_foldername ...\n{0}'.format(filename))   
    w = WCS(hdul[0].header)
    t_obs = Time(hdul[0].header["DATE-OBS"]) + hdul[0].header["EXPOSURE"] * u.s / 2  # middle of observation time
    cent_coord = yfu.center_radec(ccd_or_header=hdul[0].header, center_of_image=True)
    print(f"calcualted (RA, DEC): ({cent_coord.ra}, {cent_coord.dec})")
    print(f"in header (RA, DEC): ({hdul[0].header['RA']*u.deg}, {hdul[0].header['DEC']*u.deg})")
    offset_RA = (cent_coord.ra - hdul[0].header['RA']*u.deg).to(u.arcmin)
    offset_DEC = (cent_coord.dec - hdul[0].header['DEC']*u.deg).to(u.arcmin)
    
    altaz = AltAz(obstime=t_obs, location=Suwon)

    cent_aa = cent_coord.transform_to(altaz)
    print(f"calculated (Az, Alt): ({cent_aa.az}, {cent_aa.alt})")
    print(f"in header (Az, Alt): ({hdul[0].header['CENTAZ ']*u.deg}, {hdul[0].header['CENTALT']*u.deg})")
    offset_AZ = (cent_aa.az - hdul[0].header['CENTAZ']*u.deg).to(u.arcmin)
    offset_ALT = (cent_aa.alt - hdul[0].header['CENTALT']*u.deg).to(u.arcmin)
    
    return offset_RA, offset_DEC, offset_AZ, offset_ALT

#%%        
# =============================================================================
def get_new_foldername_from_filename(filename):
    """
    Parameters
    ----------
    filename : str, path-like
        The path to the original FITS file.
    """
    #print('Starting get_new_foldername ...\n{0}'.format(filename))   
    filename_stem = filename.split(".")
    filename_el = filename_stem[-2].split("_")
    #print("filename_el: ", filename_el)
    timez = 9

    if int(filename_el[3][17:19])>=60 :
        obs_UT = datetime.strptime("{}59".format(filename_el[3][:17]), '%Y-%m-%d-%H-%M-%S')
    else:
         obs_UT = datetime.strptime(filename_el[3], '%Y-%m-%d-%H-%M-%S')
    obs_LST = obs_UT + timedelta(hours = timez)
    if obs_LST.hour < 12 :
        obs_LST = obs_LST - timedelta(days = 1)
    filename_el[3] = obs_LST.strftime('%Y-%m-%d-%H-%M-%S')
    if filename_el[1].upper() == 'BIAS':
        new_foldername = '{6}_{8}/Cal/-_{1}_-_{3}_-_-_{6}_-_{8}/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    elif filename_el[1].upper() == 'DARK' :
        new_foldername = '{6}_{8}/Cal/-_{1}_-_{3}_{4}_-_{6}_-_{8}/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    elif filename_el[1].upper() == 'FLAT' :
        new_foldername = '{6}_{8}/Cal_{5}/-_{1}_-_{3}_-_{5}_{6}_-_{8}/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    else : 
        new_foldername = '{6}_{8}/LIGHT_{5}/{0}_{1}_-_{3}_-_{5}_{6}_-_{8}/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    #write_log(log_file, 
    #            '{1} ::: \nNew foldername is {0} ...'\
    #            .format(new_foldername, datetime.now()))    
    return new_foldername


def get_new_foldername(filename):
    #log_file = 'get_new_foldername.log'
    print('Starting get_new_foldername ...\n{0}'.format(filename))
    
    filename_el1 = filename.split("bin")
    filename_el = filename_el1[0].split("_")
    
    if filename_el[1].lower() == 'BIAS':
        new_foldername = '{6}_{8}bin/Cal/-_{3}_-_{1}_-_{4}_-_{6}_-_{8}bin/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    elif filename_el[1].lower() == 'DARK' :
        new_foldername = '{6}_{8}bin/Cal/-_{3}_-_{1}_-_{4}_-_{6}_-_{8}bin/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    elif filename_el[1].lower() == 'FLAT' :
        new_foldername = '{6}_{8}bin/Cal_{5}/-_{3}_-_{1}_-_{5}_{6}_-_{8}bin/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    else : 
        new_foldername = '{6}_{8}bin/LIGHT_{5}/{0}_{1}_-_{3}_-_{5}_{6}_-_{8}bin/'\
        .format(filename_el[0],
        filename_el[1],
        filename_el[2],
        filename_el[3][:10],
        filename_el[4],
        filename_el[5],
        filename_el[6],
        filename_el[7],
        filename_el[8])
    #write_log(log_file, 
    #            '{1} ::: \nNew foldername is {0} ...'\
    #            .format(new_foldername, datetime.now()))    
    return new_foldername


#%%
#########################################
#combine_BDF
#########################################
def combine_BDF(BDFDIR,
                tryagain = False,
                **kwarg
                ):
    
    print(f"Starting: {str(BDFDIR.parts[-1])}")

    MASTERDIR = BDFDIR / master_dir
    if not MASTERDIR.exists():
        os.makedirs("{}".format(str(MASTERDIR)))
        print("{} is created...".format(str(MASTERDIR)))
    print ("MASTERDIR: ", format(MASTERDIR))

    summary = yfu.make_summary(BDFDIR/"*.fit*", 
                                verbose = False,
                                )
    
    if summary is not None :
        #print(summary)
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])

        if (MASTERDIR / "master_bias.fits").exists() and not tryagain:
            print("bias file is already exist....")
        else :
            #bias_fits = summary[summary["IMAGETYP"] == "BIAS"]["file"]
            bias_fits = summary.loc[summary["IMAGETYP"] == "BIAS"].copy()
            bias_fits.reset_index(inplace=True)
            bias_fits = bias_fits["file"]
            print(type(bias_fits))
            print(len(bias_fits))
            print(bias_fits)

            bias_comb = yfu.group_combine(
                            bias_fits.tolist(),
                            type_key = ["IMAGETYP"],
                            type_val = ["BIAS"],
                            group_key = ["EXPTIME"],
                            fmt = "master_bias.fits",  # output file name format
                            outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                            combine = "med",
                            memlimit = 2.e+10,
                            verbose = True,
                        )

        #dark_fits = summary[summary["IMAGETYP"] == "DARK"]["file"]
        dark_fits = summary.loc[summary["IMAGETYP"] == "DARK"].copy()
        dark_fits.reset_index(inplace=True)
        dark_fits = dark_fits["file"]
        print(type(dark_fits))
        print(len(dark_fits))
        print(dark_fits)

        # Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
        dark_comb = yfu.group_combine(
                        dark_fits.tolist(),
                        type_key = ["IMAGETYP"],
                        type_val = ["DARK"],
                        group_key = ["EXPTIME"],
                        fmt = "master_dark_{:.0f}sec.fits",  # output file name format
                        outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                        combine = "med",
                        memlimit = 2.e+10,
                        verbose = True,
                    )

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
                        outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                        combine = "med",
                        memlimit = 2.e+10,
                        verbose=True,
                    )

        # Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
        flat_comb = yfu.group_combine(
                        flat_fits.tolist(),
                        type_key = ["IMAGETYP"],
                        type_val = ["FLAT"],
                        group_key = ["FILTER"],
                        fmt = "master_flat_{:s}.fits",  # output file name format
                        #scale="med_sc", #norm
                        #scale_to_0th=False, #norm
                        outdir = MASTERDIR,  # output directory (will automatically be made if not exist)
                        combine = "med",
                        memlimit = 2.e+10,
                        verbose=True,
                    )
    return 0

#%%
#########################################
#solving_fits_fils
#########################################
def solving_fits_file(DOINGDIR,
                SOLVINGDIR = None,
                tryagain = False,  
                # tryASTAP = True, 
                # tryLOCAL = True,
                tryASTROMETRYNET = False,  
                **kwarg,
                ):
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)

    if SOLVINGDIR is None :
        SOLVINGDIR = DOINGDIR
    else : 
        SOLVINGDIR = DOINGDIR / SOLVINGDIR

    summary = yfu.make_summary(SOLVINGDIR/"*.fit*",
                                    verify_fix=True,
                                    ignore_missing_simple=True,
                                    )
    if summary is not None :
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])  
        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)
        print("df_light:\n{}".format(df_light))
        df_light

        for _, row  in df_light.iterrows():

            fpath = Path(row["file"])
            # fpath = Path(df_light["file"][1])
            print("fpath :" ,fpath)
            # hdul = fits.open(fpath)

            try : 
                KevinSolver(fpath, 
                            # solved_dir = None,
                            # downsample = 4,
                            # pixscale = None ,
                            # cpulimit = 15,
                            # tryASTAP = True, 
                            # tryLOCAL = True,
                            tryASTROMETRYNET = tryASTROMETRYNET, 
                            makeLOCALsh = False,
                            )
            except Exception as err :
                print("X"*60)
                print(err)

    return 0

#%%
#########################################
#ccd_Reducuction
#########################################
def ccd_Reducuction (DOINGDIR,
                    BDFDIR,
                    tryagain = False,
                    trynightsky = True,
                    **kwarg) :
    DOINGDIR = Path(DOINGDIR)
    print(f"Starting: {str(DOINGDIR.parts[-1])}")
    
    MASTERDIR = BDFDIR / master_dir
    sMASTERDIR = DOINGDIR / master_dir
    REDUCEDDIR = DOINGDIR / reduced_dir

    if not sMASTERDIR.exists():
        os.makedirs(str(sMASTERDIR))
        print("{} is created...".format(str(sMASTERDIR)))

    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        print("{} is created...".format(str(REDUCEDDIR)))

    summary = yfu.make_summary(DOINGDIR/"*.fit*")
    if summary is not None :
        #print(summary)
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])

        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)

        for _, row in df_light.iterrows():

            fpath = Path(row["file"])
            ccd = yfu.load_ccd(fpath)
            filt = ccd.header["FILTER"]
            expt = ccd.header["EXPTIME"]
            if (REDUCEDDIR/ fpath.name).exists() and not tryagain:
                print(f"reduction file already exists...\n{fpath.name}")
                pass
            else :
                try : 
                    red = yfu.ccdred(
                        ccd,
                        output = REDUCEDDIR/ fpath.name,
                        mbiaspath = MASTERDIR / "master_bias.fits",
                        mdarkpath = MASTERDIR / f"master_dark_{expt:.0f}sec.fits",
                        mflatpath = MASTERDIR / f"master_flat_{filt.upper()}_norm.fits",
                        # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                        dark_scale = False,
                        overwrite=True
                        )
                except : 
                    dark_exp = 100
                    red = yfu.ccdred(
                        ccd,
                        output = REDUCEDDIR / fpath.name,
                        mbiaspath = MASTERDIR / "master_bias.fits",
                        mdarkpath = MASTERDIR / f"master_dark_{dark_exp}sec.fits",
                        mflatpath = MASTERDIR / f"master_flat_{filt.upper()}_norm.fits",
                        dark_scale = True,
                        exptime_dark = dark_exp,
                        # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                        overwrite=True,
                        )

    if trynightsky == True : 
        REDUC_nightsky = DOINGDIR / reduced_nightsky_dir
        if not REDUC_nightsky.exists():
            os.makedirs("{}".format(str(REDUC_nightsky)))
            print("{} is created...".format(str(REDUC_nightsky)))
    
        summary = yfu.make_summary(REDUCEDDIR /"*.fit*")
        if summary is not None :
            print("len(summary):", len(summary))
            print("summary:", summary)
            #print(summary["file"][0])   

            df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
            df_light = df_light.reset_index(drop=True)

            for filt in ["B", "V", "R", "L", "R", "G", "B", "b", "v", "r"]:
            #for filt in ["V"]:
                df_light_filt = df_light.loc[df_light["FILTER"] == filt].copy()
                
                if df_light_filt.empty:
                    print(f"The dataframe(df_light_filt) {filt} is empty")
                    pass
                else:

                    print("len(df_light_filt):", len(df_light_filt))
                    print("df_light_filt:", df_light_filt)
                
                    if (sMASTERDIR / f"nightskyflat-{filt}.fits").exists() and not tryagain :
                        pass
                    else :      
                        File_Num = 80
                        if len(df_light_filt["file"]) > File_Num :
                            combine_lst = df_light_filt["file"].tolist()[:File_Num]
                        else : 
                            combine_lst = df_light_filt["file"].tolist()
                        if (sMASTERDIR / f"nightskyflat-{filt}_norm.fits").exists():
                            pass
                        else :
                            try : 
                                ccd = yfu.imcombine(
                                                    combine_lst, 
                                                    combine="med",
                                                    scale="avg", 
                                                    scale_to_0th=False, #norm
                                                    reject="sc", 
                                                    sigma=2.5,
                                                    verbose=True,
                                                    memlimit = 2.e+11,
                                                    )
                            except :
                                ccd = yfu.imcombine(
                                                    combine_lst, 
                                                    combine="med",
                                                    scale="avg", 
                                                    scale_to_0th=False, #norm
                                                    reject="sc", 
                                                    # sigma=2.5,
                                                    verbose=True,
                                                    memlimit = 2.e+11,
                                                    )
                            ccd.write(sMASTERDIR / f"nightskyflat-{filt}_norm.fits", overwrite=True)
                            print (f"Create Create nightskyflat-{filt}_norm.fits +++...")

            for _, row in df_light.iterrows():
                fpath = Path(row["file"])
                ccd = yfu.load_ccd(REDUCEDDIR / fpath.name)
                filt = row["FILTER"]
                if (REDUC_nightsky / fpath.name).exists() and not tryagain:
                    print(f"Nightsky reduction file already exists...\n{fpath.name}")
                    pass
                else :
                    try:    
                        ccd = yfu.ccdred(
                                        ccd, 
                                        mflatpath = sMASTERDIR / f"nightskyflat-{filt}_norm.fits",
                                        output = REDUC_nightsky / fpath.name
                                    )
                    except : 
                        # _Python_utilities.write_log(err_log_file, "FileNotFoundError") 
                        pass
    return 0

#%%
#########################################
#plot_light_curve_variables_using_csv
#########################################
def plot_light_curve_variables_using_csv(DOINGDIR,
                                        READINGDIR = reduced_dir,
                                        FWHM_INIT = 6,
                                        Mag_target = 12.5,
                                        ERR_Min = 0.5,
                                        coord_deltas = np.arange(0.00001, 0.00050, 0.00001),
                                        #  READINGDIR = _astro_utilities.reduced__nightsky_dir,
                                         **kwarg
                                        ) :
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)

    READINGDIR = DOINGDIR / READINGDIR
    # READINGDIR = DOINGDIR / _astro_utilities.reduced_nightsky_dir

    DIFFPRESULTDIR = DOINGDIR / f"{READINGDIR.parts[-1]}_DPhot_Mag{Mag_target}"
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

                    # plt.show()
                    # plt.close()
#%%
#########################################
#reduceLightFrame
#########################################
def reduceLightFrame(
        DOINGDIR,
        MASTERDIR,
        OWrite=False,
        **kwargs
        ):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    DOINGDIR: pathlike
        The path to the original .
    summary : dataframe
        
    Returns
    -------
    
    '''
    DOINGDIR = Path(DOINGDIR)
    print (DOINGDIR.parts[-1])
    sMASTERDIR = DOINGDIR / master_dir
    REDUCEDDIR = DOINGDIR / reduced_dir

    if not sMASTERDIR.exists():
        os.makedirs(str(sMASTERDIR))
        print("{} is created...".format(str(sMASTERDIR)))

    if not REDUCEDDIR.exists():
        os.makedirs(str(REDUCEDDIR))
        print("{} is created...".format(str(REDUCEDDIR)))

    summary = yfu.make_summary(DOINGDIR/"*.fit*")
    if summary is not None :
        #print(summary)
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])

        df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        df_light = df_light.reset_index(drop=True)

        for _, row in df_light.iterrows():

            fpath = Path(row["file"])
            ccd = yfu.load_ccd(fpath)
            filt = ccd.header["FILTER"]
            expt = ccd.header["EXPTIME"]
            if (not (REDUCEDDIR/fpath.name).exists()) or OWrite==True :
                red = yfu.ccdred(
                    ccd,
                    output=Path(f"{REDUCEDDIR/ fpath.name}"),
                    mdarkpath=str(MASTERDIR / "master_dark_{:.0f}sec.fits".format(expt)),
                    mflatpath=str(MASTERDIR / "master_flat_{}_norm.fits".format(filt.upper())),
                    # flat_norm_value=1,  # 1 = skip normalization, None = normalize by mean
                    # overwrite=OWrite,
                    )
                print (f"Reduce Reduce {fpath.name} +++...")

    return 0

#%%
#########################################
#makeNightskyflatReduceLightFrame
#########################################
def makeNightskyflatReduceLightFrame(
        DOINGDIR,
        MASTERDIR,
        OWrite=False,
        **kwargs
        ):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    DOINGDIR: pathlike
        The path to the original .
    summary : dataframe
        
    Returns
    -------
    
    '''
    DOINGDIR = Path(DOINGDIR)
    print (DOINGDIR.parts[-1])
    sMASTERDIR = DOINGDIR / master_dir
    REDUCEDDIR = DOINGDIR / reduced_dir
    REDUCNSKYDIR = DOINGDIR / reduced_nightsky_dir

    # if not MASTERDIR.exists():
    # shutil.copytree(MASTERDIR, MASTERDIR, dirs_exist_ok=True)

    if not sMASTERDIR.exists():
        os.makedirs(str(sMASTERDIR))
        print("{} is created...".format(str(sMASTERDIR)))

    if not REDUCNSKYDIR.exists():
        os.makedirs("{}".format(str(REDUCNSKYDIR)))
        print("{} is created...".format(str(REDUCNSKYDIR)))
    
    summary = yfu.make_summary(DOINGDIR/"*.fit*")
    if summary is not None :
        print("len(summary):", len(summary))
        print("summary:", summary)
        #print(summary["file"][0])   

        summary_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
        summary_light = summary_light.reset_index(drop=True) 

        for filt in ["V", "L", "R", "G", "B"]:
        #for filt in ["V"]:
            summary_light_filt = summary_light.loc[summary_light["FILTER"] == filt].copy()
            
            if summary_light_filt.empty:
                print("The dataframe(summary_light_filt) is empty")
                pass
            else:
                if (not (sMASTERDIR / f"nightskyflat-{filt}.fits").exists()) or OWrite==True :
                    print("len(summary_light_filt):", len(summary_light_filt))
                    print("summary_light_filt:", summary_light_filt)
                    
                    File_Num = 80
                    if len(summary_light_filt["file"]) > File_Num :
                        combine_lst = summary_light_filt["file"].tolist()[:File_Num]
                    else : 
                        combine_lst = summary_light_filt["file"].tolist()
                    try : 
                        ccd = yfu.imcombine(
                                            combine_lst, 
                                            combine="med",
                                            scale="avg", 
                                            scale_to_0th=False, 
                                            reject="sc", 
                                            sigma=2.5,
                                            verbose=True,
                                            memlimit = 2.e+11,
                                            )
                        ccd.write(sMASTERDIR / f"nightskyflat-{filt}.fits", overwrite=True)
                        print (f"Create Create nightskyflat-{filt}.fits +++...")
                    except :
                        ccd = yfu.imcombine(
                                            combine_lst, 
                                            combine="med",
                                            scale="avg", 
                                            scale_to_0th=False, 
                                            reject="sc", 
                                            # sigma=2.5,
                                            verbose=True,
                                            memlimit = 2.e+11,
                                            )
                        ccd.write(sMASTERDIR / f"nightskyflat-{filt}.fits", overwrite=True)
                        print (f"Create Create nightskyflat-{filt}.fits +++...")

        for _, row in summary_light.iterrows():
            fpath = Path(row["file"])
            filt = row["FILTER"]
            if (not (REDUCNSKYDIR/fpath.name).exists()) or OWrite==True :
                ccd = yfu.ccdred(
                                fpath, 
                                mflatpath=str(sMASTERDIR / f"nightskyflat-{filt}.fits"),
                                output=REDUCNSKYDIR/fpath.name,
                                )
                print (f"Reduce using nightskyflat {fpath.name} +++...")
    return 0
#%%    
# #########################################
# mag_inst
# #########################################
def mag_inst(flux, ferr):
    m_inst = -2.5 * np.log10(flux)
    merr   = 2.5/ np.log(10) * ferr / flux
    return m_inst, merr

def linf(x, a, b):
    return a + b*x
#%%
# #%%
# #########################################
# #solvingAstrometrynet
# #########################################
# def solvingAstrometrynet(        
#         DOINGDIR,
#         OWrite=False,
#         **kwargs
#         ):
#     ''' Gives the new path of the FITS file from header.
#     Parameters
#     ----------
#     DOINGDIR: pathlike
#         The path to the original .
#     summary : dataframe
        
#     Returns
#     -------
    
#     '''
#     ast = AstrometryNet()

#     # ger from nova.astrometry.net
#     ast.api_key = 'bldvwzzuvktnwfph' #must changed...
#     DOINGDIR = Path(DOINGDIR)
#     print("DOINGDIR", DOINGDIR)
#     if "RiLA600_STX-16803_" in str(DOINGDIR.parts[-2]) :
#         DOINGDIR = DOINGDIR / reduced_nightsky_dir
#     if "GSON300_STF-8300M_" in str(DOINGDIR.parts[-2]) :
#         DOINGDIR = DOINGDIR / reduced_dir
    
#     summary = yfu.make_summary(DOINGDIR/"*.fit*")
#     if summary is not None :
#         print("len(summary):", len(summary))
#         print("summary:", summary)
#         #print(summary["file"][0])  
#         df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
#         df_light = df_light.reset_index(drop=True)
#         print("df_light:\n{}".format(df_light))

#         for _, row  in df_light.iterrows():
#             fpath = Path(row["file"])
#             print(fpath)
#             hdul = fits.open(fpath)

#             submission_id = None
#             solve_timeout = 600

#             if 'PIXSCALE' in hdul[0].header:
#                 PIXc = hdul[0].header['PIXSCALE']
#             else : 
#                 PIXc = calPixScale(hdul[0].header['FOCALLEN'], 
#                                                     hdul[0].header['XPIXSZ'],
#                                                     hdul[0].header['XBINNING'])
#             print("PIXc : ", PIXc)
#             hdul.close()

#             SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
#             print("SOLVE:", SOLVE, "ASTAP:", ASTAP, "LOCAL:", LOCAL)

#             if SOLVE :
#                 print(f"{fpath.name} is already solved...")
#             else :             
#                 try_again = True                
                    
#                 while try_again:
#                     try:
#                         if not submission_id:
#                             wcs_header = ast.solve_from_image(str(fpath),
#                                                 force_image_upload=True,
#                                                 solve_timeout = solve_timeout,
#                                                 submission_id=submission_id)
#                         else:
#                             wcs_header = ast.monitor_submission(submission_id,
#                                                                 solve_timeout = solve_timeout)
#                     except TimeoutError as e:
#                         submission_id = e.args[1]
#                     else:
#                         # got a result, so terminate
#                         try_again = False

#                 if not wcs_header:
#                     # Code to execute when solve fails
#                     print("fits file solving failure...")

#                 else:
#                     # Code to execute when solve succeeds
#                     print("fits file solved successfully...")

#                     with fits.open(str(fpath), mode='update') as hdul:
#                         for card in wcs_header :
#                             try: 
#                                 print(card, wcs_header[card], wcs_header.comments[card])
#                                 hdul[0].header.set(card, wcs_header[card], wcs_header.comments[card])
#                             except : 
#                                 print(card)
#                         hdul.flush

#                     print(str(fpath)+" is created...")
#     return 0

#%%
#########################################
#diff_Photometry_PS1
#########################################
def diff_Photometry_PS1 (DOINGDIR,
                        tryagain = False,
                        LOCATION = None,
                        SKYC_KW = None,
                        FWHM_INIT = 6,
                        Mag_Low = 11.5,
                        Mag_High = 15,
                        Mag_target = 12.5,
                        Mag_delta = 2,
                        ERR_Min = 0.5,
                        READINGDIR = reduced_dir,
                        # READINGDIR =  reduced_nightsky_dir,
                    **kwarg
                    ) :
    from astropy.wcs import WCS
    from astropy.time import Time
    from astropy.nddata import Cutout2D
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    from photutils.aperture import CircularAperture as CAp
    from photutils.aperture import CircularAnnulus as CAn
    from photutils.aperture import aperture_photometry as apphot
    import seaborn as sns
    from scipy.optimize import curve_fit

    FWHM = FWHM_INIT
    R_AP = 1.5 * FWHM_INIT # Aperture radius
    R_IN = 4 * FWHM_INIT   # Inner radius of annulus
    R_OUT = 6 * FWHM_INIT 
    
    DOINGDIR = Path(DOINGDIR)
    print("DOINGDIR", DOINGDIR)

    READINGDIR = DOINGDIR / READINGDIR
    # READINGDIR = DOINGDIR / READINGDIR

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

                    SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
                    print(SOLVE, ASTAP, LOCAL)
                    
                    if SOLVE :
                        wcs = WCS(hdul[0].header)
                        # It is used as a rough estimate, so no need to be accurate:
                        #PIX2ARCSEC = 0.62*u.arcsec
                            
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
                        if 'PIXSCALE' in hdul[0].header:
                            PIX2ARCSEC = hdul[0].header['PIXSCALE']
                        else : 
                            PIX2ARCSEC = calPixScale(hdul[0].header['FOCALLEN'], 
                                                            hdul[0].header['XPIXSZ'],
                                                            hdul[0].header['XBINNING'])
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

                        im = zimshow(axs, hdul[0].data, )
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

                        im = zimshow(axs, hdul[0].data, )

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
                        poptm, _ = curve_fit(linf, df_phot_stars_na[f"{flt}mag"],
                                            df_phot_stars_na["mag"],
                                            sigma= df_phot_stars_na["merr"], absolute_sigma=True)
                        poptc, _ = curve_fit(linf, df_phot_stars_na["grcolor"],
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
                        axs[0, 1].plot(axs[0, 1].get_xlim(), linf(np.array(axs[0, 1].get_xlim()), *poptc),
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

                        axs[1, 1].plot(axs[1, 1].get_xlim(), linf(np.array(axs[1, 1].get_xlim()), *poptc),
                                    "g-", lw=1, label=f"$y = {{{poptc[1]:+.3f}}}x {{{poptc[0]:+.3f}}}$\n(curve_fit)")
                        axs[1, 1].legend(fontsize=8, loc='best')

                        axs[1, 2].plot(_xx, _xx + zeropt_med,
                                    label=f"${{{flt}}}_{{inst}} = {{{flt}}}_{{PS1}}$ {zeropt_med:+.03f}\n(median vlaue)",
                                    color="r", lw=1, ls="-")
                        axs[1, 2].plot(axs[1, 2].get_xlim(), linf(np.array(axs[1, 2].get_xlim()), *poptm),
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

                        im = zimshow(axs, hdul[0].data, )
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

                            msky, sky_std, nsky, nrej = sky_fit(sky_pixel, method='mode',
                                                                                mode_option='sex')


                            flux_star = apphot_result['aperture_sum'][i] - msky * ap_area  # total - sky

                            flux_err  = np.sqrt(apphot_result['aperture_sum'][i] * gain    # Poissonian (star + sky)
                                                + ap_area * rdnoise**2 # Gaussian
                                                + (ap_area * (gain * sky_std))**2 / nsky )

                            mag_ann[i], merr_ann[i] = mag_inst(flux_star, flux_err)
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
                print (f'Error messgae .......\n{err}')
                # _Python_utilities.write_log(err_log_file, err)
    return 0
#%%
#########################################
#checkAsteroids
#########################################
def checkAsteroids(DOINGDIR,
        summary,
        ):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    DOINGDIR: pathlike
        The path to the original .
    summary : dataframe
        
    Returns
    -------
    
    '''
    #####################################################################
    # Our object (will be queried to JPL HORIZONS)
    #OBJID = '2159' # 

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
    FWHM_INIT = 6

    # Photometry parameters
    R_AP = 1.5*FWHM_INIT # Aperture radius
    R_IN = 4*FWHM_INIT   # Inner radius of annulus
    R_OUT = 6*FWHM_INIT  # Outer radius of annulus

    Mag_UP = 17
    #######################################################

    ASTRESULTDIR = DOINGDIR / Asteroid_result_dir
    if not ASTRESULTDIR.exists():
        os.makedirs("{}".format(str(ASTRESULTDIR)))
        print("{} is created...".format(str(ASTRESULTDIR)))

    DOINGDIR = DOINGDIR / reduced_nightsky_dir

    summary_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
    summary_light = summary_light.reset_index(drop=True) 

    df_light = summary.loc[summary["IMAGETYP"] == "LIGHT"].copy()
    df_light = df_light.reset_index(drop=True)
    print("df_light:\n{}".format(df_light))

    for _, row  in df_light.iterrows():
        fpath = Path(row["file"])
        hdul = fits.open(fpath)

        SOLVE, ASTAP, LOCAL = checkPSolve(fpath)
        print(SOLVE, ASTAP, LOCAL)
        
        if SOLVE :
            wcs = WCS(hdul[0].header)
            # It is used as a rough estimate, so no need to be accurate:
            #PIX2ARCSEC = 0.62*u.arcsec
            if 'PIXSCALE' in hdul[0].header:
                PIX2ARCSEC = hdul[0].header['PIXSCALE']
            else : 
                PIX2ARCSEC = calPixScale(hdul[0].header['FOCALLEN'], 
                                                hdul[0].header['XPIXSZ'],
                                                hdul[0].header['XBINNING'])

            # D.2. Find the observation time and exposure time to set the obs time
            t_start = Time(hdul[0].header['DATE-OBS'], format='isot')
            t_expos = hdul[0].header['EXPTIME'] * u.s
            t_middle = t_start + t_expos / 2 # start time + 0.5 * exposure time
            #print(f"t_start: {t_start}, t_expos: {t_expos}, t_middle: {t_middle}")
            
            cent_coord = yfu.center_radec(ccd_or_header=hdul[0].header, 
                                        center_of_image=True)
            results_ast = Skybot.cone_search(cent_coord, 
                                            50*u.arcmin, 
                                            t_middle)
            #print(results_ast.pprint(max_width=80) )

            offset_RA = (cent_coord.ra.to(u.deg) - hdul[0].header['RA']*u.deg).to(u.arcmin)
            offset_DEC = (cent_coord.dec.to(u.deg) - hdul[0].header['DEC']*u.deg).to(u.arcmin) 
            altaz = AltAz(obstime=t_middle, location=Suwon)   
            cent_aa = cent_coord.transform_to(altaz)
            offset_AZ = (cent_aa.az.to(u.deg) - hdul[0].header['CENTAZ']*u.deg).to(u.arcmin)
            offset_ALT = (cent_aa.alt.to(u.deg) - hdul[0].header['CENTALT']*u.deg).to(u.arcmin)
            
            df_ast = results_ast.to_pandas()
            df_ast

            df_targ = df_ast[df_ast['V'] < Mag_UP]
            df_targ = df_targ.sort_values(by=['V'])
            df_targ = df_targ.reset_index(drop=True)
            df_targ

            if df_targ.empty:
                pass
            else:
                df_targ_eph = pd.DataFrame()

                for i, row in df_targ.iterrows() :
                    try : 
                        #print("type(row)", type(row))
                        #Query the ephemerides of this target! 
                        obj = Horizons(id=row['Number'], 
                                    location=observatory_code, 
                                    epochs=t_middle.jd)
                        obj_ephem = obj.ephemerides()
                        #print(obj_ephem)
                        df_eph = obj_ephem.to_pandas()
                        df_targ_eph = pd.concat([df_targ_eph, df_eph], axis = 0)
                    except : 
                        continue

                #print(df_targ_eph)
                df_targ_eph = df_targ_eph.reset_index(drop=True)
                df_targ_eph = pd.concat([df_targ, df_targ_eph], axis = 1)
                print("df_targ_eph :", df_targ_eph)

                duplicated_columns_list = []
                list_of_all_columns = list(df_targ_eph.columns)
                for column in list_of_all_columns:
                    if list_of_all_columns.count(column) > 1 and not column in duplicated_columns_list:
                        duplicated_columns_list.append(column)
                duplicated_columns_list

                for column in duplicated_columns_list:
                    list_of_all_columns[list_of_all_columns.index(column)] = column
                    list_of_all_columns[list_of_all_columns.index(column)] = column + '_1'

                df_targ_eph.columns = list_of_all_columns
                #print(df_targ_eph.columns)
                df_targ_eph.to_csv(f"{ASTRESULTDIR/fpath.stem}_AST_Mag{Mag_UP}.csv")
                df_targ_eph.dropna(subset = ['RA', 'DEC', 'V', 'RA_1', 'DEC_1', 'V_1'], inplace=True)
                df_targ_eph[['RA', 'DEC', 'V', 'RA_1', 'DEC_1', 'V_1']]
                
                # RADEC_targ = np.array([df_targ_eph['RA'], df_targ_eph["DEC"]]).T
                # RADEC_targ
                # pos_targ_init = SkyCoord(RADEC_targ, 
                #         **SKYC_KW).to_pixel(wcs, origin=1, mode='wcs')
                # print("pos_targ_init:", pos_targ_init)

                pos_targ_init = SkyCoord(df_targ_eph['RA']*u.deg, df_targ_eph["DEC"]*u.deg, 
                                        **SKYC_KW).to_pixel(wcs, origin=0, mode='wcs')
                pos_targ_init = np.array(pos_targ_init).T
                print("pos_targ_init:", pos_targ_init)

            if hdul[0].header['CCDNAME'] == 'STF-8300M' :
                val_figsize = (13, 5.2)
                val_fraction = 0.035
            if hdul[0].header['CCDNAME'] == 'STX-16803' :
                val_figsize=(12, 6.2)
                val_fraction = 0.0455

            fig_set = plt.figure(figsize=val_figsize)
            ax1 = plt.subplot2grid((1,2), (0,0),
                                fig=fig_set)
            im1 = zimshow(ax1, hdul[0].data, )
            ax1.set_title('Pixel coordinate system', fontsize=9)
            ax1.tick_params(labelsize=8)
            plt.colorbar(im1, ax = ax1, fraction=val_fraction, pad=0.04)

            ax2 = plt.subplot2grid((1,2), (0,1),
                                projection=wcs,
                                fig=fig_set)
            im2 = zimshow(ax2, hdul[0].data, )
            ax2.set_title('World coordinate system', fontsize=9)
            ax2.coords.grid(True, color='white', ls=':')
            ax2.coords['ra'].set_axislabel('Right Ascension (J2000)', minpad=0.5, fontsize=8)
            ax2.coords['ra'].set_ticklabel_position('bl')
            ax2.coords['dec'].set_axislabel('Declination (J2000)', minpad=0.4, fontsize=8)
            ax2.coords['dec'].set_ticklabel_position('bl')
            ax2.coords['ra'].set_major_formatter('hh:mm')
            ax2.coords['dec'].set_major_formatter('dd:mm')
            ax2.coords['ra'].display_minor_ticks(True)
            ax2.coords['dec'].display_minor_ticks(True)
            ax2.coords['ra'].set_minor_frequency(1)
            ax2.coords['dec'].set_minor_frequency(1)
            ax2.tick_params(labelsize=8)

            if df_targ.empty:
                pass
            else:
                targ_ap = CAp(pos_targ_init,
                        r=R_AP, 
                        )
                targ_an = CAn(pos_targ_init,
                        r_in=R_IN,
                        r_out=R_OUT)
                
                #targ_ap.plot(ax1, color="r")
                targ_an.plot(ax1, color="r")
                #targ_ap.plot(ax2, color="r")
                targ_an.plot(ax2, color="r")

                ax1.annotate(f"{pos_targ_init}",
                        xy=(0, 0), xytext=(0, -0.1),
                        xycoords='axes fraction',
                        va='top', ha='left',
                        fontsize=7)

                ax2.annotate(f"{df_targ_eph[['Number', 'RA', 'DEC', 'V']]}",
                        xy=(0, 0), xytext=(0, -0.1),
                        xycoords='axes fraction',
                        va='top', ha='left',
                        fontsize = 6)
            plt.colorbar(im2, ax = ax2, fraction=val_fraction, pad=0.04)
            plt.suptitle(f"fname: {fpath.name}")

            ax2.annotate(f"image center (RA, DEC): ({cent_coord.ra:.03f}, {cent_coord.dec:.03f})\ntelescope center (RA, DEC): ({hdul[0].header['RA']*u.deg:.03f}, {hdul[0].header['RA']*u.deg:.03f})\noffset (RA, DEC): ({offset_RA:.03f}, {offset_DEC:.03f})\noffset (AZ, ALT): ({offset_AZ:.03f}, {offset_ALT:.03f})",
                        xy=(0, 0), xytext=(0.6, -0.1),
                        xycoords='axes fraction',
                        va='top', ha='left',
                        fontsize = 6)
            plt.tight_layout()
            plt.savefig(f"{ASTRESULTDIR/fpath.stem}_AST_Mag{Mag_UP}.png")

            if df_targ.empty:
                pass
            else:
                cutsizes = 49
                for i, row in df_targ_eph.iterrows():
                
                    #1. cut asteroia area
                    #print(i)
                    cut_hdu = Cutout2D(
                                data = hdul[0].data,
                                position = (pos_targ_init[i]),
                                size=(cutsizes, cutsizes) #cut ccd
                                )
                    avg, med, std = sigma_clipped_stats(cut_hdu.data)  # by default, 3-sigma 5-iteration.

                    fig_set = plt.figure(figsize=(8, 5.5))
                    
                    ax11 = plt.subplot2grid((2, 2), (0,0),
                                fig=fig_set)
                    im11 = zimshow(ax11, cut_hdu.data)
                    ax11.plot(round(cutsizes/2), round(cutsizes/2), 'rx')
                    ax11.set_ylabel('pixels')
                    ax11.grid(ls=':')
                    ax11.set_title(f'Asteroid area image', fontsize=9)
                    ax11.annotate(   f"mean: {np.mean(cut_hdu.data):.01f}, std: {np.std(cut_hdu.data):.01f} \nmax: {np.max(cut_hdu.data):.01f}, min: {np.min(cut_hdu.data):.01f} \nNumber of Pixel: {np.shape(cut_hdu.data)[0]:.0f}x{np.shape(cut_hdu.data)[1]:.0f}",
                        xy=(0, 0), xytext=(0.1, -0.20),
                        xycoords='axes fraction',
                        va='top', ha='left',
                        fontsize=8)
                    plt.colorbar(im11,
                                ax=ax11,
                                label="ADU",
                                fraction=0.0455, pad=0.04)
                    #print("Image size is: ", cut_hdu.data.shape)

                    #2. Get center dx, dy
                    thresh_3sig = med + 3 * std
                    mask_3sig = (cut_hdu.data < thresh_3sig)
                    center = centroid_com(
                                data = cut_hdu.data,
                                mask = mask_3sig
                                )

                    centerdx = center[0] - ((cutsizes+1)/2)
                    centerdy = center[1] - ((cutsizes+1)/2)
                    # print("type(center):", type(center))
                    # print("center:", center)
                    # print("center dx, dy:", centerdx, centerdy)

                    ax12 = plt.subplot2grid((2,2), (0,1),
                                fig=fig_set)
                    ax12.grid(ls=':')
                    ax12.set_title(f'The new center of asteroid', fontsize=9)
                    im12 = ax12.imshow(mask_3sig.astype(int),
                        origin="lower")
                    im12 = ax12.imshow(cut_hdu.data,
                            alpha=0.4,
                            origin="lower")
                    ax12.plot(*center, 'rx')
                    ax12.annotate(f"center: {center[0]:.02f}, {center[1]:.02f}\ncenter dx, dy: {centerdx:.02f}, {centerdy:.02f}",
                            xy=(0, 0), xytext=(0.01, -0.20),
                            xycoords='axes fraction',
                            va='top', ha='left',
                            fontsize=8)
                    
                    ax11.annotate(f"asteroid No.{i}: {row['targetname']}, \n{row['datetime_str']}, V_mag {row['V']}",
                            xy=(1, 0), xytext=(-0.1, 1.33),
                            xycoords='axes fraction',
                            va='top', ha='left',
                            fontsize=8)
                    
                    plt.colorbar(im12,
                                ax=ax12,
                                label="ADU",
                                fraction=0.0455, pad=0.04)
                    plt.suptitle(f"{fpath.name}", 
                                fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(f"{ASTRESULTDIR/fpath.stem}_AST_Mag{Mag_UP}_{i:02d}.png")
                    #plt.show()
                    plt.close()

    return 0


#%%                                
def connectMariaDB():
    #import pymysql
    import pymysql.cursors
    #conda install pymysql
    
    #mariaDB info
    db_host = 'parksparks.iptime.org'
    db_user = 'root'
    db_pass = 'rlgusl01'
    db_name = 'CCD_obs'
    db_port = 3307
        
    conn = pymysql.connect(host = db_host,
                          port = db_port,
                          user = db_user, password = db_pass,
                          db = db_name, charset = 'utf8mb4',
                          cursorclass = pymysql.cursors.DictCursor)
    
    return conn


#%%
def subp_solve_field(fullname, save_dir_name, sub_start_time): 
    import subprocess
    print('-'*60)
    print(fullname)
    with subprocess.Popen(['solve-field', 
                           '-O', #--overwrite: overwrite output files if they already exist
                           #'--scale-units', 'arcsecperpix', #pixel scale
                           #'--scale-low', '0.1', '--scale-high', '0.40', #pixel scale
                           '-g', #--guess-scale: try to guess the image scale from the FITS headers
                           #'-p', # --no-plots: don't create any plots of the results
                           '-D', '{0}'.format(save_dir_name), 
                           '{0}'.format(fullname)], 
                          stdout=subprocess.PIPE) as proc :
        print(proc.stdout.read())

        '''
        solve-field -O fullname
       '''
    return 0