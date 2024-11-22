#!/bin/sh
# PROJECT="03-EXO";
# YRMN="2024-08" ;
# OPTIC="GSON300";
# CCDNAME="STF-8300M";
# BIN="1bin" ; 

# # for BDF in "BIAS" "DARK" ;
# # do 
# #     find /mnt/Rdata/OBS_data/CCD_obs_raw/${CCDNAME}_${BIN}/Cal -type f -name "-_${BDF}*${YRMN}*.fit*" -exec cp -f "{}" "/mnt/Rdata/OBS_data/${PROJECT}/-_-_-_${YRMN}_-_${OPTIC}_${CCDNAME}_-_${BIN}/-_CAL-BDF_-_${YRMN}_-_${OPTIC}_${CCDNAME}_-_${BIN}/" \;
# # done ;
# # find /mnt/Rdata/OBS_data/CCD_obs_raw/${CCDNAME}_${BIN}/Cal_${OPTIC} -type f -name "-_FLAT*${YRMN}*.fit*" -exec cp -f "{}" "/mnt/Rdata/OBS_data/${PROJECT}/-_-_-_${YRMN}_-_${OPTIC}_${CCDNAME}_-_${BIN}/-_CAL-BDF_-_${YRMN}_-_${OPTIC}_${CCDNAME}_-_${BIN}/" \;

# find /mnt/Rdata/OBS_data/CCD_obs_raw/${CCDNAME}_${BIN}/Cal_${OPTIC} -type f -name "-_FLAT*${YRMN}*.fit*" -exec cp -f "{}" "/mnt/Rdata/OBS_data/${PROJECT}/-_-_-_2024-09_-_${OPTIC}_${CCDNAME}_-_${BIN}/-_CAL-BDF_-_2024-09_-_${OPTIC}_${CCDNAME}_-_${BIN}/" \;

# rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/01-Variable' '/mnt/Rdata/OBS_data/' 
# rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/02-Asteroid' '/mnt/Rdata/OBS_data/' 
# rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/03-EXO' '/mnt/Rdata/OBS_data/'
# rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/04-Spectra' '/mnt/Rdata/OBS_data/'
# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/2024-OA' '/mnt/Rdata/OBS_data/'


# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/01-Variable' '/mnt/Rdata/OBS_data/' 
# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/02-Asteroid' '/mnt/Rdata/OBS_data/' 
# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/03-EXO' '/mnt/Rdata/OBS_data/'
# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/04-Spectra' '/mnt/Rdata/OBS_data/'
# rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/2024-OA' '/mnt/Rdata/OBS_data/'

# BASEDIR = "/mnt/Rdata/OBS_data"
# PROJECTDIR = "01-Variable

# for dir_list in "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
# do
#         rsync -avuz --progress --rsh='ssh -p2022' "guitar79@parksparks.iptime.org:/volume1/Rdata/OBS_data/01-Variable/$dir_list" "/mnt/Rdata/OBS_data/01-Variable/"
# done

# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.wcs' -delete
# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.rdls' -delete
# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.asy' -delete
# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.corr' -delete
# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.match' -delete
# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f -name '*.solved' -delete

# find /mnt/Rdata/OBS_data/CCD_obs_raw/ -type f \( -name '*.wcs' -o -name '*.rdls' -o -name '*.asy' -o -name '*.corr' -o -name '*.match' -o -name '*.soled' \) -delete

# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23116최현준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/23074오서준/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23067신재헌/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22018김한준/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23097임윤준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22004권민우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23054박하람/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22095정은재/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22029박주영/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22022노현우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22115최준서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22073이재욱/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22032박지환/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/21100정영우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22028박성환/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22045양현서/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22106조형준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22050오태원/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22027박성민/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22125홍은찬/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22030박지원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22093정우현/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22098정이찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22110최석원/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22024문준원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22035박홍준/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22072이재우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22053용승주/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22080이혁준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22107지민기/

# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/23074오서준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23116최현준/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22018김한준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23067신재헌/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22004권민우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23097임윤준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22095정은재/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/23054박하람/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22022노현우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22029박주영/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22073이재욱/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22115최준서/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/21100정영우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22032박지환/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22045양현서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22028박성환/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22050오태원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22106조형준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22125홍은찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22027박성민/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22093정우현/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22030박지원/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22110최석원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22098정이찬/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22035박홍준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22024문준원/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22053용승주/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22072이재우/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22107지민기/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/22080이혁준/ 

# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/23074오서준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/23075오승민/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22018김한준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/23027김재우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22004권민우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/23108조형석/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22095정은재/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/23069안선우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22022노현우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22005권순민/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22073이재욱/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22008김도현/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/21100정영우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22039손희원/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22045양현서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22088장태훈/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22050오태원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22012김수아/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22125홍은찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22034박현수/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22093정우현/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22082임비건/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22110최석원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22048오은총/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22035박홍준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22069이은우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22053용승주/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22103조연우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22107지민기/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22121함석규/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/22118최현진/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/3반/22108차무겸/




# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/23074오서준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/23116최현준/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22018김한준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/23067신재헌/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22004권민우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/23097임윤준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22095정은재/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/23054박하람/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22022노현우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22029박주영/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22073이재욱/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22115최준서/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/21100정영우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22032박지환/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22045양현서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22028박성환/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22050오태원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22106조형준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22125홍은찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22027박성민/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22093정우현/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22030박지원/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22110최석원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22098정이찬/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22035박홍준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22024문준원/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22053용승주/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22072이재우/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22107지민기/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/22080이혁준/ 

# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/23074오서준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23075오승민/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22018김한준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23027김재우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22004권민우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23108조형석/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22095정은재/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23069안선우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22022노현우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22005권순민/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22073이재욱/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22008김도현/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/21100정영우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22039손희원/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22045양현서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22088장태훈/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22050오태원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22012김수아/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22125홍은찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22034박현수/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22093정우현/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22082임비건/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22110최석원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22048오은총/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22035박홍준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22069이은우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22053용승주/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22103조연우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22107지민기/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22121함석규/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22118최현진/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22108차무겸/


# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23075오승민/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/23074오서준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23027김재우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22018김한준/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23108조형석/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22004권민우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22095정은재/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/23069안선우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22022노현우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22005권순민/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22073이재욱/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22008김도현/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/21100정영우/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22039손희원/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22045양현서/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22088장태훈/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22050오태원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22012김수아/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22125홍은찬/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22034박현수/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22093정우현/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22082임비건/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22110최석원/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22048오은총/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22035박홍준/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22069이은우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22053용승주/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22103조연우/
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22121함석규/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22107지민기/ 
# rsync -avuz --progress --delete  /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/3반/22108차무겸/* /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/22118최현진/





# for NNAME in "23116최현준" "23067신재헌" "23097임윤준" "23054박하람" "22029박주영" "22115최준서" "22032박지환" "22028박성환" "22106조형준" "22027박성민" "22030박지원" "22098정이찬" "22024문준원" "22072이재우" "22080이혁준";
# for NNAME in "23074오서준" "22018김한준" "22004권민우" "22095정은재" "22022노현우" "22073이재욱" "21100정영우" "22045양현서" "22050오태원" "22125홍은찬" "22093정우현" "22110최석원" "22035박홍준" "22053용승주" "22107지민기" "22118최현진";

# do 
#     # rsync -avuz --progress /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/1반/${NNAME}/ /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/1반/${NNAME}/ ;
#     rsync -avuz --progress /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/${NNAME}/ /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/${NNAME}/ ;
# done ;

# rsync -avuz --progress /mnt/Rdata/OBS_data/2024-OA/_측광과제-교사참고/2반/ /mnt/Rdata/OBS_data/2024-OA/_측광과제-학생수행/2반/ ;


