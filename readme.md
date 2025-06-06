
# ys bach
https://github.com/ysBach/SNU_AOclass/blob/master/Notebooks/04-Aperture_Phot_01.ipynb

# first time
cd ~/Downloads/ && git clone https://github.com/ysBach/ysvisutilpy && cd ysvisutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && rm -rf ysfitsutilpy && git clone https://github.com/ysBach/ysfitsutilpy && cd ysfitsutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && git clone https://github.com/ysBach/ysphotutilpy && cd ysphotutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ && git clone https://github.com/ysBach/SNUO1Mpy && cd SNUO1Mpy && git pull && pip install -e . && cd ..


//10.10.10.22/homes     /mnt/homes      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/Pdrive     /mnt/Pdrive      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/Rdata     /mnt/Rdata      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/photo     /mnt/photo      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0



# second time...
cd ~/Downloads/ysvisutilpy && git pull && pip install -e . 
cd ~/Downloads/ysfitsutilpy && git pull && pip install -e . 
cd ~/Downloads/ysphotutilpy && git pull && pip install -e . 
cd ~/Downloads/SNUO1Mpy && git pull && pip install -e . 

//10.10.10.22/homes     /mnt/homes      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/Pdrive     /mnt/Pdrive      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/Rdata     /mnt/Rdata      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0
//10.10.10.22/photo     /mnt/photo      cifs    credentials=/home/guitar79/.credentials,uid=1000,gid=1000,user  0       0

sudo nano .credentials 
username=guitar79
password=Pkh19255102@


pip install ysfitsutilpy==0.2
#pip install ysfitsutilpy==0.2
pip install ysphotutilpy==0.1


find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/ -type f -name '*.wcs' -delete

find '/mnt/Rdata/ASTRO_data/2024-OA/_측광과제-교사참고/1반/23067신 재헌/HD189733b_LIGHT_-_2024-06-27_-_RiLA600_STX-16803_-_1bin' -type f -name '*.fit' -exec solve-field -O --downsample 6 --cpulimit 20 "{}" \;

find '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin/AH-CAM_LIGHT_-_2022-01-23_-_RiLA600_STX-16803_-_2bin' -type f -name '*.fit' -exec solve-field -O --downsample 4 --cpulimit 20 "{}" \;

find '/mnt/Rdata/ASTRO_data/C3-EXO/-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin/Kelt-1b_LIGHT_-_2024-11-23_-_RiLA600_ASI6200MMPro_-_3bin/' -type f -name '*.fit' -exec solve-field -O --downsample 4 --cpulimit 20 "{}" \;

find '/mnt/Rdata/ASTRO_data/C3-EXO/-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin/WASP-180Ab_LIGHT_-_2025-01-02_-_GSON300_STF-8300M_-_1bin' -type f -name '*.fit' -exec solve-field -O --downsample 4 --cpulimit 20 --nsigma 15 "{}" \;

find '/mnt/Rdata/ASTRO_data/C3-EXO/-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin/HAT-P-61b_LIGHT_-_2025-01-02_-_RiLA600_ASI6200MMPro_-_3bin/' -type f -name '*.fit' -exec solve-field -O --downsample 4 --cpulimit 20 --nsigma 15 "{}" \;


find '/mnt/Rdata/ASTRO_data/C5-Test/-_-_-_-_GSON300_STF-8300M_-_1bin' -type f -name '*.fit' -exec solve-field -O --downsample 4 --nsigma 15 --cpulimit 20 "{}" \;

find '/mnt/Rdata/ASTRO_data/C5-Test/-_-_-_-_GSON300_STF-8300M_-_1bin' -type f -name '*.fit' -exec astap -f "{}" -wcs -analyse2 -update \;

find '/mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STF-8300M_1bin/LIGHT_FS60CB/' -type f -name '*.fit' -exec astap -f "{}" -wcs -analyse2 -update \;

astap -f {str(fpath)} -fov {hfov} -z 2 -wcs -analyse2 -update"




find /mnt/Rdata/ASTRO_data/C1-Variable/ -type f \( -name '*.wcs' -o -name '*.rdls' -o -name '*.asy' -o -name '*.corr' -o -name '*.match' -o -name '*.soled' -o -name '*.axy' -o -name '*PS1_query.png' \) -delete

find /mnt/Rdata/ASTRO_data/C2-Asteroid/ -type f \( -name '*.wcs' -o -name '*.rdls' -o -name '*.asy' -o -name '*.corr' -o -name '*.match' -o -name '*.soled' -o -name '*.axy' -o -name '*PS1_query.png' \) -delete

find /mnt/Rdata/ASTRO_data/C3-EXO/ -type f \( -name '*.wcs' -o -name '*.rdls' -o -name '*.asy' -o -name '*.corr' -o -name '*.match' -o -name '*.soled' -o -name '*.axy' -o -name '*PS1_query.png' \) -delete

yrmn = "2017-05" ;
find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/ASI6200MMPro_3bin/Cal -type f -name '-_BIAS*2024-11*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C3-EXO/-_-_-_2024-11_-_RiLA600_ASI6200MMPro_-_3bin/-_CAL-BDF_-_2024-11_-_GSON300_STF-8300M_-_3bin/' \;

find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STX-16803_2bin/Cal -type f -name '-_BIAS*2017-05*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin/CAL-BDF_-_-_2017-05_-_RiLA600_STX-16803_-_2bin/' \;

find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STX-16803_2bin/Cal -type f -name '-_DARK*2021-10*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/CAL-BDF_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/' \;

find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STX-16803_2bin/Cal_RiLA600  -type f -name '-_FLAT*2021-10*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/CAL-BDF_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/' \;

find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STX-16803_2bin/Cal_RiLA600  -type f -name '-_FLAT*2021-10*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/CAL-BDF_-_-_2021-10_-_RiLA600_STX-16803_-_2bin/' \;

find /mnt/Rdata/ASTRO_data/A3_CCD_obs_raw/STF-8300M_1bin/Cal -type f -name '-_BIAS*2024-11*.fit*' -exec cp "{}" '/mnt/Rdata/ASTRO_data/C3-EXO/-_-_-_2024-11_-_GSON300_STF-8300M_-_1bin/-_CAL-BDF_-_2024-11_-_GSON300_STF-8300M_-_1bin/' \;

sudo swapoff -a; sudo swapon -a

sudo apt install rename

rename 's/\.new$/.fit/' *.new

Precision
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C2-Asteroid' '/mnt/Rdata/ASTRO_data/' 
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C3-EXO/' '/mnt/Rdata/ASTRO_data/'



<!-- PROJECDIR = BASEDIR / "C1-Variable"
TODODIR = PROJECDIR / "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-03_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin"
TODODIR = PROJECDIR / "-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin"  -->

# PROJECDIR = BASEDIR / "C2-Asteroid"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2022-_-_RiLA600_STX-16803_-_2bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2023-_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C3-EXO"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-05_-_RiLA600_STX-16803_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_GSON300_STF-8300M_-_1bin"
# TODODIR = PROJECDIR / "-_-_-_2024-06_-_RiLA600_STX-16803_-_2bin"

# PROJECDIR = BASEDIR / "C4-Spectra"
# TODODIR = PROJECDIR / "-_-_-_2024-05_TEC140_ASI183MMPro_-_1bin"

#!/bin/sh
BASEDIR = "/mnt/Rdata/ASTRO_data"
PROJECTDIR = "C1-Variable"

for dir_list in "-_-_-_2016-_-_RiLA600_STX-16803_-_2bin"
do
    rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/${PROJECTDIR}/$dir_list' '/mnt/Rdata/ASTRO_data/${PROJECTDIR}/' 
done

#!/bin/sh
mnt_name="/mnt/Rdata/"

for dir_list in "FB107" "NMSC" "MODIS_AOD"
do
    dir_name="${mnt_name}$dir_list"
    if [ ! -e ${dir_name} ]; then
        sudo mkdir ${dir_name}
    elif [ ! -d ${dir_name} ]; then
        echo "${dir_name} already exists but is not a directory."
    else
        echo "${dir_name} already exists AND is a directory."
    fi

    sudo mount.cifs //10.114.0.120/${dir_list}/ $dir_name -o user=guitar79,pass=Pkh19255102@,uid=1000,gid=1000
done



rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable' '/mnt/Rdata/ASTRO_data/' 
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C2-Asteroid' '/mnt/Rdata/ASTRO_data/' 
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C3-EXO' '/mnt/Rdata/ASTRO_data/'
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C4-Spectra' '/mnt/Rdata/ASTRO_data/'
rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/2024-OA' '/mnt/Rdata/ASTRO_data/'

rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable' '/mnt/Rdata/ASTRO_data/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C2-Asteroid' '/mnt/Rdata/ASTRO_data/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C3-EXO' '/mnt/Rdata/ASTRO_data/'
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C4-Spectra' '/mnt/Rdata/ASTRO_data/'
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/2024-OA' '/mnt/Rdata/ASTRO_data/'


rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/-_-_-_2016-_-_RiLA600_STX-16803_-_2bin' '/mnt/Rdata/ASTRO_data/C1-Variable/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin' '/mnt/Rdata/ASTRO_data/C1-Variable/' 

rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/2024-OA/' '/mnt/Rdata/ASTRO_data/'

Wool
rsync -avuz --progress --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/'

rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/' && rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/-_-_-_2017-05_-_RiLA600_STX-16803_-_2bin' '/mnt/Rdata/ASTRO_data/C1-Variable/' && rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/-_-_-_2017-06_-_RiLA600_STX-16803_-_2bin' '/mnt/Rdata/ASTRO_data/C1-Variable/' && rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2017-01_-_RiLA600_STX-16803_-_2bin' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/' && 
rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2021-10_-_RiLA600_STX-16803_-_2bin' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable/-_-_-_2022-01_-_RiLA600_STX-16803_-_2bin' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable/' 


rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C1-Variable' 'O:\/mnt/Rdata/ASTRO_data/' 

rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C2-Asteroid' '/mnt/Rdata/ASTRO_data/' 

&& rsync -avuz --progress --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C3-EXO' '/mnt/Rdata/ASTRO_data/'
rsync -avuz --progress --delete --rsh='ssh -p2022' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/C4-Spectra' '/mnt/Rdata/ASTRO_data/'

rsync -avuz --progress --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C1-Variable' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C2-Asteroid' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C3-EXO' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/' 
rsync -avuz --progress --delete --rsh='ssh -p2022' '/mnt/Rdata/ASTRO_data/C4-Spectra' 'guitar79@parksparks.iptime.org:/volume1/Rdata/ASTRO_data/' 




