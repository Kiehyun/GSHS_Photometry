# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018
@author: guitar79@naver.com
"""
#%%
import os
from astropy.io import fits
import _Python_utilities

log_dir = "logs/"
log_file = "{}{}.log".format(log_dir, os.path.basename(__file__)[:-3])
err_log_file = "{}{}_err.log".format(log_dir, os.path.basename(__file__)[:-3])
print ("log_file: {}".format(log_file))
print ("err_log_file: {}".format(err_log_file))
if not os.path.exists('{0}'.format(log_dir)):
    os.makedirs('{0}'.format(log_dir))

#%%
#########################################
# mariaDB info
#########################################
import pymysql
db_host = '192.168.0.20'
#db_host = '10.114.0.120'

db_user = 'modis'
db_pass = 'Modis12345!'
db_name = 'MODIS_Aerosol'

db_user = 'FB107'
db_pass = 'Gses12345!'
db_name = 'FB107'

tb_name = 'FITS_file_info'

conn = pymysql.connect(host=db_host, port=3306,
                       user=db_user, password=db_pass, db=db_name,
                       charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

cur = conn.cursor()

qry = "SHOW TABLES FROM `{}` LIKE '{}'".format(db_name, tb_name)

table_exist = cur.execute(qry)

if table_exist == 0:
    qry = """CREATE TABLE IF NOT EXISTS `{}`.`{}` (
        `id` INT NOT NULL AUTO_INCREMENT ,
        `fullname` VARCHAR(16384) NULL default NULL ,
        `DATE-OBS` VARCHAR(28) NULL DEFAULT NULL ,
        `IMAGETYP` VARCHAR(8) NULL DEFAULT NULL ,
        `OBJECT` VARCHAR(20) NULL DEFAULT NULL ,
        `FOCALLEN` VARCHAR(10) NULL DEFAULT NULL ,
        `XPIXSZ` VARCHAR(10) NULL DEFAULT NULL ,
        `PIXSCALE` VARCHAR(10) NULL DEFAULT NULL, 
        `OBJCTRA` VARCHAR(10) NULL DEFAULT NULL ,      
        `OBJCTDEC` VARCHAR(10) NULL DEFAULT NULL ,          
        `OBJCTALT` VARCHAR(10) NULL DEFAULT NULL ,             
        `OBJCTAZ` VARCHAR(10) NULL DEFAULT NULL ,              
        `OBJCTHA` VARCHAR(10) NULL DEFAULT NULL ,   
    	`EXPTIME` VARCHAR(20) NULL DEFAULT NULL ,
        `Update-DT` TIMESTAMP on update CURRENT_TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP , 
        PRIMARY KEY (`id`)) ENGINE = InnoDB;""".format(db_name, tb_name)
    cur.execute(qry)
    conn.commit()
else :
    qry = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}';".format(db_name, tb_name)
    cur.execute(qry)
    column_dic = cur.fetchall()
    print(column_dic)

    #ALTER TABLE `FITS_file_info` ADD `CTYPE1` VARCHAR(16) NULL DEFAULT NULL AFTER `OBJCTHA`, ADD `CTYPE2` VARCHAR(16) NULL DEFAULT NULL AFTER `CTYPE1`, ADD `EQUINOX` VARCHAR(10) NULL DEFAULT NULL AFTER `CTYPE2`, ADD `LONPOLE` VARCHAR(10) NULL DEFAULT NULL AFTER `EQUINOX`, ADD `LATPLOE` VARCHAR(16) NULL DEFAULT NULL AFTER `LONPOLE`, ADD `CRVAL1` VARCHAR(16) NULL DEFAULT NULL AFTER `LATPLOE`, ADD `CRVAL2` VARCHAR(16) NULL DEFAULT NULL AFTER `CRVAL1`, ADD `CRPIX1` VARCHAR(16) NULL DEFAULT NULL AFTER `CRVAL2`, ADD `CRPIX2` VARCHAR(16) NULL DEFAULT NULL AFTER `CRPIX1`, ADD `CUNIT1` VARCHAR(16) NULL DEFAULT NULL AFTER `CRPIX2`, ADD `CUNIT2` VARCHAR(16) NULL DEFAULT NULL AFTER `CUNIT1`, ADD `CD1_1` VARCHAR(16) NULL DEFAULT NULL AFTER `CUNIT2`, ADD `CD1_2` VARCHAR(16) NULL DEFAULT NULL AFTER `CD1_1`, ADD `CD2_1` VARCHAR(16) NULL DEFAULT NULL AFTER `CD1_2`, ADD `CD2_2` VARCHAR(16) NULL DEFAULT NULL AFTER `CD2_1`;



#%%
#########################################
BASEDIRs = ['../A3_CCD_obs_raw/']

fullnames = []
for dirName in BASEDIRs :
    try :
        fullnames.extend(_Python_utilities.getFullnameListOfallFiles("{}".format(dirName)))
    except Exception as err :
        #_Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
        print(err)
        continue
fullnames = sorted(fullnames)
print("fullnames: {}".format(fullnames))

#%%
n = 0

for fullname in fullnames :
    n += 1
    print('#'*40,
        "\n{2:.01f}%  ({0}/{1}) {3}".format(n, len(fullnames), (n/len(fullnames))*100, os.path.basename(__file__)))
    print ("Starting...   fullname: {}".format(fullname))

    #fullname = fullnames[10]
    if fullname[-4:].lower() == ".fit":
        fullname_el = fullname.split("/")
        filename_el = fullname_el[-1].split("_")
        hdul = fits.open(fullname)

        if 'DATE-OBS' in hdul[0].header \
            and 'EXPTIME' in hdul[0].header:
            try:
                qry = """SELECT `id` FROM `{}`.`{}` WHERE `fullname`= '{}';""".format(db_name, tb_name, fullname)
                check_row = cur.execute(qry)
                print("check_row: {}".format(check_row))

                if check_row == 0:
                    qry = """INSERT INTO `{0}`.`{1}`                                         
                            (`fullname`,  `DATE-OBS`, `EXPTIME`) 
                            VALUES ('{2}', '{3}', '{4}');""".format(db_name, tb_name,
                                    fullname, hdul[0].header['DATE-OBS'], hdul[0].header['EXPTIME'])
                    print("qry: {}".format(qry))

                else:
                    qry = """SELECT * 
                            FROM `{0}`.`{1}` 
                            WHERE `{1}`.`fullname` = {2};""".format(db_name, tb_name, fullname)
                    print("qry: {}".format(qry))
                cur.execute(qry)
                conn.commit()
            except Exception as err :
                #_Python_utilities.write_log(err_log_file, f'''{fpath}, {str(err)}''', verbose=verbose)
                print("err with {}".format(fullname))
                continue