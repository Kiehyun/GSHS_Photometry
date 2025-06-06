
# astro_Python
anaconda environment

## 가상환경 리스트
conda env list

## 가상환경 만들기
conda create -n astro_Python_env python=3.10

## activate 가상환경 시작
conda activate astro_Python_env 

## deactivate 가상환경 종료
conda deactivate

## install module
pip install photutils==1.12
pip install ysfitsutilpy ysphotutilpy 
pip install opencv-python sep astro_ndslice seaborn pymysql chardet xarray
<!-- pip install exotic -->
conda install -c conda-forge jupyter-book pyppeteer
cd ~/Downloads/ && rm -rf ysfitsutilpy && git clone https://github.com/ysBach/ysfitsutilpy && cd ysfitsutilpy && git pull && pip install -e . && cd ..
cd ~/Downloads/ysfitsutilpy && pip install -e . && cd ..

cd ~/Downloads/ && rm -rf ysphotutilpy && git clone https://github.com/ysBach/ysphotutilpy && cd ysphotutilpy && git pull && pip install -e . && cd ..


cd c:\Users\Kiehyun\Downloads 
del ysfitsutilpy
git clone https://github.com/ysBach/ysfitsutilpy
cd ysfitsutilpy
cd c:\Users\Kiehyun\Downloads\ysfitsutilpy
git pull 
pip install -e . 

cd c:\Users\Kiehyun\Downloads 
del ysphotutilpy
git clone https://github.com/ysBach/ysphotutilpy
cd ysphotutilpy
cd c:\Users\Kiehyun\Downloads\ysphotutilpy
git pull 
pip install -e .


## 가상환경 내보내기 (export)
conda env export > astro_Python_env.yaml

## .yaml 파일로 새로운 가상환경 만들기
conda env create -f astro_Python_env_241208.yaml

## .yaml 파일로 가상환경 업데이트(activate 되어있을 때)
conda env update --file astro_Python_env_241208.yaml

## .yaml 파일로 가상환경 업데이트(deactivate 되어있을 때)
conda env update --astro_Python_env --file astro_Python_env_241208.yaml

## 가상환경 제거하기
conda env remove -n astro_Python_env

https://uwgdqo.tistory.com/33

https://github.com/leoybkim/exoplanet/
https://github.com/rzellem/EXOTIC 