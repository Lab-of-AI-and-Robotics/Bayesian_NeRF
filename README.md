<div align=center>
  
# Bayesian NeRF: Quantifying Uncertainty with Volume Density in Neural Radiance Fields

Sibaek Lee, Kyeongsu Gang, Hyeonwoo Yu

[Paper](https://arxiv.org/abs/2404.06727) | [Video](https://youtu.be/wp5jW4S_jqo)

</div>

## Overview
We present the Bayesian Neural Radiance Field (NeRF), which explicitly quantifies uncertainty in geometric volume structures without the need for additional networks, making it adept for challenging observations and uncontrolled images. 

## Video Result
The results have been adjusted to account for the uncertainty in (Far Left) Base, (Left-Center) Color, (Center) Density, (Right-Center) Density and Color, and (Far Right) Occupancy.

<img src="figs/4_ori.gif" width="128" height="128"/><img src="figs/4_col.gif" width="128" height="128"/><img src="figs/4_den.gif" width="128" height="128"/><img src="figs/4_dencol.gif" width="128" height="128"/><img src="figs/4_occu.gif" width="128" height="128"/>

## Dataset setting
- Synthetic data (Blender) and real-world data (LLFF) : [NeRF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
- Modelnet Dataset : [ModelNet dataset](https://modelnet.cs.princeton.edu/)


Our proposed method excels in scenarios with limited data, hence we recommend downloading the dataset and appropriately processing it for experimentation.

## Enviroment Setting

```
conda create --name bayesian_nerf python=3.8
conda activate bayesian_nerf
pip install -r requirements.txt
```
** It is also possible to set up the environment using conda install instead of pip install


## Running the example data (chair scene 4 images)
```
# copy sample dataset to each method folder
cd Bayesian_NeRF
cp -r NeRF_for_rgb_img/NeRF_baseline/data NeRF_for_rgb_img/NeRF_color/
cp -r NeRF_for_rgb_img/NeRF_baseline/data NeRF_for_rgb_img/NeRF_density/
cp -r NeRF_for_rgb_img/NeRF_baseline/data NeRF_for_rgb_img/NeRF_density_and_color/
cp -r NeRF_for_rgb_img/NeRF_baseline/data NeRF_for_rgb_img/NeRF_occupancy/
```

<br>

```
cd NeRF_for_rgb_img

cd NeRF_baseline
python run_nerf.py --config configs/synthetic.txt --expname ../../result/chair/4_baseline --datadir ./data/nerf_synthetic/chair_4
cd ..

cd NeRF_color
python run_nerf.py --config configs/synthetic.txt --expname ../../result/chair/4_color --datadir ./data/nerf_synthetic/chair_4
cd ..

cd NeRF_density
python run_nerf.py --config configs/synthetic.txt --expname ../../result/chair/4_density --datadir ./data/nerf_synthetic/chair_4
cd ..

cd NeRF_density_and_color
python run_nerf.py --config configs/synthetic.txt --expname ../../result/chair/4_den_col --datadir ./data/nerf_synthetic/chair_4
cd ..

cd NeRF_occupancy
python run_nerf.py --config configs/synthetic.txt --expname ../../result/chair/4_occupancy --datadir ./data/nerf_synthetic/chair_4
cd ..

cd ..

```


## Running the code

- RGB img
```
cd NeRF_for_rgb_img

cd [Method Name]
python run_nerf.py --config configs/synthetic.txt --expname <Output Path> --datadir <Dataset Path>

cd ..
cd ..
```


- Depth img
```
cd NeRF_for_depth_img

python <Method.py> --config configs/coarse.txt --expname <Output Path> --datadir <Dataset Path>

cd ..
```


