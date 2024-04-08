
# Bayesian NeRF: Quantifying Uncertainty with Volume Density in Neural Radiance Fields



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


## Running the example data (lego scene 4 images)
```
cd NeRF_for_rgb_img

cd NeRF_baseline
python run_nerf.py --config configs/synthetic.txt --expname ../../result/lego/4_baseline --datadir ./data/nerf_synthetic/lego_4
cd ..

cd NeRF_color
python run_nerf.py --config configs/synthetic.txt --expname ../../result/lego/4_color --datadir ./data/nerf_synthetic/lego_4
cd ..

cd NeRF_density
python run_nerf.py --config configs/synthetic.txt --expname ../../result/lego/4_density --datadir ./data/nerf_synthetic/lego_4
cd ..

cd NeRF_density_and_color
python run_nerf.py --config configs/synthetic.txt --expname ../../result/lego/4_den_col --datadir ./data/
cd ..

cd NeRF_occupancy
python run_nerf.py --config configs/synthetic.txt --expname ../../result/lego/4_occupancy --datadir ./data/
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


