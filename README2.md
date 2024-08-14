# NICE-SLAM-with-uncertainty


## Dependencies
```bash
# create conda env
conda create -n nice-slam python==3.8
# install torch

# other requirements
pip install -r requirements.txt
```

## Usage
### Run
Download the data as below and the data is saved into the ./Datasets/Replica folder. If the data is saved in the other folder, write the directory to the config file.(configs/Replica/room0.yaml)
```bash
# Example of Replica dataset, room0
# vanila NICE-SLAM
python -W ignore run.py configs/Replica/room0.yaml
# with uncertainty
python -W ignore run.py configs/Replica/room0.yaml --uncert
```
### Evaluation
To evaluate the reconstruction error, download the ground truth Replica meshes first.
```bash
bash scripts/download_cull_replica_mesh.sh
```
Then run the command below.
```bash
# Example of Replica, room0
OUTPUT_FOLDER=output/Replica/room0
GT_MESH=cull_replica_mesh/room0.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
```
### Run + Evaluation
To automatically run and evaluate, run the command below.
The results will be saved in the directory written to the bash file.
```bash
mkdir eval_output
# Replica, room0
bash room0.sh
# All of the Replica data
bash Replica_all.sh
```

## Acknowledgements
This implementation is based on[ NICE-SLAM](https://github.com/cvg/nice-slam/tree/master).
