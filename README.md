# uncertainty_nerf

* For implementation details, see ./nerf_uncertainty.pdf
* Correct all NaN issues

<br/>

```
* terminal output Example:
# some experimental results:
1) strong restriction for Lagrangian (such as using exp(1e+4 * (truecolor-lambda)) returns NaN
2) using only uncertainty loss does not converge
3) using SGD does not converge
4) maximum ranges of each uncertainty (or sigma) are arbitrary chosen. need experimental examinations

# when use ReLU for Lagrangian, it is hard to handle the constraint (lambda-truecolor>0)
uncertainty_density(max:200):  tensor(91.6017, grad_fn=<MeanBackward0>)
dist2Xuncertainty_sum(max:5):  tensor(1.9720, grad_fn=<MeanBackward0>)
S_ai(max:10):  tensor(1.0467, grad_fn=<MeanBackward0>)
U_ai:  tensor(-2.0020, grad_fn=<MeanBackward0>)
S2_A(max:50):  tensor(4.4825, grad_fn=<MeanBackward0>)
U_A:  tensor(-0.3586, grad_fn=<MeanBackward0>)
1TempA:  tensor(9846.4727, grad_fn=<MeanBackward0>)
1TempB:  tensor(21.0245, grad_fn=<MeanBackward0>)
rgb:  tensor(0.5203, grad_fn=<MeanBackward0>)
uncert_map tensor(137.7527, grad_fn=<MeanBackward0>)
lam:  tensor(11.4594, grad_fn=<MeanBackward0>)
target_s:  tensor(0.5203)
val 1:  tensor(0.6361, grad_fn=<MeanBackward0>)
val 2:  tensor(0.6717, grad_fn=<MeanBackward0>)
val 3:  tensor(1.5640, grad_fn=<MeanBackward0>)
val 4:  tensor(0.1144, grad_fn=<MeanBackward0>)
# of (lamda - true color) < 0 :  3486
loss mse:  tensor(0.0099, grad_fn=<MeanBackward0>)
loss_unc:  tensor(32.6391, grad_fn=<MeanBackward0>)
iteration :  10000

# when use exp for Lagrangian
uncertainty_density(max:200):  tensor(82.9269, grad_fn=<MeanBackward0>)
dist2Xuncertainty_sum:  tensor(1.5590, grad_fn=<MeanBackward0>)
S_ai(max:10):  tensor(0.8232, grad_fn=<MeanBackward0>)
U_ai:  tensor(-2.0454, grad_fn=<MeanBackward0>)
S2_A(max:50):  tensor(3.1011, grad_fn=<MeanBackward0>)
U_A:  tensor(-1.1285, grad_fn=<MeanBackward0>)
1TempA:  tensor(45631.0469, grad_fn=<MeanBackward0>)
1TempB:  tensor(11.3169, grad_fn=<MeanBackward0>)
rgb:  tensor(0.5155, grad_fn=<MeanBackward0>)
uncert_map tensor(136.4511, grad_fn=<MeanBackward0>)
lam:  tensor(369.3352, grad_fn=<MeanBackward0>)
target_s:  tensor(0.5126)
val 1:  tensor(5.2473, grad_fn=<MeanBackward0>)
val 2:  tensor(0.4557, grad_fn=<MeanBackward0>)
val 3:  tensor(9.8470, grad_fn=<MeanBackward0>)
val 4:  tensor(0., grad_fn=<MeanBackward0>)
# of (lamda - true color) < 0 :  525
loss mse:  tensor(0.0082, grad_fn=<MeanBackward0>)
loss_unc:  tensor(31.8276, grad_fn=<MeanBackward0>)
iteration :  10000
```



<br/>

## Enviroment setting
We recommend creating a virtual environment using conda and then setting up the environment with the following code.

```
git clone https://github.com/Lab-of-AI-and-Robotics/uncertainty_nerf.git
cd uncertainty_nerf
pip install -r requirements.txt
```
<br/>

## Dataset

- #### Quick Start

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

- #### Official NeRF dataset
    You can find NeRF dataset in the [NeRF](https://www.matthewtancik.com/nerf) link. Download this link and setting dataset. To demonstrate the effect of uncertainty in our experiments, we adjusted the number of datasets provided and conducted experiments.


<br/>

## Running the code 
Below is an example of a training execution command. You can set --config, --expname, and --datadir according to your own situation to run code.
```
python run_nerf.py --config configs/fern.txt --expname fern_test --datadir ./data/nerf_llff_data/fern
```



<br/>

## Additional information
- Baseline code github repository : [banila nerf Link!](https://github.com/yenchenlin/nerf-pytorch)
- beta_min is minimum value of uncertainty.
- The raw2output function handles most of the processing, with other functions passing the processed values to raw2output.
- raw[...,4] : uncertainty of density, # raw[...,3] : density mean value, # raw[...,:3] : color mean value






<br/>


