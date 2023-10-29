# uncertainty_nerf
To Minjae:Please interpret all the following explanations into English

* For implementation details, see ./nerf_uncertainty.pdf

* Correct all NaN issues

* terminal output Example:

```
# some experimental results:
1) strong restriction for Lagrangian (such as using exp(1e+4 * (truecolor-lambda)) returns NaN
2) using only uncertainty loss does not converge
3) using SGD does not converge

# when use ReLU for Lagrangian, it is hard to handle the constraint (lambda-truecolor>0)
uncertainty_density(max:200):  tensor(190.1944, grad_fn=<MeanBackward0>)
dist2Xuncertainty_sum(max:5):  tensor(7.4993, grad_fn=<MeanBackward0>)
S_ai(max:10):  tensor(2.5781, grad_fn=<MeanBackward0>)
U_ai:  tensor(-2.1569, grad_fn=<MeanBackward0>)
S2_A(max:50):  tensor(36.2568, grad_fn=<MeanBackward0>)
U_A:  tensor(-11.9440, grad_fn=<MeanBackward0>)
1TempA:  tensor(1.4028e+09, grad_fn=<MeanBackward0>)
1TempB:  tensor(1808.2527, grad_fn=<MeanBackward0>)
rgb:  tensor(0.5169, grad_fn=<MeanBackward0>)
uncert_map tensor(179.8027, grad_fn=<MeanBackward0>)
lam:  tensor(4.8892, grad_fn=<MeanBackward0>)
target_s:  tensor(0.5212)
val 1:  tensor(1.2728, grad_fn=<MeanBackward0>)
val 2:  tensor(1.7378, grad_fn=<MeanBackward0>)
val 3:  tensor(2.5739, grad_fn=<MeanBackward0>)
val 4:  tensor(0., grad_fn=<MeanBackward0>)
# of (lamda - true color) < 0 :  5184
loss mse:  tensor(0.0091, grad_fn=<MeanBackward0>)
loss_unc:  tensor(5.5845, grad_fn=<AddBackward0>)
iteration :  10000

# when use exp for Lagrangian
uncertainty_density(max:200):  tensor(101.0658, grad_fn=<MeanBackward0>)
dist2Xuncertainty_sum(max:5):  tensor(2.3786, grad_fn=<MeanBackward0>)
S_ai(max:10):  tensor(1.1693, grad_fn=<MeanBackward0>)
U_ai:  tensor(0.4142, grad_fn=<MeanBackward0>)
S2_A(max:50):  tensor(18.6530, grad_fn=<MeanBackward0>)
U_A:  tensor(-7.8704, grad_fn=<MeanBackward0>)
1TempA:  tensor(30070.9531, grad_fn=<MeanBackward0>)
1TempB:  tensor(13.1023, grad_fn=<MeanBackward0>)
rgb:  tensor(0.5018, grad_fn=<MeanBackward0>)
uncert_map tensor(169.0121, grad_fn=<MeanBackward0>)
lam:  tensor(683.2997, grad_fn=<MeanBackward0>)
target_s:  tensor(0.5003)
val 1:  tensor(5.6391, grad_fn=<MeanBackward0>)
val 2:  tensor(1.2556, grad_fn=<MeanBackward0>)
val 3:  tensor(5.6291, grad_fn=<MeanBackward0>)
val 4:  tensor(0., grad_fn=<MeanBackward0>)
# of (lamda - true color) < 0 :  250
loss mse:  tensor(0.0070, grad_fn=<MeanBackward0>)
loss_unc:  tensor(12.5238, grad_fn=<AddBackward0>)
iteration :  10000

```

baseline code : https://github.com/yenchenlin/nerf-pytorch (바닐라 널프 깃헙코드 주소)





- def raw2outputs 함수에서 모든 loss를 위한 값들을 계산하며 그이외의 함수들은 값을 전달하는것뿐임. (구현시 이 함수가 가장 중요!)
- raw가 model의 아웃풋이며 (raw[...,4] : uncertainty of density, # raw[...,3] : density mean value, # raw[...,:3] : color mean value )
- 현재 loss가 nan 으로 가는 문제는 해결했지만, 학습시 rgb가 올바르게 학습되지 않는 문제가 남아있음.

### 기본 설치(CONDA 환경 만든후)
```
git clone https://github.com/Lab-of-AI-and-Robotics/uncertainty_nerf.git
cd uncertainty_nerf
pip install -r requirements.txt
```

### 실행명령어
아래 명령어를 입력 log 폴더에 결과가 저장됨
```
python run_nerf.py --config configs/fern.txt --expname fern_test --datadir ./data/nerf_llff_data/fern
```

### 파라미터 관련 정보
run_nerf.py에서 아래 코드는 학습에 필요한 파라미터를 바꾸는 값 (lr과 beta_min(uncertainty의 최솟값)이 학습에 영향을 끼침)
```
    parser.add_argument("--lrate", type=float, default=5e-5, # 4 
                        help='learning rate')
    parser.add_argument("--beta_min",   type=float, default=0.001) # 얘도 중요한 파라미터임, 건들여보자
    parser.add_argument("--w",   type=float, default=0.01) 
    parser.add_argument("--total_epoch",   type=int, default=300100, 
                        help='total_epoch')
```

- loss는 run_nerf_helpers.py의 위에 loss_uncert3 가 정의되어 있음


### 학습 위치 정보
zero_grad() 함수 뒤부터 학습과 관련된 코드가 있음
```
optimizer.zero_grad()
```


### 데이터셋
데이터셋은 간단하게 synthetic은 chair / llff는 fern을 넣어두었음. 추가 다른 scene에 대해서는 별도로 다운로드 필요
