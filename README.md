# uncertainty_nerf
To Minjae:Please interpret all the following explanations into English

*For implementation details, see ./nerf_uncertainty.pdf
*Correct all NaN issues
*terminal output Example:
```
uncertainty:  tensor(0.0196, grad_fn=<MeanBackward0>)
S_ai:  tensor(0.3470, grad_fn=<MeanBackward0>)
U_ai:  tensor(-0.0435, grad_fn=<MeanBackward0>)
1TempA:  tensor(95620.0312, grad_fn=<MeanBackward0>)
1TempB:  tensor(213687.8438, grad_fn=<MeanBackward0>)
rgb:  tensor(0.5022, grad_fn=<MeanBackward0>)
lam:  tensor(142606.7031, grad_fn=<MeanBackward0>)
S_A:  tensor(0.0452, grad_fn=<MaxBackward1>)
U_A:  tensor(10.6829, grad_fn=<MeanBackward0>)
target_s:  tensor(0.5073)
val 1:  tensor(10.6473, grad_fn=<MeanBackward0>)
val 2:  tensor(-4.5688, grad_fn=<MeanBackward0>)
val 3:  tensor(21.1087, grad_fn=<MeanBackward0>)
val 4:  tensor(0., grad_fn=<MeanBackward0>)
count number that lamda - true color < 0 :  0
iteration :  10250
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
