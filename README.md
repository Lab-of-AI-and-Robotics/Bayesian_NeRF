# uncertainty_nerf
By Minjae Lee, UNIST

baseline code : https://github.com/yenchenlin/nerf-pytorch (바닐라 널프 깃헙코드 주소)

곧 실행방법 실험하여 업데이트 예정입니다.


- 위아래 ######### 과같이 하여 원본코드를 #### 안에두고, 수정하였음
- 구현할때 SSIM을 계산을 위해서 baseline폴더에 external을 추가하였음. (그리고 ssim lpips 계산을 위해 run_nerf_helpers에 구현하였음)
- run_nerf.py 코드에서도 이미지를 저장할때 수치적인 값들과 UNCERT IMAGE도 저장하도록 render_path 함수에 코드들 추가하였음
- baseline code에 버전 오류가 있어 baseline code의 load_llf.py 코드중 return imageio.imread(f, ignoregamma=True) 를 return imageio.imread(f)로 수정하였음.
- def raw2outputs 함수에서 모든 loss를 위한 값들을 계산하며 그이외의 함수들은 값을 전달하는것뿐임.
- 10-25일 : 모든 nan문제 해결(LR을 조절하면서 어디가 발산하는지 확인하고 특히 log를 조심! ) / but color값이 0 or 1으로 학습이 제대로 되지 않는 문제 남아있음

### 실행명령어
실행할시 log 폴더에 결과가 저장됨
```
python run_nerf.py --config configs/fern.txt --expname fern_test --datadir ./data/nerf_llff_data/fern
```

### 파라미터 수정
run_nerf.py에서의 아래 코드에서 가능 (lr과 beta_min(uncertainty의 최솟값)이 학습에 영향을 끼침)
```
    parser.add_argument("--lrate", type=float, default=5e-5, # 4 
                        help='learning rate')
    parser.add_argument("--beta_min",   type=float, default=0.001) # 얘도 중요한 파라미터임, 건들여보자
    parser.add_argument("--w",   type=float, default=0.01) 
    parser.add_argument("--total_epoch",   type=int, default=300100, 
                        help='total_epoch')
```

### 학습 위치 
zero_grad() 함수 뒤부터 학습 진행
```
optimizer.zero_grad()
```