# import torch

# # 저장된 모델 파일의 경로를 지정합니다.
# checkpoint_path = "/home/minjaelee/Desktop/coding/nerf-pytorch/logs/blender_paper_lego/200000.tar"

# # Checkpoint 파일 로드
# checkpoint = torch.load(checkpoint_path)

# # network_fn weights의 size 출력
# print("network_fn weights sizes:")
# for key, value in checkpoint['network_fn_state_dict'].items():
#     print(key, ":", value.size())

# # network_fine weights의 size 출력
# print("\nnetwork_fine weights sizes:")
# for key, value in checkpoint['network_fine_state_dict'].items():
#     print(key, ":", value.size())

# # optimizer weights의 size (필요한 경우) 출력
# # print("\noptimizer weights sizes:")
# # for key, value in checkpoint['optimizer_state_dict'].items():
# #     if 'state' not in key:  # state는 optimizer의 내부 상태이므로 weight size와는 관련 없습니다.
# #         print(key, ":", value.size())

import torch

# 저장된 모델 파일의 경로를 지정합니다.
checkpoint_path = "/home/minjaelee/Desktop/coding/nerf-pytorch/logs/blender_paper_lego/200000.tar"

# Checkpoint 파일 로드
checkpoint = torch.load(checkpoint_path)

def print_parameters_and_count(weights_dict):
    total_params = 0
    for key, value in weights_dict.items():
        print(key, ":", value.size())
        print(value)  # 파라미터 값 출력
        total_params += value.numel()  # 파라미터 개수 누적
    return total_params

# network_fn 파라미터 출력 및 개수 계산
print("network_fn weights sizes and values:")
print("-----------------------------------")
total_params_fn = print_parameters_and_count(checkpoint['network_fn_state_dict'])

# network_fine 파라미터 출력 및 개수 계산
print("\n\nnetwork_fine weights sizes and values:")
print("--------------------------------------")
total_params_fine = print_parameters_and_count(checkpoint['network_fine_state_dict'])

print("\nTotal parameters in network_fn:", total_params_fn)
print("Total parameters in network_fine:", total_params_fine)
print("Total parameters in both networks:", total_params_fn + total_params_fine)
