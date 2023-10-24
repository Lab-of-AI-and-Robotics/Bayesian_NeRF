import numpy as np

test=np.load('poses_bounds.npy')
test = np.delete(test,(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19),axis=0)
print(test)
np.save('/home/takseungjun/Uncert_NeRF/data/nerf_llff_data/fern/4',test)
