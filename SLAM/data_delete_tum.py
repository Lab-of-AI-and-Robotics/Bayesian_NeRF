import os

def delete_even_indexed_images(folder_path):
    files = os.listdir(folder_path)
    
    files.sort()

    for index, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        
        # dataset /3 
        if index % 3 != 0:
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def remove_even_lines(file_path):
    result_lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        # dataset /3 
        if i < 3 or (i >= 3 and (i - 3) % 3 == 0):
            result_lines.append(lines[i])

    with open(file_path, 'w') as file:
        file.writelines(result_lines)


folder_path1 = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/depth'
folder_path2 = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb'
file_path_txt1 = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/depth.txt'
file_path_txt2 = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb.txt'
file_path_gt = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/groundtruth.txt'
file_path_accel = 'Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/accelerometer.txt'
remove_even_lines(file_path_txt1)
remove_even_lines(file_path_txt2)
remove_even_lines(file_path_gt)
remove_even_lines(file_path_accel)
delete_even_indexed_images(folder_path1)
delete_even_indexed_images(folder_path2)

