import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision

# images_root = '/group/30042/zhouxiawang/outputs/LDVMPose/examples/webvid_val'
# samples = sorted(glob(images_root + '/*/*'))
# print(f'Number of samples: {len(samples)}')

output_root = 'assets\\videos\\teasers'
os.makedirs(output_root, exist_ok=True)

######################################
cur_output_root = output_root + '\\rose_3.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\rose_2.0\\images'
traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\shaking_10\\images'

pose_path = 'assets\\videos\\our_results\\camera_ZoomOut.png'


######################################
cur_output_root = output_root + '\\horse.mp4'
v1_path = 'C:\\Users\\zhouxiawang\\Downloads\\tmp\\horse\\images'
traj_path = 'C:\\Users\\zhouxiawang\\Downloads\\tmp\\horizon_10\\images'

pose_path = 'assets\\videos\\our_results\\camera_ZoomIn.png'



imgs = sorted(glob(v1_path + '/*.png'))
imgs = [img.split('\\')[-1] for img in imgs]
print(f'Number of images: {len(imgs)}')

pose_img = cv2.imread(pose_path)
pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
pose_img = cv2.resize(pose_img, (256, 256))
white = np.ones((256, 256, 3)) * 255

video = []
for img in imgs:
    # import pdb; pdb.set_trace()
    v1_img = cv2.imread(v1_path + '\\' + img)

    v1_img = cv2.cvtColor(v1_img, cv2.COLOR_BGR2RGB)
    

    traj_img = cv2.imread(traj_path + '\\' + img)
    traj_img = cv2.cvtColor(traj_img, cv2.COLOR_BGR2RGB)

    # img = np.concatenate((white, pose_img, traj_img, v1_img), axis=1)
    img = np.concatenate((pose_img, traj_img, v1_img), axis=1)
    video.append(img)
video = torch.Tensor(video) # 256 x 256 x 3
torchvision.io.write_video(f'{cur_output_root}', video, 10, video_codec='h264', options={'crf': '10'})
