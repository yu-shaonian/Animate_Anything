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
cur_output_root = output_root + '\\dog.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\dog2\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\dog_0\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\dog1\\images'

pose_path = 'assets\\videos\\our_results\\camera_Round-R_ZoomIn.png'


######################################
# cur_output_root = output_root + '\\cat.mp4'
# v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\cat0\\images'
# v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\cat1\\images'
# v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\cat2\\images'

# pose_path = 'assets\\videos\\our_results\\camera_Round-R_ZoomIn.png'

######################################
# cur_output_root = output_root + '\\castle.mp4'
# v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\castle0\\images'
# v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\castle1\\images'
# v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\castle2\\images'

# pose_path = 'assets\\videos\\our_results\\camera_d971457c81bca597.png'

######################################
# cur_output_root = output_root + '\\temple.mp4'
# v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\temple0\\images'
# v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\temple1\\images'
# v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\temple2\\images'

######################################
cur_output_root = output_root + '\\camera_d971457c81bca597.mp4'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\castle1\\images'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\temple0\\images'
v3_path='D:\\work\\LVDM_output\\supp\\00057_0_camera_pose_motiontype_trajectory_uc_neg_condT800_7.5_supp_7001_sunrise\\samples\\d971457c81bca597\\masterpiece__best_quality__night__outdoors__sky__ocean__scenery\\6\\images'


######################################
cur_output_root = output_root + '\\camera_Round-R_ZoomIn.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\dog_0\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\cat0\\images'
v3_path='D:\\work\\LVDM_output\\supp\\Round-ZoomIn\\a_rabbit_eating_carrot\\3\\images'

pose_path = 'assets\\videos\\our_results\\camera_Round-R_ZoomIn.png'

imgs = sorted(glob(v1_path + '/*.png'))
imgs = [img.split('\\')[-1] for img in imgs]
print(f'Number of images: {len(imgs)}')

pose_img = cv2.imread(pose_path)
pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)

video = []
for img in imgs:
    # import pdb; pdb.set_trace()
    v1_img = cv2.imread(v1_path + '\\' + img)
    v2_img = cv2.imread(v2_path + '\\' + img)
    v3_img = cv2.imread(v3_path + '\\' + img)

    v1_img = cv2.cvtColor(v1_img, cv2.COLOR_BGR2RGB)
    v2_img = cv2.cvtColor(v2_img, cv2.COLOR_BGR2RGB)
    v3_img = cv2.cvtColor(v3_img, cv2.COLOR_BGR2RGB)

    img = np.concatenate((pose_img, v1_img, v2_img, v3_img), axis=1)
    video.append(img)
video = torch.Tensor(video) # 256 x 256 x 3
torchvision.io.write_video(f'{cur_output_root}', video, 10, video_codec='h264', options={'crf': '10'})
