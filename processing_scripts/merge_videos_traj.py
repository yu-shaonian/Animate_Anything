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
cur_output_root = output_root + '\\chime.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\chime1\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\chime3\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\chime7\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\shake_1\\images'

######################################
cur_output_root = output_root + '\\sunflower.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\sunflower0\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\sunflower1\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\sunflower2\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\shake_1\\images'

######################################
cur_output_root = output_root + '\\paper_plane.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\plane1'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\plane2'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\plane3'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\s_curve_3\\images'

######################################
cur_output_root = output_root + '\\leaf.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\leaf0\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\leaf1\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\leaf2\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\s_curve_3\\images'

######################################
cur_output_root = output_root + '\\zebras.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twozebras0\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twozebras1\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twozebras2\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\horizon_right_p12v\\images'

######################################
cur_output_root = output_root + '\\twocats.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twocats0\\images'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twocats1\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\twocats2\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\horizon_right_p12v\\images'

######################################
cur_output_root = output_root + '\\shake_1.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\sunflower1\\images'
v2_path='D:\\work\\LVDM_output\\supp\\rose_swaying_in_the_wind\\3\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\chime1\\images'

traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\shake_1\\images'

######################################

cur_output_root = output_root + '\\s_curve_3_v1.mp4'
v1_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\plane3'
v2_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\leaf2\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\a_piece_of_snowflake_falls\\6\\images'
v3_path = 'D:\\work\\LVDM_output\\supp\\motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_1086_s_curve_3\\samples\\s_curve_3\\a_piece_of_snowflake_floating_in_the_blue_sky__slowly_falling_down\\4\\images'
v3_path = 'D:\\work\LVDM_output\\supp\\motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_1234_s_curve_3\samples\s_curve_3\\a_feather_floating_in_the_blue_sky__slowly_falling_down\\1\\images'
traj_path = 'D:\\work\\LVDM_output\\supp\\MotionCtrl_page_videos_v1\\MotionCtrl_page_videos\\images\\MotionCtrl\\s_curve_3\\images'



imgs = sorted(glob(v1_path + '/*.png'))
imgs = [img.split('\\')[-1] for img in imgs]
print(f'Number of images: {len(imgs)}')

video = []
for img in imgs:
    # import pdb; pdb.set_trace()
    v1_img = cv2.imread(v1_path + '\\' + img)
    v2_img = cv2.imread(v2_path + '\\' + img)
    v3_img = cv2.imread(v3_path + '\\' + img)

    v1_img = cv2.cvtColor(v1_img, cv2.COLOR_BGR2RGB)
    v2_img = cv2.cvtColor(v2_img, cv2.COLOR_BGR2RGB)
    v3_img = cv2.cvtColor(v3_img, cv2.COLOR_BGR2RGB)

    traj_img = cv2.imread(traj_path + '\\' + img)
    traj_img = cv2.cvtColor(traj_img, cv2.COLOR_BGR2RGB)

    img = np.concatenate((traj_img, v1_img, v2_img, v3_img), axis=1)
    video.append(img)
video = torch.Tensor(video) # 256 x 256 x 3
torchvision.io.write_video(f'{cur_output_root}', video, 10, video_codec='h264', options={'crf': '10'})
