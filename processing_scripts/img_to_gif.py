import os
import sys
from glob import glob

import cv2
import imageio
import numpy as np

root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\our_results'
root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\teasers'
root_path="D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\videocomposer"
root_path='/Users/zhouxiawang/Downloads/two_horse_walking_in_opposite_directions__distant_view/1/images'

in_pathes = sorted(glob(root_path + '/*.png'))
print(f'Number of videos: {len(in_pathes)}')

video = []
for in_path in in_pathes:
    img = cv2.imread(in_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video.append(img)
imageio.mimsave(f'{root_path}\\two_hourse.gif', video, fps=10, loop=0)

# in_pathes = in_pathes[:1]

# for in_path in in_pathes:
#     video_name = in_path.split('\\')[-1].split('.')[0]
#     video_reader = VideoReader(in_path, ctx=cpu(0))
#     frame_num = len(video_reader)
#     print(f'Number of frames: {frame_num} in {video_name}')

#     cnt = 0
#     video = []
#     while True:
#         try:
#             frame = video_reader.next().asnumpy()
#             video.append(frame)
#             # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             # cnt += 1
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         except:
#             break
#     imageio.mimsave(f'{root_path}\\{video_name}.gif', video, fps=10)