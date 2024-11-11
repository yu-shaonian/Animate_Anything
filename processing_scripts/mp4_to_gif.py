import os
import sys
from glob import glob

import cv2
import imageio
import numpy as np
from decord import VideoReader, cpu

root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\our_results'
# root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\teasers'
# root_path="D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\videocomposer"

in_pathes = sorted(glob(root_path + '\\*.mp4'))
print(f'Number of videos: {len(in_pathes)}')

# in_pathes = in_pathes[:1]

for in_path in in_pathes:
    if 'speed_' not in in_path:
        continue
    # video_name = in_path.split('\\')[-1].split('.')[0]
    video_name = in_path.split('\\')[-1][:-4]
    video_reader = VideoReader(in_path, ctx=cpu(0))
    frame_num = len(video_reader)
    print(f'Number of frames: {frame_num} in {video_name}')

    cnt = 0
    video = []
    while True:
        try:
            frame = video_reader.next().asnumpy()
            video.append(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    imageio.mimsave(f'{root_path}\\{video_name}.gif', video, fps=10)