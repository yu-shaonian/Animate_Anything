import os
import sys

import cv2
import torch
import torchvision
from decord import VideoReader, cpu

root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\our_results'
root_path='D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\teasers'
root_path="D:\\work\\project\\MotionCtrl_project_page\\wzhouxiff.github.io\\projects\\MotionCtrl\\assets\\videos\\videocomposer"

root_path='assets/videos/teasers'
videos = [
    'camera_d971457c81bca597.mp4',
    'camera_Round-R_ZoomIn.mp4',
    'shake_1.mp4',
    's_curve_3_v1.mp4',
    # 'rose_3.mp4',
    # 'horse.mp4',
    ]

images = []
for video in videos:
    video = f'{root_path}/{video}'
    video_reader = VideoReader(video, ctx=cpu(0))
    frame_num = len(video_reader)
    print(f'Number of frames: {frame_num} in {video}')

    while True:
        try:
            frame = video_reader.next().asnumpy()
            images.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
video = torch.Tensor(images) # 256 x 256 x 3
torchvision.io.write_video(f'{root_path}/merge.mp4', video, 10, video_codec='h264', options={'crf': '10'})
