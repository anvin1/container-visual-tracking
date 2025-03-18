
import cv2
import os
import random
import numpy as np
import torchvision.transforms as transforms
import torch
## this function return taken frames 20 - 40 frames
def extract_video_frames(video_path, start_frame_index, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

## this function for bouding box data extracting
def get_bounding_boxes(directory, start_frame_index, num_frames):
    txt_files = [filename for filename in os.listdir(directory) if filename.endswith(".txt")]
    txt_files.sort()
    bounding_boxes = []
    for idx in range(start_frame_index, start_frame_index + num_frames):
        file_name = txt_files[idx]
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            bounding_box = line.split()[-4:]
            bounding_box = [float(value) for value in bounding_box]
            bounding_boxes.append(bounding_box)
    return bounding_boxes

##  this function for random picking
def pick_random_sequence(total_frames, sequence_length):
    start_frame_index = random.randint(0, total_frames - sequence_length)
    num_frames = sequence_length
    return start_frame_index, num_frames

##  this function for total frames return
def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

## this function for caculating 4 corner box in pixel units
def center2corner(box_to_preprocess):
    center_x, center_y, width, height = box_to_preprocess
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    return top_left_x, top_left_y, width, height

## this function for caculating center, width and height
def box_to_pixel(frame, box_to_preprocess):
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y, width, height = box_to_preprocess
    center_x_pixel= int(center_x*frame_width)
    center_y_pixel =int(center_y*frame_height)
    width_pixel=int(width*frame_width)
    height_pixel=int(height*frame_height)
    
    return center_x_pixel, center_y_pixel, width_pixel, height_pixel

