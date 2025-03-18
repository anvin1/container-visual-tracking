
import cv2
import os
import random
import numpy as np
import torchvision.transforms as transforms
import torch
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

def pick_random_sequence(total_frames, sequence_length):
    start_frame_index = random.randint(0, total_frames - sequence_length)
    num_frames = sequence_length
    return start_frame_index, num_frames

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def center2corner(box_to_preprocess):
    center_x, center_y, width, height = box_to_preprocess
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    return top_left_x, top_left_y, width, height

def box_to_pixel(frame, box_to_preprocess):
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y, width, height = box_to_preprocess
    center_x_pixel= int(center_x*frame_width)
    center_y_pixel =int(center_y*frame_height)
    width_pixel=int(width*frame_width)
    height_pixel=int(height*frame_height)
    
    return center_x_pixel, center_y_pixel, width_pixel, height_pixel

def genSample(ground_th, img_size, sample_num, sample_range, trans_f, scale_f):

    h = img_size[0]
    w = img_size[1]

    bb = np.array([ground_th[0]+ ground_th[2]/2., ground_th[1]+ground_th[3]/2., ground_th[2], ground_th[3]])
    samples = np.tile(bb, [sample_num, 1])
  
    samples[:, 0:2] = samples[:, 0:2] + trans_f *np.round(np.mean(ground_th[2:4]) * np.maximum(-1, np.minimum(1, 0.5 * np.random.randn(sample_num, 2)))) 
    samples[:, 2:] = samples[:, 2:] * np.tile(1.05**(scale_f * np.maximum(-1, np.minimum(1, 0.5 * np.random.randn(sample_num, 1)))), [1, 2])
    samples[:, 2] = np.maximum(10, np.minimum(w - 10, samples[:, 2]))
    samples[:, 3] = np.maximum(10, np.minimum(h - 10, samples[:, 3]))

    bb_samples = np.transpose(np.vstack((np.transpose(samples[:,0]-samples[:,2]/2), np.transpose(samples[:,1]-samples[:,3]/2), np.transpose(samples[:,2]), np.transpose(samples[:,3]))))
    bb_samples[:, 0] = np.maximum(1 - bb_samples[:, 2] / 2., np.minimum(w - bb_samples[:, 2] / 2., bb_samples[:, 0]))
    bb_samples[:, 1] = np.maximum(1 - bb_samples[:, 3] / 2., np.minimum(h - bb_samples[:, 3] / 2., bb_samples[:, 1]))
    samples = np.round(bb_samples)

    r = func_iou(samples, ground_th)
    idx_temp = np.where(sample_range<r)
    idx = idx_temp[0]
    samples = samples[idx][:]
    
    return samples

def func_iou(bb, gtb):
    gtbb = np.tile(gtb,[bb.shape[0],1])
    iou = np.zeros((bb.shape[0],1))
    for i in range(bb.shape[0]):
        iw = min(bb[i][2]+bb[i][0],gtbb[i][2]+gtbb[i][0]) - max(bb[i][0],gtbb[i][0]) + 1
        ih = min(bb[i][3]+bb[i][1],gtbb[i][3]+gtbb[i][1]) - max(bb[i][1],gtbb[i][1]) + 1
        if iw>0 and ih>0:
            ua = (bb[i][2]+1)*(bb[i][3]+1) + (gtbb[i][2]+1)*(gtbb[i][3]+1) - iw*ih
            iou[i][:] = iw*ih/ua
    return iou

def cal_distance(samples, ground_th):
    distance = samples[:, 0:2] + samples[:, 2:4] / 2 - ground_th[:, 0:2] - ground_th[:, 2:4] / 2
    distance = distance / samples[:, 2:4]
    rate=ground_th[:, 2] / samples[:, 2]
    rate= np.array(rate).reshape(rate.shape[0], 1)
    rate = rate - 1.0
    distance = np.hstack([distance, rate])
    return distance


size=107
def getbatch(img, boxes):
    crop_size = size
    num_boxes = boxes.shape[0]
    imo_g = np.zeros([num_boxes, crop_size, crop_size, 3])
    imo_l = np.zeros([num_boxes, crop_size, crop_size, 3])


    for i in range(num_boxes):

        bbox = boxes[i]
        img_crop_l, img_crop_g = crop_image(img, bbox)
             
        imo_g[i] = img_crop_g
        imo_l[i] = img_crop_l

    
    imo_g = torch.tensor(imo_g, dtype=torch.float32).permute(0, 3, 1, 2)
    imo_l = torch.tensor(imo_l, dtype=torch.float32).permute(0, 3, 1, 2)
    
    return imo_g, imo_l

def crop_image(img, bbox, img_size=size, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]
  
    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')

        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    

    scaled_l = cv2.resize(cropped, (img_size, img_size))

   
    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_g = cv2.resize(cropped, (img_size, img_size))

    return scaled_l, scaled_g
    
delta_s=0.05
def move_crop(pos_, deta_pos, img_size, rate):

    flag = 0
    if deta_pos[2] > delta_s or deta_pos[2] < -delta_s:
        deta_pos[2] = 0

    if isinstance(pos_, tuple):
        pos_ = np.array(pos_)

    if pos_.ndim == 1:
        pos_ = pos_.reshape(1, 4)
        deta_pos = np.array(deta_pos).reshape(1, 3)
        flag = 1

    pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] * (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])



    pos[pos[:, 2] < 10, 2] = 10
    pos[pos[:, 3] < 10, 3] = 10


    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] + pos[:, 2] > img_size[1], 0] = \
        img_size[1] - pos[pos[:, 0] + pos[:, 2] > img_size[1], 2] - 1
    pos[pos[:, 1] + pos[:, 3] > img_size[0], 1] = \
        img_size[0] - pos[pos[:, 1] + pos[:, 3] > img_size[0], 3] - 1
    pos[pos[:, 0] < 0, 0] = 0
    pos[pos[:, 1] < 0, 1] = 0

    pos[pos[:, 2] > img_size[1], 2] = img_size[1]
    pos[pos[:, 3] > img_size[0], 3] = img_size[0]
    if flag == 1:
        pos = pos[0]

    return pos

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        interArea = (xB - xA) * (yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou