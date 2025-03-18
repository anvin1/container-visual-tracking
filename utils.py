import cv2
import numpy as np

def overlap_ratio(rect1, rect2):

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image(img, bbox, img_size=128, padding=16, valid=False):
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

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= img_w or max_x <= 0 or min_y >= img_h or max_y <= 0:
        cropped = 128 * np.ones((img_size, img_size, 3), dtype='uint8')
    else:
        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')

        src_min_x = max(0, min_x)
        src_min_y = max(0, min_y)
        src_max_x = min(img_w, max_x)
        src_max_y = min(img_h, max_y)

        dst_min_x = max(0, -min_x)
        dst_min_y = max(0, -min_y)
        dst_max_x = dst_min_x + (src_max_x - src_min_x)
        dst_max_y = dst_min_y + (src_max_y - src_min_y)

        if src_min_x < src_max_x and src_min_y < src_max_y:
            cropped[dst_min_y:dst_max_y, dst_min_x:dst_max_x, :] = img[src_min_y:src_max_y, src_min_x:src_max_x, :]

    scaled = cv2.resize(cropped, (img_size, img_size))
    return scaled