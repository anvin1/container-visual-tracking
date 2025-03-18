import cv2
import numpy as np
import numba
from numba import prange, jit
import math
import time

cv2.setUseOptimized(True)
cv2.setNumThreads(8) 

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def dark_channel(im, r):

    h, w = im.shape[:2]
    pad = r // 2
    dark = np.empty((h, w), dtype=np.float32)
    
    for i in prange(h):
        for j in range(w):
            a = im[i, j, 0]
            b = im[i, j, 1]
            c = im[i, j, 2]
            if a < b:
                if a < c:
                    dark[i, j] = a
                else:
                    dark[i, j] = c
            else:
                if b < c:
                    dark[i, j] = b
                else:
                    dark[i, j] = c
    
    result = np.empty((h, w), dtype=np.float32)
    for i in prange(h):
        for j in range(w):
            min_val = 1.0
            for di in range(max(0, i-pad), min(h, i+pad+1)):
                for dj in range(max(0, j-pad), min(w, j+pad+1)):
                    val = dark[di, dj]
                    if val < min_val:
                        min_val = val
            result[i, j] = min_val
    return result

def atmospheric_light(im, dark, quality=True):
    h, w = dark.shape
    if quality:
        size = h * w
        numpx = max(math.floor(size / 1000), 1)
        dark_flat = dark.flatten()
        indices = np.argpartition(dark_flat, -numpx)[-numpx:]
        rows = indices // w
        cols = indices % w
        A = np.empty(3, dtype=np.float32)
        A[0] = np.mean(im[rows, cols, 0])
        A[1] = np.mean(im[rows, cols, 1])
        A[2] = np.mean(im[rows, cols, 2])
    else:
        _, _, _, maxLoc = cv2.minMaxLoc(dark)
        A = im[maxLoc[1], maxLoc[0], :]
    return A

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def transmission_estimate(im, A, r):
    h, w = im.shape[:2]
    omega = 0.95
    im_div_A = np.empty_like(im)
    for ind in range(3):
        for i in range(h):
            for j in range(w):
                im_div_A[i, j, ind] = im[i, j, ind] / A[ind]
    dark =dark_channel(im_div_A, r)
    transmission = 1.0 - omega * dark
    return transmission

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def recover(im, t, A, tx=0.01):

    h, w = im.shape[:2]
    res = np.empty_like(im)
    for i in prange(h):
        for j in range(w):
            t_val = t[i, j] if t[i, j] > tx else tx
            for k in range(3):
                res[i, j, k] = (im[i, j, k] - A[k]) / t_val + A[k]
    return res

def guided_filter(guide, src, radius, eps, scale):

    guide_small = cv2.resize(guide, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    src_small = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    mean_I = cv2.boxFilter(guide_small, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(src_small, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide_small * src_small, cv2.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_small * guide_small, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    mean_a_up = cv2.resize(mean_a, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_b_up = cv2.resize(mean_b, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    q = mean_a_up * guide + mean_b_up
    return q

def optimized_visibility_enhancement(frame, dark_patch_size=15, guided_filter_radius=30, quality=True):

    if quality:
        downscale = 0.75   
        gf_scale = 0.75    
    else:
        downscale = 0.5    
        gf_scale = 0.5
    
    small_frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    frame_small = small_frame.astype(np.float32) / 255.0

    dark = dark_channel(frame_small, dark_patch_size)
    A = atmospheric_light(frame_small, dark, quality=quality)
    
    te = transmission_estimate(frame_small, A, dark_patch_size)
    
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_refined = guided_filter(gray_small, te, guided_filter_radius, 0.001, gf_scale)
    
    J = recover(frame_small, t_refined, A, 0.1)
    enhanced_small = np.clip(J * 255, 0, 255).astype('uint8')
    
    enhanced = cv2.resize(enhanced_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return enhanced

def dehaze(frame, quality=True):
 
    h, w = frame.shape[:2]

    return optimized_visibility_enhancement(frame, dark_patch_size=9, guided_filter_radius=90, quality=quality)
  

