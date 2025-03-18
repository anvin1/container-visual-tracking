
import numpy as np
import torch
from torch.autograd import Variable
import cv2
from sample_generator import*
import torch.optim as optim
from options import *
from data_prov import *

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2))
    b = np.array((boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
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


def crop_image_blur(img, bbox):
    x, y, w, h = np.array(bbox, dtype='float32')
    img_h, img_w, _ = img.shape
    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

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
        cropped = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    return cropped

def getbatch_actor(img, boxes):
    crop_size = 107

    num_boxes = boxes.shape[0]
    imo_g = np.zeros([num_boxes, crop_size, crop_size, 3])
    imo_l = np.zeros([num_boxes, crop_size, crop_size, 3])

    for i in range(num_boxes):
        bbox = boxes[i]
        img_crop_l, img_crop_g, out_flag = crop_image_actor(img, bbox)

        imo_g[i] = img_crop_g
        imo_l[i] = img_crop_l

    imo_g = imo_g.transpose(0, 3, 1, 2).astype('float32')
    # imo_g = imo_g - 128.
    imo_g = torch.from_numpy(imo_g)
    imo_g = Variable(imo_g)
    imo_g = imo_g.cuda()
    imo_l = imo_l.transpose(0, 3, 1, 2).astype('float32')
    # imo_l = imo_l - 128.
    imo_l = torch.from_numpy(imo_l)
    imo_l = Variable(imo_l)
    imo_l = imo_l.cuda()

    return imo_g, imo_l, out_flag
import numpy as np
import cv2




def crop_image_actor(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h
    out_flag = 0
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
        # if max(abs(min_y - min_y_val) / half_h, abs(max_y - max_y_val) / half_h, abs(min_x - min_x_val) / half_w, abs(max_x - max_x_val) / half_w) > 0.3:
        #     out_flag = 1
        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_l = cv2.resize(cropped, (img_size, img_size))

    # a, b = w / 2+0.25*w, h / 2+0.25*h
    a, b = w,h
    min_x = int(center_x - a + 0.5)
    min_y = int(center_y - b + 0.5)
    max_x = int(center_x + a + 0.5)
    max_y = int(center_y + b + 0.5)


    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        if max(abs(min_y - min_y_val) / half_h, abs(max_y - max_y_val) / half_h, abs(min_x - min_x_val) / half_w, abs(max_x - max_x_val) / half_w) > 0.3:
            out_flag = 1
        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_g = cv2.resize(cropped, (img_size, img_size))

    return scaled_l, scaled_g, out_flag

def move_crop(pos_, deta_pos, img_size, rate):
    flag = 0
    if pos_.shape.__len__() == 1:
        pos_ = np.array(pos_).reshape([1, 4])
        # deta_pos = np.array(deta_pos).reshape([1, 4])
        deta_pos = np.array(deta_pos).reshape([1, 3])

        flag = 1

    
    pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] * (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])

    # pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])

    if np.max((pos[:, 3] > (pos[:, 2] / rate) * 1.2)) == 1.0:
        pos[:, 3] = pos[:, 2] / rate

    if np.max((pos[:, 3] < (pos[:, 2] / rate) / 1.2)) == 1.0:
        pos[:, 2] = pos[:, 3] * rate

    pos[pos[:, 2] < 10, 2] = 10
    pos[pos[:, 3] < 10, 3] = 10

    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] > img_size[1], 0] = img_size[1]
    pos[pos[:, 1] > img_size[0], 1] = img_size[0]
    pos[pos[:, 0] < -pos[:, 2], 0] = -pos[:, 2]
    pos[pos[:, 1] < -pos[:, 3], 1] = -pos[:, 3]

    if flag == 1:
        pos = pos[0]

    return pos

def cal_distance(samples, ground_th):
    distance = samples[:, 0:2] + samples[:, 2:4] / 2.0 - ground_th[:, 0:2] - ground_th[:, 2:4] / 2.0
    distance = distance / samples[:, 2:4]
    rate1 = ground_th[:, 2] / samples[:, 2]
    rate1 = np.array(rate1).reshape(rate1.shape[0], 1)
    rate1 = rate1 - 1.0
    rate2 = ground_th[:, 3] / samples[:, 3]
    rate2 = np.array(rate2).reshape(rate2.shape[0], 1)
    rate2 = rate2 - 1.0
    distance = np.hstack([distance, rate1])
    # distance = np.hstack([distance, rate1])
    return distance

def init_actor(actor, image, gt):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)

    batch_num = 64
    maxiter = 80
    actor = actor.cuda()
    actor.train()
    init_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()
    _, _, out_flag_first = getbatch_actor(np.array(image), np.array(gt).reshape([1, 4]))

    # actor_samples = np.round(gen_samples(SampleGenerator('gaussian', image.size, 0.3, 1.5, None), gt, 1500, [0.7, 1], [0.9, 1.1]))
    actor_samples = np.round(gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, None),
                                         gt, 1500, [0.7, 1], [0.9, 1.1]))
    idx = np.random.permutation(actor_samples.shape[0])

    batch_img_g, batch_img_l, _ = getbatch_actor(np.array(image), actor_samples)
    batch_distance = cal_distance(actor_samples, np.tile(gt, [actor_samples.shape[0], 1]))
    batch_distance = torch.FloatTensor(batch_distance).cuda()

    while len(idx) < batch_num * maxiter:
        idx = np.concatenate([idx, np.random.permutation(actor_samples.shape[0])])

    pointer = 0

    for iter in range(maxiter):
        next = pointer + batch_num
        cur_idx = idx[pointer: next]
        pointer = next

        feat = actor(batch_img_g[cur_idx], batch_img_l[cur_idx])
        loss = loss_func(feat, batch_distance[cur_idx])

        actor.zero_grad()  
        loss.backward()  
        init_optimizer.step()  

        if loss.item() < 0.0001:
            deta_flag = 0
            print(f"Loss: {loss.item()}")
            return deta_flag, out_flag_first
    
    print(f"Loss: {loss.item()}")
    deta_flag = 1
    return deta_flag, out_flag_first

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def forward_samples(model, image, samples, out_layer='conv3'):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)

    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats

def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)

    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])

        optimizer.step()

