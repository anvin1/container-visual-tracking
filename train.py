import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import time


from train_network import *
from data_processing import *


def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch.cuda.empty_cache()  

video_dir = "vid"
dataset_dir = "bb_gt"

video_dataset_pairs = []
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    video_path = os.path.join(video_dir, f"{folder}.mp4")
    if os.path.isdir(folder_path) and os.path.exists(video_path):
        video_dataset_pairs.append((video_path, os.path.join(folder_path, "obj_train_data")))


episode_rewards = []
episode = 250000
var =0.5
tracker = DDPG(num_actions=3)
torch.cuda.empty_cache()

for i in range(episode):
    
    selected_video, selected_dataset = random.choice(video_dataset_pairs)
    total_frames = get_total_frames(selected_video)

    min_frames=20
    max_frames=40
    num_frames =random.randint(min_frames,max_frames)

    max_start_frame_index= total_frames - num_frames
    start_frame_index = random.randint(0, max_start_frame_index)

    
    frames = extract_video_frames(selected_video, start_frame_index, num_frames)
    bounding_boxes = get_bounding_boxes(selected_dataset, start_frame_index, num_frames)
    
    print(f"Interation: {i} ")

    frame_index=0
    start_time = time.time()
    for frame, ground_th in zip(frames, bounding_boxes):
        pi_ground_th=box_to_pixel(frame,ground_th)
        gt_4 = center2corner(pi_ground_th)
        img_size= frame.shape
        if frame_index == 0:
            pos = gt_4
            rate = gt_4[2]/gt_4[3]

            resize_img=107
            img_crop_l, img_crop_g = crop_image(frame, pos)

            imo_l = np.array(img_crop_l).reshape(1, resize_img, resize_img, 3)
            imo_g = np.array(img_crop_g).reshape(1, resize_img, resize_img, 3)

            imo_g = torch.tensor(imo_g, dtype=torch.float32).permute(0, 3, 1, 2)
            imo_l = torch.tensor(imo_l, dtype=torch.float32).permute(0, 3, 1, 2)
          
            episode_reward = 0

            sample_num=128
            sample_range=0.7
            trans_f=0.2
            scale_f=1.1

            samples = np.round(genSample(gt_4, img_size, sample_num, sample_range, trans_f, scale_f))
            expert_action = cal_distance(samples, np.tile(pos, [samples.shape[0], 1]))
            expert_action[expert_action>1]=1
            expert_action[expert_action<-1]=-1

            l2_buffer = L2Buffer()
            batch_g, batch_l = getbatch(frame, samples)
            l2_buffer.add(batch_g, batch_l, expert_action)
            tracker.L2_train(l2_buffer)
            frame_index=1
          
            continue
        
     
        delta_s=0.05
        pos_=pos
        resize_img=107
        img_crop_l, img_crop_g = crop_image(frame, pos)

        imo_l = np.array(img_crop_l).reshape(1, resize_img, resize_img, 3)
        imo_g = np.array(img_crop_g).reshape(1, resize_img ,resize_img, 3)

        imo_g = torch.tensor(imo_g, dtype=torch.float32).permute(0, 3, 1, 2)
        imo_l = torch.tensor(imo_l, dtype=torch.float32).permute(0, 3, 1, 2)
    
        
        delta_pos=tracker.action(imo_g,imo_l)
 
        if np.random.random(1) < var:
            delta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt_4, gt_4]))
            test= move_crop(pos_, delta_pos_[0], img_size, rate)
            r_test = compute_iou(pos_, test)
            if np.max(abs(delta_pos_)) < 1:
                            delta_pos = delta_pos_[0]

        if delta_pos[2] > delta_s or delta_pos[2] < -delta_s:
                    delta_pos[2] = 0

        pos_ = move_crop(pos_, delta_pos, img_size, rate)

        img_crop_l_, img_crop_g_ = crop_image(frame, pos_)
        
        imo_l_ = np.array(img_crop_l_).reshape(1, resize_img, resize_img, 3)
        imo_g_ = np.array(img_crop_g_).reshape(1, resize_img, resize_img, 3)

        imo_g_ = torch.tensor(imo_g_, dtype=torch.float32).permute(0, 3, 1, 2)
        imo_l_ = torch.tensor(imo_l_, dtype=torch.float32).permute(0, 3, 1, 2)


        r = compute_iou(pos_, gt_4)

        if r >= 0.7:
            reward = 1
        else:
            reward = -1
        done = False
        episode_reward += reward
        pos = pos_

        
        tracker.memory.add(imo_g,imo_l, delta_pos, reward,  imo_g_,imo_l_, done)
    

        if i % 2000== 0:
            cv2.rectangle(frame, (int(pos_[0]), int(pos_[1])), (int(pos_[0]+pos_[2]), int(pos_[1]+pos_[3])), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(gt_4[0]), int(gt_4[1])), (int(gt_4[0]+gt_4[2]), int(gt_4[1]+gt_4[3])), (0, 0, 255), 2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
       

    tracker.train()
    print(var)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    tracker.episode += 1  
    if i % 10000==0:
        torch.save(tracker.Actor.state_dict(), '_actor_model.pth')
        torch.save(tracker.Critic.state_dict(), '_critic_model.pth')
        torch.save(tracker.TargetActor.state_dict(), '_target_actor_model.pth')
        torch.save(tracker.TargetCritic.state_dict(), '_target_critic_model.pth')
        
   
    episode_rewards.append(episode_reward)


    if i > 10000 and episode_reward != num_frames:
            cv2.rectangle(frame, (int(pos_[0]), int(pos_[1])), (int(pos_[0]+pos_[2]), int(pos_[1]+pos_[3])), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(gt_4[0]), int(gt_4[1])), (int(gt_4[0]+gt_4[2]), int(gt_4[1]+gt_4[3])), (0, 0, 255), 2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    if i % 10000 == 0 and i!=0:
        var = var * 0.95
        print(var)

cv2.destroyAllWindows()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()