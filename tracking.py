from model import *
from options import *
from PIL import Image
import time
from actor_network import ActorNetwork
from bbreg import *

from dehazing_module import dehaze
from process_function import *
from data_vid_process import get_total_frames, extract_video_frames,get_bounding_boxes,box_to_pixel,center2corner


num_actions=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


video_dir = r"vid"
dataset_dir = r"bb_gt"

video_dataset_pairs = []
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    video_path = os.path.join(video_dir, f"{folder}.mp4")
    if os.path.isdir(folder_path) and os.path.exists(video_path):
        video_dataset_pairs.append((video_path, os.path.join(folder_path, "obj_train_data")))

episode=1
pred_bboxes = []
gt_bboxes = []


#actor and critic model
actor=ActorNetwork("vggm1-4.npy",num_actions=num_actions).to(device)
model_actor_path="actor.pth"
actor.load_state_dict(torch.load(model_actor_path, weights_only=True))
actor=actor.cuda()
model = MDNet("critic.pth")
model=model.cuda()
model.set_learnable_params(opts['ft_layers'])
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
criterion = BinaryLoss()
init_optimizer = set_optimizer(model, opts['lr_init'])
update_optimizer = set_optimizer(model, opts['lr_update'])

for i in range(episode):
    for selected_video, selected_dataset in video_dataset_pairs:
        total_frames = get_total_frames(selected_video)

        start_frame_index = 0
        num_frames = total_frames - 1
    
        frames = extract_video_frames(selected_video, start_frame_index, num_frames) 
        bounding_boxes = get_bounding_boxes(selected_dataset, start_frame_index, num_frames)
        frame_index=0

    
        
        for frame, ground_th in zip(frames, bounding_boxes):
            
            # frame resize to 640x480
            frame= cv2.resize(frame, (640, 480))  

            pi_ground_th=box_to_pixel(frame,ground_th)
            gt = center2corner(pi_ground_th)
            img_size = frame.shape[1], frame.shape[0]
            
        
            if frame_index ==0:
                frame = dehaze(frame)
        
                target_bbox=np.array(gt)
                init_bbox=gt
                rate=init_bbox[2]/init_bbox[3]

                result = np.zeros((len(frames), 4))
                result_bb = np.zeros((len(frames), 4))
                result[0] = target_bbox
                result_bb[0] = target_bbox
                success = 1

                # training regression box
                bbreg_examples = gen_samples(SampleGenerator('uniform', img_size, 0.3, 1.5, 1.1),
                                    target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
                bbreg_feats = forward_samples(model, frame, bbreg_examples)
                bbreg = BBRegressor(img_size)
                bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

                # Draw pos/neg samples
                pos_examples = gen_samples(SampleGenerator('gaussian',  img_size, 0.1, 1.2),
                                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

                neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform',  img_size, 1, 2, 1.1),
                                target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', img_size, 0, 1.2, 1.1),
                            target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
                neg_examples = np.random.permutation(neg_examples)

                # Extract pos/neg features
                pos_feats = forward_samples(model, frame, pos_examples)
                neg_feats = forward_samples(model, frame, neg_examples)
                feat_dim = pos_feats.size(-1)
                
                # # Initial training

                train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

                deta_flag, out_flag_first = init_actor(actor, frame, target_bbox)
                
                
                init_generator = SampleGenerator('gaussian',  img_size, opts['trans_f'], 1, valid=False)
                sample_generator = SampleGenerator('gaussian',  img_size, opts['trans_f'], opts['scale_f'], valid=False)
                pos_generator = SampleGenerator('gaussian',  img_size, 0.1, 1.2)
                neg_generator = SampleGenerator('uniform',  img_size, 1.5, 1.2)
            
                # Init pos/neg features for update
                pos_feats_all = [pos_feats[:opts['n_pos_update']]]
                neg_feats_all = [neg_feats[:opts['n_neg_update']]]
                data_frame = [0]

                pos_score = forward_samples(model, frame, np.array(init_bbox).reshape([1, 4]), out_layer='fc6')
              
                img_learn = [frame]
                pos_learn = [ init_bbox]
                score_pos = [pos_score.cpu().numpy()[0][1]]
                frame_learn = [0]
                pf_frame = []
                update_lenth = 20
                detetion = 0
                frame_index=1

                continue


            img_g, img_l, out_flag = getbatch_actor(np.array(frame), np.array(target_bbox).reshape([1, 4]))

            # actor move box
            deta_pos = actor(img_g, img_l)
            deta_pos = deta_pos.data.clone().cpu().numpy()
            delta_s=0.05
            if deta_pos[:, 2] > delta_s or deta_pos[:, 2] < -delta_s:
                    deta_pos[:, 2] = 0
           
            if len(pf_frame) and frame_index == (pf_frame[-1] + 1):
                    deta_pos[:, 2] = 0
                
            # critic evluates
            img_size=frame.shape[0:2]
            pos_ = np.round(move_crop(np.array(target_bbox), deta_pos,  img_size, rate))
            posk = pos_.copy()  
            r = forward_samples(model,frame, np.array(pos_).reshape([1, 4]), out_layer='fc6')
            r = r.cpu().numpy()


          
            if r[0][1]>0:
                target_bbox=pos_
                target_score=r[0][1]
                bbreg_bbox=pos_
                success = 1
                if not out_flag:
                    fin_score = r[0][1]
                    img_learn.append(frame)
                    pos_learn.append(target_bbox)
                    score_pos.append(fin_score)
                    frame_learn.append(frame_index)
                    while len(img_learn) > update_lenth * 2:
                        del img_learn[0]
                        del pos_learn[0]
                        del score_pos[0]
                        del frame_learn[0]
                result[frame_index] = target_bbox
                result_bb[frame_index] = bbreg_bbox
            else:
                detetion+=1
                if len(pf_frame) == 0:
                    pf_frame = [frame_index]
                else:
                    pf_frame.append(frame_index)

                if (len(frame_learn) == update_lenth*2 and data_frame[-1] not in frame_learn ) or data_frame[-1] == 0:
                    for num in range(max(0, img_learn.__len__() - update_lenth), img_learn.__len__()):
                        if frame_learn[num] not in data_frame:
                            gt_ = pos_learn[num]
                            image_ = img_learn[num]

                            pos_examples = np.round(gen_samples(pos_generator, gt_,
                                                                opts['n_pos_update'],
                                                                opts['overlap_pos_update']))
                            neg_examples = np.round(gen_samples(neg_generator, gt_,
                                                                opts['n_neg_update'],
                                                                opts['overlap_neg_update']))
                            
                            pos_feats_ = forward_samples(model, image_, pos_examples)
                            neg_feats_ = forward_samples(model, image_, neg_examples)
                            pos_feats_all.append(pos_feats_)
                            neg_feats_all.append(neg_feats_)
                            data_frame.append(frame_learn[num])
                            if len(pos_feats_all) > 10:
                                del pos_feats_all[0]
                                del neg_feats_all[0]
                                del data_frame[0]
                        else:
                            pos_feats_ = pos_feats_all[data_frame.index(frame_learn[num])]
                            neg_feats_ = neg_feats_all[data_frame.index(frame_learn[num])]

                        if num == max(0, img_learn.__len__() - update_lenth):
                            pos_feats = pos_feats_
                            neg_feats = neg_feats_

                        else:
                            pos_feats = torch.cat([pos_feats, pos_feats_], 0)
                            neg_feats = torch.cat([neg_feats, neg_feats_], 0)


                if success:
                    sample_generator.set_trans_f(opts['trans_f'])
                else:
                    sample_generator.set_trans_f(opts['trans_f_expand'])
                samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
                if frame_index < 10 or out_flag or ((init_bbox[2] * init_bbox[3]) > 1000 and (target_bbox[2] * target_bbox[3] / (init_bbox[2] * init_bbox[3]) > 2.5 or target_bbox[2] * target_bbox[3] / (init_bbox[2] * init_bbox[3]) < 0.4)):
                    sample_generator.set_trans_f(opts['trans_f_expand'])
                    samples_ = np.round(gen_samples(sample_generator, np.hstack([target_bbox[0:2] + target_bbox[2:4] / 2 - np.array(init_bbox)[2:4] / 2, np.array(init_bbox)[2:4]]), opts['n_samples']))
                    samples = np.vstack([samples, samples_])

                sample_scores = forward_samples(model, frame, samples, out_layer='fc6')
                top_scores, top_idx = sample_scores[:, 1].topk(5)
                top_idx = top_idx.cpu().numpy()
                target_score = top_scores.mean()
                target_bbox = samples[top_idx].mean(axis=0)
                success = target_score > opts['success_thr']
                if success:
                    bbreg_samples = samples[top_idx]
                    bbreg_feats = forward_samples(model, frame, bbreg_samples)
                    bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                    bbreg_bbox = bbreg_samples.mean(axis=0)

                    img_learn.append(frame)
                    pos_learn.append(target_bbox)
                    score_pos.append(target_score)
                    frame_learn.append(frame_index)
                    while len(img_learn) > 2*update_lenth:
                        del img_learn[0]
                        del pos_learn[0]
                        del score_pos[0]
                        del frame_learn[0]

                else:
                    bbreg_bbox = target_bbox

                if not success:
                    target_bbox = result[frame_index - 1]
                    bbreg_bbox = result_bb[frame_index - 1]

        
            # visualization
            cv2.rectangle(frame, 
                         (int(target_bbox[0]), int(target_bbox[1])), 
                         (int(target_bbox[0]+target_bbox[2]), int(target_bbox[1]+target_bbox[3])), 
                         (0, 0, 255), 
                         2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            pred_bboxes.append(target_bbox)
            gt_bboxes.append(gt)
            frame_index += 1

    
cv2.destroyAllWindows()
           


