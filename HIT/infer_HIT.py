import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import json
import wandb

from utils import compute_dict_mean, set_seed, load_data # data functions
from constants import TASK_CONFIGS
from model_util import make_policy, make_optimizer


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data #qpos: (Batch, Episode length, state_dim)
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # print("image_data" ,image_data.shape)
    # print("qpos_data ", qpos_data.shape)
    # print("action_data ", action_data.shape)
    # print("is_pad ", is_pad.shape)
    # print()


    # HIT target format
    # # image_data torch.Size([48, 2, 3, 360, 640])       # # image_data torch.Size([48, 4, 3, 360, 640])
    # # qpos_data  torch.Size([48, 35])
    # # action_data  torch.Size([48, 50, 40])
    # # is_pad  torch.Size([48, 50])

    # Current format
    # image_data torch.Size([48, 2, 3, 480, 640])
    # qpos_data  torch.Size([48, 23])
    # action_data  torch.Size([48, 50, 23])
    # is_pad  torch.Size([48, 50])

        # Needs to be set
        # 'height':args['height'],
        # 'width':args['width'],

    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1



import h5py
def load_episode(hdf5_path):
    """
    Loads the episode from disk into memory.
    Expects the same structure your converter wrote:
      /observations/obs_buf      (TxD)       -- optional
      /observations/qpos         (TxQ)
      /action                     (TxA)
      /observations/timestamps    (T,)
      /observations/images/cam_left  (TxHxWxC)
      /observations/images/cam_right (TxHxWxC)
    Returns a dict of numpy arrays.
    """
    with h5py.File(hdf5_path, "r") as f:
        # read everything
        qpos        = f["observations/qpos"][:]          # shape (T, Q)
        action      = f["action"][:]                      # shape (T, A)
        timestamps  = f["observations/timestamps"][:]     # shape (T,)
        cam_left    = f["observations/images/cam_left"][:]   # (T, H, W, C)
        cam_right   = f["observations/images/cam_right"][:]  # (T, H, W, C)
    # figure out how many “real” steps we have (i.e. before padding)
    # here we assume any zero‐timestamp is padding
    real_steps = int(np.argmax(timestamps == 0) if np.any(timestamps == 0) else len(timestamps))
    return {
        "qpos":       qpos,
        "action":     action,
        "timestamps": timestamps,
        "cam_left":   cam_left,
        "cam_right":  cam_right,
        "length":     real_steps
    }

def iterate_episode(ep_data):
    """
    Generator yielding, for t in [0 .. length‑1], tuples of
      (image_data, qpos_data, action_data, is_pad)
    where:
      image_data is a torch.FloatTensor of shape (2, C, H, W) in [0,1]
      qpos_data   is a torch.FloatTensor of shape (Q,)
      action_data is a torch.FloatTensor of shape (A,)
      is_pad      is a bool (always False here)
    """
    T = ep_data["length"]
    for t in range(T):
        # grab numpy arrays
        left  = ep_data["cam_left"][t]    # HxWxC (uint8)
        right = ep_data["cam_right"][t]
        qpos  = ep_data["qpos"][t]        # Q
        act   = ep_data["action"][t]      # A

        # to torch, reorder to (C,H,W) and normalize
        left_t  = torch.from_numpy(left).permute(2,0,1).float().div(255.0)
        right_t = torch.from_numpy(right).permute(2,0,1).float().div(255.0)
        images = torch.stack([left_t, right_t], dim=0)  # (2, C, H, W)

        qpos_t = torch.from_numpy(qpos).float()
        act_t  = torch.from_numpy(act).float()

        yield images, qpos_t, act_t, False






def main_train():
    set_seed(1)
    task_name = "data_picking"
    ckpt_dir = "picking1/" 
    policy_class = "HIT" 
    chunk_size = 50 
    hidden_dim = 512

    lr = 1e-5
    seed = 0

    backbone = "resnet18" 
    same_backbones = True 
    use_pos_embd_image = 1 
    use_pos_embd_action = 1
    dec_layers = 6

    use_mask = True 
    # --data_aug 
    nheads = 8


    pretrained_path = "picking_new_mask/_data_picking_HIT_resnet18_True"

    

    task_config = TASK_CONFIGS[task_name]
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = task_config.get('state_dim', 40)
    action_dim = task_config.get('action_dim', 40)
    state_mask = task_config.get('state_mask', np.ones(state_dim))
    action_mask = task_config.get('action_mask', np.ones(action_dim))

    if use_mask:
        state_dim = sum(state_mask)
        action_dim = sum(action_mask)
        state_idx = np.where(state_mask)[0].tolist()
        action_idx = np.where(action_mask)[0].tolist()
    else:
        state_idx = np.arange(state_dim).tolist()
        action_idx = np.arange(action_dim).tolist()
    lr_backbone = 1e-5

    policy_config = {'lr': lr,
                        'hidden_dim': hidden_dim,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'num_queries': chunk_size,
                        'camera_names': camera_names,
                        'action_dim': action_dim,
                        'state_dim': state_dim,
                        'backbone': backbone,
                        'same_backbones': same_backbones,
                        'lr_backbone': lr_backbone,
                        'context_len': 183+chunk_size, #for 224,400
                        'num_queries': chunk_size, 
                        'use_pos_embd_image': use_pos_embd_image,
                        'use_pos_embd_action': use_pos_embd_action,
                        'feature_loss': None,
                        'feature_loss_weight': None,
                        'self_attention': True,
                        'state_idx': state_idx,
                        'action_idx': None,
                        'state_mask': state_mask,
                        'action_mask': action_mask,
                        }


    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    # if config['load_pretrain']:
    loading_status = policy.deserialize(torch.load(f'{pretrained_path}/policy_last.ckpt', map_location='cuda'))
    print(f'loaded! {loading_status}')


    policy.cuda()



# --task_name data_picking --ckpt_dir picking/ --policy_class HIT --chunk_size 50 --hidden_dim 512 --batch_size 40 --dim_feedforward 512 --lr 1e-5 --seed 0 --num_steps 100000 --eval_every 100000 --validate_every 1000 --save_every 10000 --no_encoder --backbone resnet18 --same_backbones --use_pos_embd_image 1 --use_pos_embd_action 1 --dec_layers 6 --gpu_id 0 --feature_loss_weight 0.005 --use_mask --data_aug --wandb --height 480 --width 640

    DATA_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/data' 

    # TASK_CONFIGS = {
    #     # G1 ADDITION
    #     'data_picking':{
    #         'dataset_dir': DATA_DIR + '/data_picking',
    #         'camera_names': ['cam_left', 'cam_right'],
    #         'observation_name': ['qpos'],
    #         'state_dim':23,
    #         'action_dim':23,
    #         'state_mask': [0]*15 + [1]*8,   # Try 15 legs + waist, 8 upper body
    #         'action_mask': [0]*11 + [1]*8
    #     }}



    task_config = {
            'dataset_dir': DATA_DIR + '/data_picking',
            'camera_names': ['cam_left', 'cam_right'],
            'observation_name': ['qpos'],
            'state_dim':23,
            'action_dim':23,
            'state_mask': [0]*15 + [1]*8,   # Try 15 legs + waist, 8 upper body
            'action_mask': [0]*15 + [1]*8
        }
    dataset_dir = task_config['dataset_dir']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)
    randomize_index = task_config.get('randomize_index', False)

    batch_size_train = 10
    batch_size_val = 10
    skip_mirrored_data = False
    width = 640
    height = 480
    normalize_resnet = False
    data_aug = True
    feature_loss_weight = 0.005
    grayscale  = False
    randomize_color = False
    randomize_data_degree = 3
    randomize_data = False
    

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, 
                                                           batch_size_train, batch_size_val, chunk_size, 
                                                           skip_mirrored_data, True, 
                                                           policy_class, stats_dir_l=stats_dir, 
                                                           sample_weights=sample_weights, 
                                                           train_ratio=train_ratio,
                                                           width=width,
                                                           height=height,
                                                           normalize_resnet=normalize_resnet,
                                                           data_aug=data_aug,
                                                           observation_name=task_config['observation_name'],
                                                           feature_loss = feature_loss_weight > 0,
                                                           grayscale = grayscale,
                                                           randomize_color = randomize_color,
                                                            randomize_data_degree = randomize_data_degree,
                                                            randomize_data = randomize_data,
                                                            randomize_index = randomize_index,  
                                                           )


    # # train_dataloader = repeater(train_dataloader)
    # # data = next(train_dataloader)

    # val_dataloader = repeater(val_dataloader)
    # data = next(val_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader) 
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # data = next(train_dataloader)
    # # print(data)

    # image_data, qpos_data, action_data, is_pad = data #qpos: (Batch, Episode length, state_dim)
    # image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()



    # print(image_data.shape)

    # # qpos = torch.zeros(1,23).cuda()
    # # image = torch.zeros(1, 2, 3, 480, 640).cuda()
    # datapoint = 1

    # qpos = qpos_data[datapoint].reshape(1,23)
    # image = image_data[datapoint,:2,:,:,:].reshape(1,2, 3, 480, 640)

    # import cv2
    # cv2.imshow("cam",image_data[datapoint,0,...].permute(1, 2, 0).cpu().numpy())
    # cv2.waitKey(0)
    # cv2.imshow("cam",image_data[datapoint,1,...].permute(1, 2, 0).cpu().numpy())
    # cv2.waitKey(0)
    # # cv2.imshow("cam",image_data[0,2,...].permute(1, 2, 0).cpu().numpy())
    # # cv2.waitKey(0)
    # # cv2.imshow("cam",image_data[0,3,...].permute(1, 2, 0).cpu().numpy())
    # # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    import cv2

    ep = load_episode("/home/schakkal/humanplus/HIT/data/data_picking/default_experiment_98.hdf5")
    for idx, (img, qpos, act, is_pad) in enumerate(iterate_episode(ep)):
        # print(f"Step {idx:4d}: qpos={qpos.numpy()}, action={act.numpy()}, is_pad={is_pad}")

        img0 = img[0].permute(2, 0, 1).cpu().numpy()

        img1 = img[1].permute(2, 0, 1).cpu().numpy()
        
        combined = np.hstack((img0, img1))

        # Show both in one window
        cv2.imshow("cam0 and cam1", combined)
        cv2.waitKey(0)

        qpos = qpos.reshape(1,23).cuda()
        image = img.permute(0,2,3,1).reshape(1,2, 3, 480, 640).cuda()
        print("qpos ", qpos.shape)
        print("qpos ", image.shape)
        
        action = policy(qpos, image)








    # cv2.imshow("cam",image_data[0,2,...].permute(1, 2, 0).cpu().numpy())
    # cv2.waitKey(0)
    # cv2.imshow("cam",image_data[0,3,...].permute(1, 2, 0).cpu().numpy())
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("qpos ", qpos.shape)
    print("image ", image.shape)
    action = policy(qpos, image)
    torch.set_printoptions(
        threshold=10_000_000,  # allow up to 10M elements before truncating
        edgeitems=3,           # how many items to show at the beginning & end of each dimension
        linewidth=200          # try to keep each printed line under 200 chars
    )
    print("predicted action ", action[0,:,:])
    print("True action ", action_data[0,:, -8:])
    # print("True action ", action_data[0])



if __name__ == '__main__':
    torch.cuda.set_device(0)

    main_train()
    
 