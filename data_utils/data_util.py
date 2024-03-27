import numpy as np
import torch
from Motion.transforms import quat2repr6d
from Motion import BVH
from scipy import io
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from data_utils.babel.data.datautils.babel_label import label as BABEL_label
import json 
import os 
from collections import defaultdict
from data_utils.mixamo.motion import MotionData
import torch.nn.functional as F


def auto_label_mapping(labels):
    label_map = defaultdict(lambda: len(label_map))
    new_labels = [label_map[label] for label in labels]
    return new_labels



def lerp_sample(motion, size=None):
    motion = F.interpolate(motion, size=size, mode='linear', align_corners=False)
    return motion


def load_sin_motion(args):
    motion_data = {}    
    suffix = args.sin_path.lower()[-4:]
    # assert suffix in ['.mat','.bvh']
    if args.dataset == 'humanml':
        assert suffix == '.npy'
        try:
            motion = np.load(args.sin_path)  # only motion npy
            if len(motion.shape) == 2:
                motion = np.transpose(motion)
                motion = np.expand_dims(motion, axis=1)

        except:
            motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype
        
    elif args.dataset == 'babel':
        with open(args.sin_path, 'rb') as f : 
            data = pkl.load(f)
        BABEL_label_rev = {v:k for k, v in BABEL_label.items()}
        rot = data['smpl_param']
        trs = data['smpl_trans']        
        motion_length = data['data_len']
        sub_len = data['sub_len'].cpu().numpy()
        org_labels = data['label_mask'].cpu().numpy()
        labels = auto_label_mapping(org_labels)
        label_seq = [BABEL_label_rev[int(i)] for i in data['label_mask']]

        motion_data['label_seq'] = label_seq
        motion_data['labels'] = labels
        motion_data['sub_len'] = sub_len.tolist()
        with open(os.path.join(args.save_path,'label.json'), 'w') as f :
            json.dump(motion_data, f, indent=4)
        pad_trs = torch.zeros([motion_length, 3])
        padded_trs = torch.cat([trs,pad_trs],1)
        motion = torch.cat([rot, padded_trs],axis=1).unsqueeze(-1)
        motion = motion.permute(1,2,0)
        motion = motion.to(torch.float32)
        

    elif args.dataset == 'mixamo':  # bvh
        assert suffix == '.bvh'
        # 174 - 24 joint rotations (6d) + 3 root translation + 6*4 foot contact labels + 3 padding
        repr = 'repr6d' if args.repr == '6d' else 'quat'
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        
        _, raw_motion_joints, raw_motion_frames = motion_data.raw_motion.shape
        motion = motion_data.raw_motion.squeeze()        
    else:
        assert args.dataset == 'bvh_general' and suffix == '.bvh'
        anim, _, _ = BVH.load(args.sin_path)
        if args.repr == '6d':
            repr_6d = quat2repr6d(torch.tensor(anim.rotations.qs))
            motion = np.concatenate([anim.positions, repr_6d], axis=2)
        else:
            motion = np.concatenate([anim.positions, anim.rotations.qs], axis=2)
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 2, 0)  # n_frames x n_joints x n_feats  ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype


    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
    
    motion_data['neighbor_list'] = get_neighbor(parents[:26])
    
    motion = motion.to(args.device)
    return motion, motion_data
def dfs(parents, x, vis, dist):
    fa = parents
    vis[x] = 1
    for y in range(len(fa)):
        if (fa[y] == x or fa[x] == y) and vis[y] == 0:
            dist[y] = dist[x] + 1
            dfs(parents, y, vis, dist)

def get_neighbor(parents, threshold=2, enforce_contact=False):
    fa = parents
    neighbor_list = []
    for x in range(0, len(fa)):
        vis = [0 for _ in range(len(fa))]
        dist = [0 for _ in range(len(fa))]
        dfs(parents, x, vis, dist)
        neighbor = []
        for j in range(0, len(fa)):
            if dist[j] <= threshold:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    contact_list = []

    root_neighbor = neighbor_list[0]
    id_root = len(neighbor_list)

    root_neighbor = list(set(root_neighbor))
    for j in root_neighbor:
        neighbor_list[j].append(id_root)
    root_neighbor.append(id_root)
    neighbor_list.append(root_neighbor)  # Neighbor for root position
    return neighbor_list

def sample_sin_motion(args):
    motion_data = {}    
    # assert suffix in ['.mat','.bvh']
    # if args.dataset == 'babel':
    with open(args.babel_pkl_path, 'rb') as f : 
        data = pkl.load(f)
    BABEL_label_rev = {v:k for k, v in BABEL_label.items()}
    rot = data['smpl_param']
    trs = data['smpl_trans']        
    motion_length = data['data_len']
    sub_len = data['sub_len'].cpu().numpy()
    org_labels = data['label_mask'].cpu().numpy()
    labels = auto_label_mapping(org_labels)
    label_seq = [BABEL_label_rev[int(i)] for i in data['label_mask']]
    motion_data['label_seq'] = label_seq
    motion_data['labels'] = labels
    motion_data['sub_len'] = sub_len
    pad_trs = torch.zeros([motion_length, 3])
    padded_trs = torch.cat([trs,pad_trs],1)
    motion = torch.cat([rot, padded_trs],axis=1).unsqueeze(-1)
    motion = motion.permute(1,2,0)
    motion = motion.to(torch.float32)

    motion = motion.to(args.device)
    return motion, motion_data


def load_seg_lerp_motion(args):
    motion_data = None
    with open(args.pkl_path, 'rb') as f:
        data = pkl.load(f)
    print(data.keys())
    motion = np.array(data['data'])
    motion = torch.from_numpy(motion).to(torch.float32) 
    motion = motion.to(args.device)
    
    labels = np.array(data['labels'])
    labels = torch.from_numpy(labels).to(torch.float32)
    labels = labels.to(args.device)
    
    return motion, labels, motion_data


def load_seg_motion(args):
    motion_data = None
    with open(args.pkl_path, 'rb') as f:
        data = pkl.load(f)

    motion = data['input'] 
    motion = torch.from_numpy(motion).to(torch.float32) 
    motion = motion.to(args.device)
    
    labels = data['pred']
    labels = torch.from_numpy(labels).to(torch.float32)
    labels = labels.to(args.device)
    
    return motion, labels, motion_data

