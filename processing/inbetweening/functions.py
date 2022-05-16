import os
import torch
import torch.nn as nn
import numpy as np
from .skeleton.quaternion import qeuler_np
from .foot_sliding.BVH import *


def PLU(x, alpha = 0.1, c = 1.0):
    relu = nn.ReLU()
    o1 = alpha * (x + c) - c
    o2 = alpha * (x - c) + c
    o3 = x - relu(x - o2)
    o4 = relu(o1 - o3) + o3
    return o4


def gen_ztta(timesteps=60, dim=256):
    ztta = np.zeros((timesteps, dim))
    for t in range(timesteps):
        for d in range(dim):
            if d % 2 == 0:
                ztta[t, d] = np.sin(t / (10000 ** (d // dim)))
            else:
                ztta[t, d] = np.cos(t / (10000 ** (d // dim)))
    return torch.from_numpy(ztta.astype(float))


def write_to_bvhfile(data, filename, joints_to_remove):
    fout = open(filename, 'w')
    line_cnt = 0
    for line in open('./processing/inbetweening/lafan1/example.bvh', 'r'):
        fout.write(line)
        line_cnt += 1
        if line_cnt >= 132:
            break
    fout.write(('Frames: %d\n' % data.shape[0]))
    fout.write('Frame Time: 0.033333\n')
    pose_data = qeuler_np(data[:,3:].reshape(data.shape[0], -1, 4), order='zyx', use_gpu=False)
    pose_data = pose_data / np.pi * 180.0
    for t in range(data.shape[0]):
        line = '%f %f %f ' % (data[t, 0], data[t, 1], data[t, 2])
        for d in range(pose_data.shape[1] - 1):
            line += '%f %f %f ' % (pose_data[t, d, 2], pose_data[t, d, 1], pose_data[t, d, 0])
        line += '%f %f %f\n' % (pose_data[t, -1, 2], pose_data[t, -1, 1], pose_data[t, -1, 0])
        fout.write(line)
    fout.close()
        

def change_bvh_axis(file_path):
    anim, names, frametime = load(file_path)
    tmp = anim.offsets.copy()
    anim.offsets[..., 2] = tmp[..., 0]
    anim.offsets[..., 0] = tmp[..., 1]
    anim.offsets[..., 1] = tmp[..., 2]

    tmp = anim.positions.copy()
    anim.positions[..., 2] = tmp[..., 0]
    anim.positions[..., 0] = tmp[..., 1]
    anim.positions[..., 1] = tmp[..., 2]

    tmp = anim.rotations.qs.copy()
    anim.rotations.qs[..., 3] = tmp[..., 1]
    anim.rotations.qs[..., 1] = tmp[..., 2]
    anim.rotations.qs[..., 2] = tmp[..., 3]

    # joint_num = anim.rotations.shape[1]
    # frame_num = anim.rotations.shape[0]

    save_file_name = file_path.replace('.bvh', '_flipped.bvh')
    save(filename= save_file_name, anim=anim, names=names, frametime=frametime)

    return save_file_name