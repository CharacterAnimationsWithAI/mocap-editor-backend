import os
import torch
import torch.nn as nn
import numpy as np
from .skeleton.quaternion import qeuler_np


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
    for line in open('./lafan1/example.bvh', 'r'):
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
        