# -*-coding:utf-8-*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from random import random
from dnnlib import camera
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import dnnlib
import legacy
import pickle

from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from torchvision import transforms, utils
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from training.networks import Encoder
import inspect
import collections
import PIL.Image

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

import lpips

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
from training.my_utils import *

data_path = {
    'hpcl':
        {
            'data1':'../dataset/car_dataset_trunc075',
            'data2':'../dataset/mvmv/training_set',
            "data3": '../dataset/compcars'

        },
    'jdt':
        {
            "data1": '/workspace/datasets/car_zj',
            "data2": '../dataset/mvmv/training_set',
            "data3": '../dataset/compcars'
        }
}


# --data=./output/car_dataset_3w_test/images --g_ckpt=car_model.pkl --outdir=../car_stylenrf_output/psp_case2/debug
@click.command()
@click.option("--g_ckpt", type=str, default='./car_model.pkl')
@click.option('--encoder', default='./case1103_2_01/network-snapshot-000050.pkl')
# @click.option('--encoder', 'encoder_pkl', default='./output/case1103_2/v01/checkpoints/network-snapshot-000000.pkl')
@click.option("--which_server", type=str, default='jdt')
@click.option("--batch", type=int, default=8)
@click.option("--local_rank", type=int, default=0)
@click.option("--out_name", type=str, default='default')

def main(g_ckpt, batch, local_rank, encoder, which_server,out_name):
    # local_rank = rank
    # setup(rank, word_size)
    # options_list = click.option()
    # print(options_list)



    random_seed = 25
    np.random.seed(random_seed)
    # torch.autograd.set_detect_anomaly(True)

    # num_gpus = torch.cuda.device_count()  # 自动获取显卡数量
    num_gpus = 1
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)
    # torch.set_default_tensor_type(torch.DoubleTensor)

    data = os.path.join( data_path[which_server]['data3'])
    input_images = os.path.join(data,'source_images')
    pose_dir = os.path.join(data,out_name,'camera_metrics')   # 计算结果存在在这里。
    img_dir = os.path.join(data,out_name,'images')   # 计算结果存在在这里。
    # log_dir = os.path.join(data,'log_output')   # 计算结果存在在这里。
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)

    '''loading models'''
    print('Loading networks from "%s"...' % g_ckpt)
    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
    from training.networks import Generator
    from torch_utils import misc
    with torch.no_grad():
        # if 'insert_layer' not in G.init_kwargs.synthesis_kwargs:  # add new attributions
        #     G.init_kwargs.synthesis_kwargs['insert_layer'] = insert_layer
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)

    testing_type = 0  # 1: E  2: EM  3:E Sep_net
    with dnnlib.util.open_url(encoder) as f:
        print('Loading encoder from "%s"...' % encoder)
        encoder_model = legacy.load_network_pkl(f)
        print(encoder_model.keys())
        if "E" in encoder_model.keys():
            E = encoder_model['E'].to(device)
            testing_type = 1
        if "M" in encoder_model.keys():
            M = encoder_model['M'].to(device)
            testing_type = 2
        if "Sep_net" in encoder_model.keys():
            Sep_net = encoder_model['Sep_net'].to(device)
            testing_type = 3
        if "G" in encoder_model.keys():
            G = encoder_model['G'].to(device)
            print("using G<generator> in Encoder!")

    ''' getting images'''
    import tqdm
    ws_avg = G.mapping.w_avg[None, None, :]
    import cv2
    resolution = 256
    images  = os.listdir(input_images)
    images.sort()
    # images = images[:10]  # testing
    for idx,img in enumerate(tqdm.tqdm(images)):
        img_path = os.path.join(input_images,img)
        img = np.array(PIL.Image.open(img_path))
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.to(device).to(torch.float32) / 127.5 - 1
        img = img.unsqueeze(0)
        # print(img.shape)

        rec_ws, _ = E(img)
        # gen_img = G.get_final_output(styles=rec_ws + ws_avg, camera_matrices=camera_matrices)
        sep_out = Sep_net(rec_ws)  # return 2 or 4
        sep_ws = sep_out[0]
        sep_ws += ws_avg
        camera_matrices = sep_out[1], sep_out[2], sep_out[3], None
        gen_img_Sep = G.get_final_output(styles=sep_ws, camera_matrices=camera_matrices)
        show_list = [img.detach(),gen_img_Sep.detach()]

        camera_0 = sep_out[1].detach().cpu().numpy()
        camera_1 = sep_out[2].detach().cpu().numpy()
        camera_2 = sep_out[3].detach().cpu().numpy()
        np.savez(os.path.join(pose_dir, f'{idx:0>6d}.npz'), camera_0=camera_0, camera_1=camera_1,
                 camera_2=camera_2)

        import shutil
        dst_name = os.path.join(img_dir, f'{idx:0>6d}.jpg')
        src_name = img_path
        shutil.copyfile(src_name, dst_name)

        # from torchvision import utils
        # root_path = encoder.split('.')[1]
        # root_path = root_path.split('/')
        # file_name = '-'.join(root_path[2:])
        #
        # # os.makedirs(f"{store_dir}/sample", exist_ok=True)
        # with torch.no_grad():
        #     sample = torch.cat(show_list)
        #     utils.save_image(
        #         sample,
        #         f"{log_dir}/{file_name}_{random_seed}_{idx}.png",
        #         nrow=int(batch),
        #         normalize=True,
        #         range=(-1, 1),
        #     )




if __name__ == "__main__":
    main()
