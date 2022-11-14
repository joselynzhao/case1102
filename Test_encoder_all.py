# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-#-*-coding:utf-8-*-# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import re
import time
import glob
import copy
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
import cv2
from renderer import Renderer
from training.my_utils import *

# ----------------------------------------------------------------------------
# hpcl
torch.cuda.current_device()
torch.cuda._initialized = True
# torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

data1 = {
    'hpcl': '../dataset/car_dataset_trunc075/trunc075/images',
    'jdt': '/workspace/datasets/car_zj/trunc075/images'
}

data2 = {
    'hpcl': '../dataset/mvmv/testing_set/images',
    'jdt': '../dataset/mvmv/testing_set/images'
}


# 不需要指定任何参数，只需要修改待测试encoder的列表，主要调整seed（选择测试样列）
@click.command()
@click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename', default='./car_model.pkl')
@click.option("--g_ckpt", type=str, default='./car_model.pkl')
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename',
              default='./output/case1020_2/debug/checkpoints/network-snapshot-000000.pkl')
@click.option("--which_server", type=str, default='jdt')
@click.option("--testing_set", type=int, default=1)  # 1 : trun075  2: mvmc
# @click.option('--insert_layer',type=int, default=3)
# @click.option('--group_name',type=str, default='01')
@click.option("--batch", type=int, default=8)
@click.option("--random_seed", type=int, default=8)
# @click.option("--which_c", type=str, default='p2')
@click.option("--local_rank", type=int, default=0)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR',
#               default='../output/test_encoders/show_psp_case2_encoder_v4')
def generate_images(
        ctx: click.Context,
        # network_pkl: str,
        g_ckpt: str,
        encoder_pkl: str,
        testing_set: int,
        # data: str,
        # insert_layer:int,
        batch: int,
        # which_c: str,
        local_rank: int,
        class_idx: Optional[int],
        random_seed: int,
        which_server: str):
    data_path = data1 if testing_set == 1 else data2
    data = data_path[which_server]
    np.random.seed(random_seed)

    num_gpus = 1  # 自动获取显卡数量
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')

    # store_dir = os.path.join(f'./output/test_encoders/In_testset/Group_{group_name}')  # 用于服务器测试
    data_name = 'trunc075' if testing_set == 1 else "mvmc"
    store_dir = f'./output/test_encoders/{data_name}'
    os.makedirs(store_dir, exist_ok=True)

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

    testing_type=0 # 1: E  2: EM  3:E Sep_net
    with dnnlib.util.open_url(encoder_pkl) as f:
        print('Loading encoder from "%s"...' % encoder_pkl)
        encoder = legacy.load_network_pkl(f)
        print(encoder.keys())
        if "E" in encoder.keys():
            E = encoder['E'].to(device)
            testing_type=1
        if "M" in encoder.keys():
            M = encoder['M'].to(device)
            testing_type=2
        if "Sep_net" in encoder.keys():
            Sep_net = encoder['Sep_net'].to(device)
            testing_type=3
        if "G" in encoder.keys():
            G =  encoder['G'].to(device)
            print("using G<generator> in Encoder!")
            
            
    print("testing_type:", testing_type)

    np.random.seed(random_seed)
    # load the dataset
    # data_dir = os.path.join(data, 'images')
    dataclass_name = 'training.dataset.ImageFolderDataset_psp_case1' if testing_set == 1 else 'training.dataset.ImageFolderDataset_mvmc_zj'
    training_set_kwargs = dict(class_name=dataclass_name, path=data, use_labels=False,
                               xflip=True)
    data_loader_kwargs = dict(pin_memory=True, num_workers=1, prefetch_factor=1)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus,
                                                seed=random_seed)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                        batch_size=batch // num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)

    # seed_list = [11]
    for idx in range(6):
        ws_avg = G.mapping.w_avg[None, None, :]
        info = next(training_set_iterator)
        img = info[0]
        camera = info[2]
        img = img.to(device).to(torch.float32) / 127.5 - 1
        if testing_set == 2:
            camera_views = camera['camera_2'][:, :2].to(device).to(torch.float32)
            camera_views[:, 1] = 0.5
            camera_matrices = G.synthesis.get_camera(batch, device=device, mode=camera_views)
            # print(camera)
        else:
            camera_matrices = get_camera_metrices(camera, device)
            camera_views = camera_matrices[2][:, :2].to(device)
            w = info[4].to(device)
            # w +=ws_avg

        rec_ws, _ = E(img)
        gen_img = G.get_final_output(styles=rec_ws + ws_avg, camera_matrices=camera_matrices)
        show_list = [img.detach(),gen_img.detach()]
        if testing_type==2:
            mapping_w = M(rec_ws, camera_views)
            mapping_w += ws_avg
            gen_img_MappingNet = G.get_final_output(styles=mapping_w, camera_matrices=camera_matrices)
            show_list.append(gen_img_MappingNet.detach())
        if testing_type==3:
            sep_ws, sep_mode = Sep_net(rec_ws)
            sep_ws += ws_avg
            gen_img_Sep = G.get_final_output(styles=sep_ws, camera_matrices=camera_matrices)
            show_list.append(gen_img_Sep.detach())
        # if w is not None:
        #     gen_img_w = G.get_final_output(styles=w, camera_matrices=camera_matrices)

        from torchvision import utils
        root_path = encoder_pkl.split('.')[1]
        root_path = root_path.split('/')
        file_name = '-'.join(root_path[2:])

        # os.makedirs(f"{store_dir}/sample", exist_ok=True)
        with torch.no_grad():
            sample = torch.cat(show_list)
            utils.save_image(
                sample,
                f"{store_dir}/{file_name}_{random_seed}_{idx}.png",
                nrow=int(batch),
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
