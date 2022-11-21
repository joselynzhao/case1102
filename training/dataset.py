# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from curses import raw
import os
from urllib import response
import numpy as np
import zipfile
import PIL.Image
import cv2
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self.xflip = xflip
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx), idx

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None:
            raw_shape[2] = raw_shape[3] = resolution
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        if hasattr(self, '_raw_shape') and image.shape[-1] != self.resolution:  # resize input image
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_dali_dataloader(self, batch_size, world_size, rank, gpu):  # TODO
        from nvidia.dali import pipeline_def, Pipeline
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        
        @pipeline_def
        def pipeline():
            jpegs, _ = fn.readers.file(
                file_root=self._path,
                files=list(self._all_fnames),
                random_shuffle=True,
                shard_id=rank, 
                num_shards=world_size, 
                name='reader')
            images = fn.decoders.image(jpegs, device='mixed')
            mirror = fn.random.coin_flip(probability=0.5) if self.xflip else False
            images = fn.crop_mirror_normalize(
                images.gpu(), output_layout="CHW", dtype=types.UINT8, mirror=mirror)
            labels = np.zeros([1, 0], dtype=np.float32)
            return images, labels
        
        dali_pipe = pipeline(batch_size=batch_size//world_size, num_threads=2, device_id=gpu)
        dali_pipe.build()
        training_set_iterator = DALIGenericIterator([dali_pipe], ['img', 'label'])
        for data in training_set_iterator:
            yield data[0]['img'], data[0]['label']

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
class ImageFolderDataset_psp_case1(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path # 直接给到image所在目录
        path_root = os.path.abspath(os.path.join(path, '..'))
        self.camera_path = os.path.join(path_root,'cameras')
        self.w_path = os.path.join(path_root,'ws')
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None:
            raw_shape[2] = raw_shape[3] = resolution
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        img1,img2,camera1,camera2,w= self._load_raw_image_psp_case1(self._raw_idx[idx])
        assert isinstance(img1, np.ndarray)
        assert list(img1.shape) == self.image_shape
        assert img1.dtype == np.uint8
        # if self._xflip[idx]:
        #     assert image.ndim == 3  # CHW
        #     image = image[:, :, ::-1]
        return img1.copy(), img2.copy(), camera1, camera2,w

    def _load_raw_image_psp_case1(self, raw_idx):
        fname = self._image_fnames[raw_idx]  # 000000——00
        image_name = fname.split('.')[0]
        ID = image_name.split('_')[0]
        step = int(image_name.split('_')[1])
        # pair_step = step+4
        # if pair_step>15:
        #     pair_step=pair_step-16
        pair_step = 0 if step==15 else step+1  # 取相邻的两个视角
        # while (step == pair_step):
        #     pair_step = np.random.randint(0, 16)
        pair_name = f'{ID}_{pair_step:02d}'
        pair_image = pair_name+'.png'
        # pair_dir = '{}_0{}.png'.format(ID, pair_step)
        def open_image(fname):
            with self._open_file(fname) as f:
                if pyspng is not None and self._file_ext(fname) == '.png':
                    image = pyspng.load(f.read())  #这条路线
                else:
                    image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
            if hasattr(self, '_raw_shape') and image.shape[-1] != self.resolution:  # resize input image
                image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            image = image.transpose(2, 0, 1)  # HWC => CHW
            return image
        image1 = open_image(fname)
        image2 = open_image(pair_image)
        # 提取w信息
        ws = '{}/{}.npz'.format(self.w_path,ID.split('.')[0])
        ws= np.load(ws)['ws'] # 19,512
        # 提取camera信息
        camera1  ='{}/{}.npz'.format(self.camera_path,image_name)
        camera1 = np.load(camera1)
        camera2  = '{}/{}.npz'.format(self.camera_path,pair_name)
        camera2 = np.load(camera2)
        return image1,image2,camera1,camera2,ws

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]  # 000000——00
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        if hasattr(self, '_raw_shape') and image.shape[-1] != self.resolution:  # resize input image
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_dali_dataloader(self, batch_size, world_size, rank, gpu):  # TODO
        from nvidia.dali import pipeline_def, Pipeline
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator

        @pipeline_def
        def pipeline():
            jpegs, _ = fn.readers.file(
                file_root=self._path,
                files=list(self._all_fnames),
                random_shuffle=True,
                shard_id=rank,
                num_shards=world_size,
                name='reader')
            images = fn.decoders.image(jpegs, device='mixed')
            mirror = fn.random.coin_flip(probability=0.5) if self.xflip else False
            images = fn.crop_mirror_normalize(
                images.gpu(), output_layout="CHW", dtype=types.UINT8, mirror=mirror)
            labels = np.zeros([1, 0], dtype=np.float32)
            return images, labels

        dali_pipe = pipeline(batch_size=batch_size // world_size, num_threads=2, device_id=gpu)
        dali_pipe.build()
        training_set_iterator = DALIGenericIterator([dali_pipe], ['img', 'label'])
        for data in training_set_iterator:
            yield data[0]['img'], data[0]['label']


#----------------------------------------------------------------------------
class ImageFolderDataset_mvmc_zj(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 which_camera='init',
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path # 直接给到image所在目录
        path_root = os.path.abspath(os.path.join(path, '..'))
        self.camera_path = os.path.join(path_root,'cameras_init') if which_camera=='init' else os.path.join(path_root,'cameras_opt')
        print("mvmc dataset training with ",self.camera_path)
        self.w_path = os.path.join(path_root,'ws')
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None:
            raw_shape[2] = raw_shape[3] = resolution
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        img1,img2,camera1,camera2= self._load_raw_image_mvmc_zj(self._raw_idx[idx])
        assert isinstance(img1, np.ndarray)
        # print(img1.shape,self.image_shape)
        # assert list(img1.shape) == self.image_shape
        assert img1.dtype == np.uint8
        # if self._xflip[idx]:
        #     assert image.ndim == 3  # CHW
        #     image = image[:, :, ::-1]
        return img1.copy(), img2.copy(), camera1, camera2

    def _load_raw_image_mvmc_zj(self, raw_idx):
        # print("loading data in mvmc_zj")
        fname = self._image_fnames[raw_idx]  # 000000——00
        image_name = fname.split('.')[0]
        ID = image_name.split('_')[0]
        step = int(image_name.split('_')[1])
        '''取不了相邻的视角'''
        # pair_step = step+4
        # if pair_step>15:
        #     pair_step=pair_step-16
        # pair_step = 0 if step==15 else step+1  # 取相邻的两个视角
        pair_step = step
        pair_name = pair_image= ''
        while (step == pair_step):
            pair_step = np.random.randint(0, 16)
            pair_name = f'{ID}_{pair_step:02d}'
            pair_image = pair_name+'.jpg'
            if not os.path.exists(os.path.join(self._path, pair_image)):
                pair_step = step
        # pair_dir = '{}_0{}.png'.format(ID, pair_step)
        def open_image(fname):
            with self._open_file(fname) as f:
                # if pyspng is not None and self._file_ext(fname) == '.jpg':
                #     image = pyspng.load(f.read())
                # else:
                image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
            if image.shape[-1] != 256:  # resize input image
            # if hasattr(self, '_raw_shape') and image.shape[-1] != 256:  # resize input image
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = image.transpose(2, 0, 1)  # HWC => CHW
            return image
        image1 = open_image(fname)
        image2 = open_image(pair_image)
        # 提取w信息
        # ws = '{}/{}.npz'.format(self.w_path,ID.split('.')[0])
        # ws= np.load(ws)['ws'] # 19,512
        # 提取camera信息
        camera1  ='{}/{}.npz'.format(self.camera_path,image_name)
        camera1 = np.load(camera1)
        camera2  = '{}/{}.npz'.format(self.camera_path,pair_name)
        camera2 = np.load(camera2)
        return image1,image2,camera1,camera2

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]  # 000000——00
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.jpg':
            #     image = pyspng.load(f.read())
            # else:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        if hasattr(self, '_raw_shape') and image.shape[-1] != self.resolution:  # resize input image
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_dali_dataloader(self, batch_size, world_size, rank, gpu):  # TODO
        from nvidia.dali import pipeline_def, Pipeline
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator

        @pipeline_def
        def pipeline():
            jpegs, _ = fn.readers.file(
                file_root=self._path,
                files=list(self._all_fnames),
                random_shuffle=True,
                shard_id=rank,
                num_shards=world_size,
                name='reader')
            images = fn.decoders.image(jpegs, device='mixed')
            mirror = fn.random.coin_flip(probability=0.5) if self.xflip else False
            images = fn.crop_mirror_normalize(
                images.gpu(), output_layout="CHW", dtype=types.UINT8, mirror=mirror)
            labels = np.zeros([1, 0], dtype=np.float32)
            return images, labels

        dali_pipe = pipeline(batch_size=batch_size // world_size, num_threads=2, device_id=gpu)
        dali_pipe.build()
        training_set_iterator = DALIGenericIterator([dali_pipe], ['img', 'label'])
        for data in training_set_iterator:
            yield data[0]['img'], data[0]['label']


#----------------------------------------------------------------------------
class ImageFolderDataset_compcars(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path # 直接给到image所在目录
        path_root = os.path.abspath(os.path.join(path, '..'))
        self.camera_path = os.path.join(path_root,'camera_metrics')
        self.w_path = os.path.join(path_root,'ws')
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None:
            raw_shape[2] = raw_shape[3] = resolution
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        img1,camera1= self._load_raw_image_compcars(self._raw_idx[idx])
        assert isinstance(img1, np.ndarray)
        # print(img1.shape,self.image_shape)
        # assert list(img1.shape) == self.image_shape
        assert img1.dtype == np.uint8
        # if self._xflip[idx]:
        #     assert image.ndim == 3  # CHW
        #     image = image[:, :, ::-1]
        return img1.copy(), camera1,
    def _load_raw_image_compcars(self, raw_idx):
        # print("loading data in mvmc_zj")
        fname = self._image_fnames[raw_idx]  # 000000——00
        image_name = fname.split('.')[0] # 去掉扩展名
        ID = image_name.split('_')[0]

        def open_image(fname):
            with self._open_file(fname) as f:
                image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
            if image.shape[-1] != 256:  # resize input image
            # if hasattr(self, '_raw_shape') and image.shape[-1] != 256:  # resize input image
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = image.transpose(2, 0, 1)  # HWC => CHW
            return image
        image1 = open_image(fname)
        # 提取w信息
        # ws = '{}/{}.npz'.format(self.w_path,ID.split('.')[0])
        # ws= np.load(ws)['ws'] # 19,512
        # 提取camera信息
        camera1  ='{}/{}.npz'.format(self.camera_path,image_name)
        camera1 = np.load(camera1)
        return image1,camera1

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]  # 000000——00
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.jpg':
            #     image = pyspng.load(f.read())
            # else:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        if hasattr(self, '_raw_shape') and image.shape[-1] != self.resolution:  # resize input image
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_dali_dataloader(self, batch_size, world_size, rank, gpu):  # TODO
        from nvidia.dali import pipeline_def, Pipeline
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator

        @pipeline_def
        def pipeline():
            jpegs, _ = fn.readers.file(
                file_root=self._path,
                files=list(self._all_fnames),
                random_shuffle=True,
                shard_id=rank,
                num_shards=world_size,
                name='reader')
            images = fn.decoders.image(jpegs, device='mixed')
            mirror = fn.random.coin_flip(probability=0.5) if self.xflip else False
            images = fn.crop_mirror_normalize(
                images.gpu(), output_layout="CHW", dtype=types.UINT8, mirror=mirror)
            labels = np.zeros([1, 0], dtype=np.float32)
            return images, labels

        dali_pipe = pipeline(batch_size=batch_size // world_size, num_threads=2, device_id=gpu)
        dali_pipe.build()
        training_set_iterator = DALIGenericIterator([dali_pipe], ['img', 'label'])
        for data in training_set_iterator:
            yield data[0]['img'], data[0]['label']
