from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
import torch
from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        # file_list: str,
        file_list_HQ: str,
        file_list_LQ: str,        
        # file_list_condition: str,           # 新增的条件文件列表
        file_list_RGB: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        # 以下这些参数原本用于生成LQ图像，现暂时保留
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        # self.file_list = file_list
        # self.image_files = load_file_list(file_list)
        self.file_list_HQ = file_list_HQ
        self.file_list_LQ = file_list_LQ
        # self.file_list_condition = file_list_condition
        self.file_list_RGB = file_list_RGB
        self.image_files_HQ = load_file_list(file_list_HQ)
        self.image_files_LQ = load_file_list(file_list_LQ)
        # self.image_files_condition = load_file_list(file_list_condition)    # 新增的条件文件列表
        self.image_files_RGB = load_file_list(file_list_RGB)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # 保留原有的HQ图像退化生成LQ图像的配置参数
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image
    
    def load_condition_features(self, index: int) -> Optional[np.ndarray]:
        """加载条件特征文件"""
        condition_file = self.image_files_condition[index]
        condition_path = condition_file["image_path"] 
        condition_bytes = None
        max_retry = 5
        while condition_bytes is None:
            if max_retry == 0:
                return None
            condition_bytes = self.file_backend.get(condition_path)
            max_retry -= 1
            if condition_bytes is None:
                time.sleep(0.5)
        # (diffbir) root@aqo889okl8js-0:/nc1test1/tl/project05# python condition_check.py
        # The loaded data is a torch.Tensor.
        # Tensor shape: torch.Size([1, 512])
        # Tensor data:
        # 特征文件是tensor格式
        try:
            # 使用 torch.load 加载 torch.Tensor
            condition_features = torch.load(condition_bytes)  # 用这个实验
            return condition_features
        except Exception as e:
            print(f"Failed to load condition features from {condition_path}: {e}")
            return None

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
            # 加载HQ图像
            img_gt = None
            while img_gt is None:
                # 从HQ文件列表中获取对应图像路径
                image_file_HQ = self.image_files_HQ[index]
                gt_path = image_file_HQ["image_path"]
                prompt = image_file_HQ["prompt"]
                img_gt = self.load_gt_image(gt_path)
                if img_gt is None:
                    print(f"filed to load {gt_path}, try another image")
                    index = random.randint(0, len(self) - 1)    #保留意见len(self.image_files_HQ)

            # 加载LQ图像
            img_lq = None
            while img_lq is None:
                image_file_LQ = self.image_files_LQ[index]
                lq_path = image_file_LQ["image_path"]
                img_lq = self.load_gt_image(lq_path)
                if img_lq is None:
                    print(f"filed to load {lq_path}, try another image")
                    index = random.randint(0, len(self) - 1)

            # 加载RGB图像
            img_rgb = None
            while img_rgb is None:
                image_file_RGB = self.image_files_RGB[index]
                rgb_path = image_file_RGB["image_path"]
                img_rgb = self.load_gt_image(rgb_path)
                if img_rgb is None:
                    print(f"filed to load {rgb_path}, try another image")
                    index = random.randint(0, len(self) - 1)

            img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)

            lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)

            rgb = (img_rgb[..., ::-1] / 255.0).astype(np.float32)

            # condition = self.load_condition_features(index)
            
            # # 测试图像读入是否正确
            # print("gt如下:")
            # print(gt)
            # print("lq如下:")
            # print(lq)
            # time.sleep(300)

            return gt, lq, prompt, rgb

    # def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
    #     # load gt image
    #     img_gt = None
    #     while img_gt is None:
    #         # load meta file
    #         image_file = self.image_files[index]
    #         gt_path = image_file["image_path"]
    #         prompt = image_file["prompt"]
    #         img_gt = self.load_gt_image(gt_path)
    #         if img_gt is None:
    #             print(f"filed to load {gt_path}, try another image")
    #             index = random.randint(0, len(self) - 1)        

    #     # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
    #     img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
    #     h, w, _ = img_gt.shape
    #     if np.random.uniform() < 0.5:
    #         prompt = ""

    #     # ------------------------ generate lq image ------------------------ #
    #     # blur
    #     kernel = random_mixed_kernels(
    #         self.kernel_list,
    #         self.kernel_prob,
    #         self.blur_kernel_size,
    #         self.blur_sigma,
    #         self.blur_sigma,
    #         [-math.pi, math.pi],
    #         noise_range=None,
    #     )
    #     img_lq = cv2.filter2D(img_gt, -1, kernel)
    #     # downsample
    #     scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
    #     img_lq = cv2.resize(
    #         img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
    #     )
    #     # noise
    #     if self.noise_range is not None:
    #         img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
    #     # jpeg compression
    #     if self.jpeg_range is not None:
    #         img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

    #     # resize to original size
    #     img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

    #     # BGR to RGB, [-1, 1]
    #     gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
    #     # BGR to RGB, [0, 1]
    #     lq = img_lq[..., ::-1].astype(np.float32)

    #     return gt, lq, prompt

    def __len__(self) -> int:
        # return len(self.image_files)
        return len(self.image_files_HQ)
