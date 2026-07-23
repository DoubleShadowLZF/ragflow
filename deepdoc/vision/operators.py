#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
图像预处理算子模块

提供 OCR 检测和识别流程中的图像预处理操作，包括：
- 图像解码与色彩空间转换
- 图像归一化（减均值、除标准差）
- 尺寸调整（保持比例/固定尺寸/限制边长）
- 通道格式转换（HWC ↔ CHW）
- 边缘填充（Pad/Stride 对齐）
- NMS（非极大值抑制）

所有这些算子作为可调用对象实现，通过 transform() 函数链式调用。
"""

import logging
import sys
import ast
import six
import cv2
import numpy as np
import math
from PIL import Image
from rag.utils.lazy_image import ensure_pil_image


class DecodeImage:
    """
    图像解码算子。

    将原始字节流解码为 numpy 数组格式的图像，支持：
    - RGB / GRAY 色彩模式转换
    - HWC ↔ CHW 通道顺序切换
    - 忽略 EXIF 旋转信息

    Attributes:
        img_mode: 目标色彩模式（'RGB' 或 'GRAY'）
        channel_first: 是否将通道维放在第一维（CHW 格式）
        ignore_orientation: 是否忽略 EXIF 旋转信息
    """

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        """执行图像解码"""
        img = data['image']
        if six.PY2:
            assert isinstance(img, str) and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert isinstance(img, bytes) and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION |
                               cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class StandardizeImag:
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class NormalizeImage:
    """
    图像归一化算子。

    对图像执行标准化操作：img = (img * scale - mean) / std
    使用 ImageNet 默认均值和标准差。

    Args:
        scale: 缩放因子（默认为 1/255）
        mean: 均值，默认 [0.485, 0.456, 0.406]（ImageNet）
        std: 标准差，默认 [0.229, 0.224, 0.225]（ImageNet）
        order: 通道顺序 'chw' 或 'hwc'
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            try:
                scale = float(scale)
            except ValueError:
                if '/' in scale:
                    parts = scale.split('/')
                    scale = ast.literal_eval(parts[0]) / ast.literal_eval(parts[1])
                else:
                    scale = ast.literal_eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        pil = ensure_pil_image(img)
        if isinstance(pil, Image.Image):
            img = np.array(pil)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage:
    """
    通道顺序转换算子。

    将图像从 HWC 格式转换为 CHW 格式（通道维前置），
    这是大多数深度学习模型期望的输入格式。
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        pil = ensure_pil_image(img)
        if isinstance(pil, Image.Image):
            img = np.array(pil)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys:
    """
    键过滤算子。

    从数据字典中只保留指定的键，过滤掉不需要的中间数据。
    通常作为预处理管道的最后一步，只保留模型需要的输入。

    Args:
        keep_keys: 需要保留的键名列表
    """
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class Pad:
    """
    图像填充算子。

    将图像填充到指定大小或使其尺寸能被 size_div 整除。
    通常用于确保特征图尺寸与网络 stride 对齐。

    Args:
        size: 目标尺寸 [H, W]，若为 None 则只对齐到 size_div 的倍数
        size_div: 尺寸对齐因子（默认 32）
    """
    def __init__(self, size=None, size_div=32, **kwargs):
        if size is not None and not isinstance(size, (int, list, tuple)):
            raise TypeError("Type of target_size is invalid. Now is {}".format(
                type(size)))
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.size_div = size_div

    def __call__(self, data):

        img = data['image']
        img_h, img_w = img.shape[0], img.shape[1]
        if self.size:
            resize_h2, resize_w2 = self.size
            assert (
                img_h < resize_h2 and img_w < resize_w2
            ), '(h, w) of target size should be greater than (img_h, img_w)'
        else:
            resize_h2 = max(
                int(math.ceil(img.shape[0] / self.size_div) * self.size_div),
                self.size_div)
            resize_w2 = max(
                int(math.ceil(img.shape[1] / self.size_div) * self.size_div),
                self.size_div)
        img = cv2.copyMakeBorder(
            img,
            0,
            resize_h2 - img_h,
            0,
            resize_w2 - img_w,
            cv2.BORDER_CONSTANT,
            value=0)
        data['image'] = img
        return data


class LinearResize:
    """
    线性缩放算子。

    按目标尺寸缩放图像，可选保持宽高比。如果开启 keep_ratio，
    则按较短边缩放到目标尺寸，较长边按比例计算但不超过目标最大边。

    Args:
        target_size: 目标尺寸 [H, W] 或单个整数
        keep_ratio: 是否保持宽高比
        interp: OpenCV 插值方法，默认 cv2.INTER_LINEAR
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        _im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        _im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class Resize:
    """
    直接缩放算子。

    将图像和对应的多边形标注按比例缩放到指定尺寸。

    Args:
        size: 目标尺寸 (H, W)，默认 (640, 640)
    """
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data):
        img = data['image']
        if 'polys' in data:
            text_polys = data['polys']

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        if 'polys' in data:
            new_boxes = []
            for box in text_polys:
                new_box = []
                for cord in box:
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            data['polys'] = np.array(new_boxes, dtype=np.float32)
        data['image'] = img_resize
        return data


class DetResizeForTest:
    """
    文本检测推理时的图像缩放算子。

    支持三种缩放模式：
    - type 0 (limit_side_len): 限制最长/最短边到指定长度，并对齐到 32 的倍数
    - type 1 (image_shape): 缩放到固定尺寸
    - type 2 (resize_long): 限制长边到指定长度

    对小图（宽高和 < 64）会自动填充到 32×32 以上。
    """

    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.keep_ratio = False
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
            if 'keep_ratio' in kwargs:
                self.keep_ratio = kwargs['keep_ratio']
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except BaseException:
            logging.exception("{} {} {}".format(img.shape, resize_w, resize_h))
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class E2EResizeForTest:
    """
    端到端推理时的图像缩放算子。

    将图像长边限制到 max_side_len，并对齐到 128 的倍数。
    针对 totaltext 数据集有特殊的缩放比例（1.25 倍）。
    """

    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs['max_side_len']
        self.valid_set = kwargs['valid_set']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.valid_set == 'totaltext':
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len)
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len)
        data['image'] = im_resized
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):
        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)


class KieResize:
    """
    KIE（关键信息提取）推理时的图像缩放算子。

    将图像缩放到 1024×1024 画布内，对齐到 32 的倍数，
    同时同步缩放对应的标注点坐标。
    """

    def __init__(self, **kwargs):
        super(KieResize, self).__init__()
        self.max_side, self.min_side = kwargs['img_scale'][0], kwargs[
            'img_scale'][1]

    def __call__(self, data):
        img = data['image']
        points = data['points']
        src_h, src_w, _ = img.shape
        im_resized, scale_factor, [ratio_h, ratio_w
                                   ], [new_h, new_w] = self.resize_image(img)
        resize_points = self.resize_boxes(img, points, scale_factor)
        data['ori_image'] = img
        data['ori_boxes'] = points
        data['points'] = resize_points
        data['image'] = im_resized
        data['shape'] = np.array([new_h, new_w])
        return data

    def resize_image(self, img):
        norm_img = np.zeros([1024, 1024, 3], dtype='float32')
        scale = [512, 1024]
        h, w = img.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        resize_w, resize_h = int(w * float(scale_factor) + 0.5), int(h * float(
            scale_factor) + 0.5)
        max_stride = 32
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(img, (resize_w, resize_h))
        new_h, new_w = im.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        norm_img[:new_h, :new_w, :] = im
        return norm_img, scale_factor, [h_scale, w_scale], [new_h, new_w]

    def resize_boxes(self, im, points, scale_factor):
        points = points * scale_factor
        img_shape = im.shape[:2]
        points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
        points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])
        return points


class SRResize:
    def __init__(self,
                 imgH=32,
                 imgW=128,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 infer_mode=False,
                 **kwargs):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.infer_mode = infer_mode

    def __call__(self, data):
        imgH = self.imgH
        imgW = self.imgW
        images_lr = data["image_lr"]
        transform2 = ResizeNormalize(
            (imgW // self.down_sample_scale, imgH // self.down_sample_scale))
        images_lr = transform2(images_lr)
        data["img_lr"] = images_lr
        if self.infer_mode:
            return data

        images_HR = data["image_hr"]
        _label_strs = data["label"]
        transform = ResizeNormalize((imgW, imgH))
        images_HR = transform(images_HR)
        data["img_hr"] = images_HR
        return data


class ResizeNormalize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_numpy = np.array(img).astype("float32")
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        return img_numpy


class GrayImageChannelFormat:
    """
    灰度图像通道格式化算子。

    将三通道彩色图像转换为单通道灰度图像。
    格式转换：(3, H, W) → (1, H, W)

    Args:
        inverse: 是否反转灰度值（生成负片效果）
    """

    def __init__(self, inverse=False, **kwargs):
        self.inverse = inverse

    def __call__(self, data):
        img = data['image']
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_expanded = np.expand_dims(img_single_channel, 0)

        if self.inverse:
            data['image'] = np.abs(img_expanded - 1)
        else:
            data['image'] = img_expanded

        data['src_image'] = img
        return data


class Permute:
    """
    通道置换算子。

    将图像从 HWC 格式转换为 CHW 格式（通道维前置）。
    这是大多数深度学习模型的标准输入格式。

    Args:
        to_bgr (bool): 是否将 RGB 转换为 BGR（未使用）
        channel_first (bool): 是否将 HWC 转为 CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """执行通道置换：HWC → CHW"""
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


class PadStride:
    """
    Stride 对齐填充算子。

    将图像尺寸填充到 stride 的整数倍，满足 FPN 等特征金字塔网络的要求。

    Args:
        stride: 对齐步长，默认 0（不填充）
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return im, im_info
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im, im_info


def decode_image(im_file, im_info):
    """
    读取 RGB 图像。

    支持文件路径字符串和 numpy 数组两种输入格式。

    Args:
        im_file (str|np.ndarray): 图像文件路径或 numpy 数组
        im_info (dict): 图像信息字典，会被写入 'im_shape' 和 'scale_factor'

    Returns:
        tuple: (im, im_info) - RGB 格式的 numpy 数组和更新后的信息字典
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


def preprocess(im, preprocess_ops):
    """
    通过预处理算子链处理图像。

    按顺序应用预处理操作列表，将图像从原始格式转换为模型输入格式。

    Args:
        im: 输入图像（文件路径或 numpy 数组）
        preprocess_ops: 预处理算子列表

    Returns:
        tuple: (im, im_info) - 处理后的图像和元信息
    """
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info


def nms(bboxes, scores, iou_thresh):
    """
    非极大值抑制（NMS）。

    用于去除重叠的检测框，保留置信度最高的框。
    通过贪心策略按分数从高到低排序，逐个移除 IoU 超过阈值的框。

    Args:
        bboxes: 边界框数组，shape (N, 4)，每行为 [x1, y1, x2, y2]
        scores: 每个框的置信度分数
        iou_thresh: IoU 阈值，超过此值的重叠框将被抑制

    Returns:
        list: 保留的框索引列表
    """
    import numpy as np
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    indices = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        indices.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]
    return indices
