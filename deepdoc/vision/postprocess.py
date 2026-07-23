#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
后处理模块

提供 OCR 检测和识别结果的后处理功能：
- DBPostProcess: DB（Differentiable Binarization）文本检测后处理
- CTCLabelDecode: CTC 文本识别结果解码
"""

import copy
import re
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


def build_post_process(config, global_config=None):
    """
    根据配置字典构建后处理模块实例。

    支持的模块：
    - DBPostProcess: 文本检测后处理（二值化掩码 → 文本框坐标）
    - CTCLabelDecode: 文本识别后处理（CTC 输出 → 文字字符串）

    Args:
        config: 配置字典，必须包含 'name' 键指定模块名称
        global_config: 可选的全局配置，会合并到模块配置中

    Returns:
        后处理模块实例，若 name 为 "None" 则返回 None

    Raises:
        ValueError: 指定了不支持的模块名称
    """
    support_dict = {'DBPostProcess': DBPostProcess, 'CTCLabelDecode': CTCLabelDecode}

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    module_class = support_dict.get(module_name)
    if module_class is None:
        raise ValueError(
            'post process only support {}'.format(list(support_dict)))
    return module_class(**config)


class DBPostProcess:
    """
    DB（Differentiable Binarization）文本检测后处理。

    将模型输出的概率图和二值化图转换为文本框坐标。
    支持两种检测框类型：
    - poly: 多边形框（适用于弯曲文本）
    - quad: 四边形框（适用于常规文本）

    处理流程：
    1. 对概率图进行二值化（threshold）
    2. 查找二值化图中的连通区域（轮廓检测）
    3. 对每个轮廓进行框扩张（unclip）
    4. 计算框得分，过滤低分框
    5. 将框坐标映射回原始图像尺寸
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        从二值化掩码中提取多边形文本框。

        使用轮廓检测 + 多边形拟合的方式，适用于弯曲文本场景。

        Args:
            pred: 概率图（numpy array）
            _bitmap: 二值化掩码，shape (1, H, W)，值为 {0, 1}
            dest_width: 目标图像宽度（原始图像尺寸）
            dest_height: 目标图像高度（原始图像尺寸）

        Returns:
            tuple: (boxes, scores) - 多边形框列表和对应的置信度分数
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        从二值化掩码中提取四边形文本框。

        使用最小外接矩形的方式，适用于常规文本场景。

        Args:
            pred: 概率图（numpy array）
            _bitmap: 二值化掩码，shape (1, H, W)，值为 {0, 1}
            dest_width: 目标图像宽度
            dest_height: 目标图像高度

        Returns:
            tuple: (boxes, scores) - 四边形框数组和对应的置信度分数列表
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        """
        对文本框进行扩张操作。

        通过 pyclipper 库将框向外扩张，扩张距离 = 面积 * unclip_ratio / 周长。
        这样可以确保紧密贴合文字的框在扩张后完整包含文字区域。

        Args:
            box: 原始四边形框坐标
            unclip_ratio: 扩张比例

        Returns:
            numpy array: 扩张后的框坐标
        """
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        """
        获取轮廓的最小外接矩形框并按顺时针排序。

        Args:
            contour: OpenCV 轮廓点集

        Returns:
            tuple: (box, min_side) - 排序后的四点框和最小边长
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        快速计算文本框置信度分数。

        使用框内区域的平均概率作为分数，速度较快。

        Args:
            bitmap: 概率图
            _box: 文本框四点坐标

        Returns:
            float: 置信度分数（0~1）
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """
        慢速但更精确地计算文本框置信度分数。

        使用多边形内的概率均值作为分数，适用于弯曲文本。

        Args:
            bitmap: 概率图
            contour: 文本轮廓点集

        Returns:
            float: 置信度分数（0~1）
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        """
        执行 DB 后处理，将模型输出转换为文本框列表。

        Args:
            outs_dict: 模型输出字典，包含 'maps' 键（概率图）
            shape_list: 每张图像的原始尺寸和缩放比例列表
                       每项为 [src_h, src_w, ratio_h, ratio_w]

        Returns:
            list: 每张图像的检测结果，格式为 [{'points': [[x1,y1], ...]}, ...]
        """
        pred = outs_dict['maps']
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.box_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            elif self.box_type == 'quad':
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
            else:
                raise ValueError(
                    "box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({'points': boxes})
        return boxes_batch


class BaseRecLabelDecode:
    """
    文本识别标签解码基类。

    负责将模型输出的字符索引序列转换为可读文本字符串。
    支持 CTC 解码中的去重和特殊字符处理。

    Attributes:
        character_str: 字符集列表
        dict: 字符到索引的映射
        character: 索引到字符的映射列表
    """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """
        将字符索引序列转换为文本字符串。

        处理逻辑：
        1. 去除重复的连续索引（CTC 解码需要）
        2. 过滤空白标记（索引 0）
        3. 对阿拉伯语文本进行逆序处理

        Args:
            text_index: 字符索引序列，shape (batch_size, seq_len)
            text_prob: 可选的概率序列，用于计算置信度
            is_remove_duplicate: 是否去除连续重复索引

        Returns:
            list: [(text, confidence), ...] 文本和平均置信度
        """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """
    CTC（Connectionist Temporal Classification）标签解码器。

    将 CTC 模型的输出（概率分布序列）解码为文本标签。
    通过 argmax 取每帧最可能的字符，然后去重和去除空白标记。

    继承自 BaseRecLabelDecode，添加了 CTC 特有的空白标记处理。
    """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
