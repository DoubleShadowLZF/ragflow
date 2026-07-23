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
识别器基类模块

提供所有视觉识别任务的基类 Recognizer，包括：
- ONNX 模型加载与推理
- 图像预处理和后处理
- 版面分析中的布局排序与重叠检测
- 布局清理与去重

子类包括 LayoutRecognizer（版面识别）和 TableStructureRecognizer（表格结构识别）。
"""

import gc
import logging
import os
import math
import numpy as np
import cv2
from functools import cmp_to_key


from common.file_utils import get_project_base_directory
from .operators import *  # noqa: F403
from .operators import preprocess
from . import operators
from .ocr import load_model

class Recognizer:
    """
    视觉识别器基类。

    提供 ONNX 模型推理的通用流程：
    1. preprocess: 图像预处理（缩放、归一化、通道转换）
    2. ort_sess.run: ONNX 模型推理
    3. postprocess: 后处理（NMS、阈值过滤、坐标映射）

    同时提供版面分析中常用的工具方法：
    - 重叠面积计算
    - 布局排序（按 Y/X/C/R 坐标）
    - 布局去重与清理
    - 文本框与布局区域的对齐匹配

    Attributes:
        label_list: 类别标签列表
        ort_sess: ONNX Runtime 推理会话
        run_options: ONNX Runtime 运行选项
        input_names: 模型输入节点名称列表
        output_names: 模型输出节点名称列表
        input_shape: 模型输入尺寸 (H, W)
    """
    def __init__(self, label_list, task_name, model_dir=None):
        """
        初始化识别器。

        加载 ONNX 模型并获取输入/输出节点信息。

        HuggingFace 模型下载说明：
        - Linux: export HF_ENDPOINT=https://hf-mirror.com
        - Windows: 祝你好运 ^_-

        Args:
            label_list: 类别标签列表，如 ["text", "title", "table", ...]
            task_name: 任务名称，对应模型文件名（不含 .onnx 后缀）
            model_dir: 模型文件目录，默认使用项目内置目录
        """
        if not model_dir:
            model_dir = os.path.join(
                        get_project_base_directory(),
                        "rag/res/deepdoc")
        self.ort_sess, self.run_options = load_model(model_dir, task_name)
        self.input_names = [node.name for node in self.ort_sess.get_inputs()]
        self.output_names = [node.name for node in self.ort_sess.get_outputs()]
        self.input_shape = self.ort_sess.get_inputs()[0].shape[2:4]
        self.label_list = label_list

    @staticmethod
    def sort_Y_firstly(arr, threshold):
        """
        按 Y 坐标优先排序布局元素。

        先按 top（Y 坐标）排序，若两个元素的 Y 坐标差小于 threshold，
        则按 x0（X 坐标）排序。用于确定文档中文本的阅读顺序。

        Args:
            arr: 布局元素列表，每个元素需包含 'top' 和 'x0' 键
            threshold: Y 坐标差阈值，小于此值视为同一行

        Returns:
            排序后的列表
        """
        def cmp(c1, c2):
            diff = c1["top"] - c2["top"]
            if abs(diff) < threshold:
                diff = c1["x0"] - c2["x0"]
            return diff
        arr = sorted(arr, key=cmp_to_key(cmp))
        return arr

    @staticmethod
    def sort_X_firstly(arr, threshold):
        """
        按 X 坐标优先排序布局元素。

        先按 x0（X 坐标）排序，若两个元素的 X 坐标差小于 threshold，
        则按 top（Y 坐标）排序。适用于表格列排序等场景。

        Args:
            arr: 布局元素列表，每个元素需包含 'x0' 和 'top' 键
            threshold: X 坐标差阈值

        Returns:
            排序后的列表
        """
        def cmp(c1, c2):
            diff = c1["x0"] - c2["x0"]
            if abs(diff) < threshold:
                diff = c1["top"] - c2["top"]
            return diff
        arr = sorted(arr, key=cmp_to_key(cmp))
        return arr

    @staticmethod
    def sort_C_firstly(arr, thr=0):
        """
        按列（Column）优先排序。

        先用 sort_X_firstly 按 X 排序，然后根据每个元素的 "C"（列号）
        属性和 top 坐标进行微调，确保同列元素被正确排列。

        Args:
            arr: 布局元素列表，需包含 "C" 键（列索引）
            thr: 排序阈值

        Returns:
            排序后的列表
        """
        # 先用 X 坐标排序，再微调列顺序
        arr = Recognizer.sort_X_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if "C" not in arr[j] or "C" not in arr[j + 1]:
                    continue
                if arr[j + 1]["C"] < arr[j]["C"] \
                        or (
                        arr[j + 1]["C"] == arr[j]["C"]
                        and arr[j + 1]["top"] < arr[j]["top"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    @staticmethod
    def sort_R_firstly(arr, thr=0):
        """
        按行（Row）优先排序。

        先用 sort_Y_firstly 按 Y 排序，然后根据每个元素的 "R"（行号）
        属性和 x0 坐标进行微调，确保同行元素被正确排列。

        Args:
            arr: 布局元素列表，需包含 "R" 键（行索引）
            thr: 排序阈值

        Returns:
            排序后的列表
        """
        # 先用 Y 坐标排序，再微调行顺序
        arr = Recognizer.sort_Y_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                if "R" not in arr[j] or "R" not in arr[j + 1]:
                    continue
                if arr[j + 1]["R"] < arr[j]["R"] \
                        or (
                        arr[j + 1]["R"] == arr[j]["R"]
                        and arr[j + 1]["x0"] < arr[j]["x0"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    @staticmethod
    def overlapped_area(a, b, ratio=True):
        """
        计算两个矩形框的重叠面积。

        Args:
            a: 矩形框 A，包含 top, bottom, x0, x1
            b: 矩形框 B，包含 top, bottom, x0, x1
            ratio: 若为 True，返回重叠面积占 A 面积的比例；否则返回绝对面积

        Returns:
            float: 重叠面积或重叠比例
        """
        tp, btm, x0, x1 = a["top"], a["bottom"], a["x0"], a["x1"]
        if b["x0"] > x1 or b["x1"] < x0:
            return 0
        if b["bottom"] < tp or b["top"] > btm:
            return 0
        x0_ = max(b["x0"], x0)
        x1_ = min(b["x1"], x1)
        assert x0_ <= x1_, "Bbox mismatch! T:{},B:{},X0:{},X1:{} ==> {}".format(
            tp, btm, x0, x1, b)
        tp_ = max(b["top"], tp)
        btm_ = min(b["bottom"], btm)
        assert tp_ <= btm_, "Bbox mismatch! T:{},B:{},X0:{},X1:{} => {}".format(
            tp, btm, x0, x1, b)
        ov = (btm_ - tp_) * (x1_ - x0_) if x1 - \
                                           x0 != 0 and btm - tp != 0 else 0
        if ov > 0 and ratio:
            ov /= (x1 - x0) * (btm - tp)
        return ov

    @staticmethod
    def layouts_cleanup(boxes, layouts, far=2, thr=0.7):
        """
        清理重复或高度重叠的布局区域。

        对于相邻且高度重叠的同类型布局区域，根据以下规则去重：
        1. 比较置信度分数，保留分数高的
        2. 比较区域内包含的文字面积，保留文字面积大的

        Args:
            boxes: OCR 文字块列表
            layouts: 布局区域列表
            far: 检查的邻居范围（相邻多少个布局）
            thr: 重叠比例阈值

        Returns:
            去重后的布局区域列表
        """
        def not_overlapped(a, b):
            return any([a["x1"] < b["x0"],
                        a["x0"] > b["x1"],
                        a["bottom"] < b["top"],
                        a["top"] > b["bottom"]])

        i = 0
        while i + 1 < len(layouts):
            j = i + 1
            while j < min(i + far, len(layouts)) \
                    and (layouts[i].get("type", "") != layouts[j].get("type", "")
                         or not_overlapped(layouts[i], layouts[j])):
                j += 1
            if j >= min(i + far, len(layouts)):
                i += 1
                continue
            if Recognizer.overlapped_area(layouts[i], layouts[j]) < thr \
                    and Recognizer.overlapped_area(layouts[j], layouts[i]) < thr:
                i += 1
                continue

            if layouts[i].get("score") and layouts[j].get("score"):
                if layouts[i]["score"] > layouts[j]["score"]:
                    layouts.pop(j)
                else:
                    layouts.pop(i)
                continue

            area_i, area_i_1 = 0, 0
            for b in boxes:
                if not not_overlapped(b, layouts[i]):
                    area_i += Recognizer.overlapped_area(b, layouts[i], False)
                if not not_overlapped(b, layouts[j]):
                    area_i_1 += Recognizer.overlapped_area(b, layouts[j], False)

            if area_i > area_i_1:
                layouts.pop(j)
            else:
                layouts.pop(i)

        return layouts

    def create_inputs(self, imgs, im_info):
        """
        为模型生成批量输入数据。

        将多张不同尺寸的图像填充到统一尺寸，构建 batch 输入。

        Args:
            imgs (list): numpy 图像列表
            im_info (list): 每张图像的信息列表

        Returns:
            dict: 模型输入字典，包含 'image', 'im_shape', 'scale_factor'
        """
        inputs = {}

        im_shape = []
        scale_factor = []
        if len(imgs) == 1:
            inputs['image'] = np.array((imgs[0],)).astype('float32')
            inputs['im_shape'] = np.array(
                (im_info[0]['im_shape'],)).astype('float32')
            inputs['scale_factor'] = np.array(
                (im_info[0]['scale_factor'],)).astype('float32')
            return inputs
        
        im_shape = np.array([info['im_shape'] for info in im_info], dtype='float32')
        scale_factor = np.array([info['scale_factor'] for info in im_info], dtype='float32')

        inputs['im_shape'] = np.concatenate(im_shape, axis=0)
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = img
            padding_imgs.append(padding_im)
        inputs['image'] = np.stack(padding_imgs, axis=0)
        return inputs

    @staticmethod
    def find_overlapped(box, boxes_sorted_by_y, naive=False):
        """
        在按 Y 坐标排序的框列表中，通过二分查找定位与给定框重叠最大的框。

        使用二分查找优化搜索范围，只在 Y 坐标可能重叠的区间内遍历。

        Args:
            box: 目标框，包含 top, bottom
            boxes_sorted_by_y: 按 Y 排序的框列表
            naive: 是否使用朴素遍历（不优化）

        Returns:
            int 或 None: 重叠面积最大的框索引
        """
        if not boxes_sorted_by_y:
            return
        bxs = boxes_sorted_by_y
        s, e, ii = 0, len(bxs), 0
        while s < e and not naive:
            ii = (e + s) // 2
            pv = bxs[ii]
            if box["bottom"] < pv["top"]:
                e = ii
                continue
            if box["top"] > pv["bottom"]:
                s = ii + 1
                continue
            break
        while s < ii:
            if box["top"] > bxs[s]["bottom"]:
                s += 1
            break
        while e - 1 > ii:
            if box["bottom"] < bxs[e - 1]["top"]:
                e -= 1
            break

        max_overlapped_i, max_overlapped = None, 0
        for i in range(s, e):
            ov = Recognizer.overlapped_area(bxs[i], box)
            if ov <= max_overlapped:
                continue
            max_overlapped_i = i
            max_overlapped = ov

        return max_overlapped_i

    @staticmethod
    def find_horizontally_tightest_fit(box, boxes):
        """
        找到水平方向上最紧密对齐的框。

        在同一列（相同 layoutno）的框中，找到 X 坐标距离最小的框。
        用于将 OCR 文字块与表格列对齐。

        Args:
            box: 目标框
            boxes: 候选框列表

        Returns:
            int 或 None: 最紧密匹配的框索引
        """
        if not boxes:
            return
        min_dis, min_i = 1000000, None
        for i,b in enumerate(boxes):
            if box.get("layoutno", "0") != b.get("layoutno", "0"):
                continue
            dis = min(abs(box["x0"] - b["x0"]), abs(box["x1"] - b["x1"]), abs(box["x0"]+box["x1"] - b["x1"] - b["x0"])/2)
            if dis < min_dis:
                min_i = i
                min_dis = dis
        return min_i

    @staticmethod
    def find_overlapped_with_threshold(box, boxes, thr=0.3):
        """
        在框列表中查找与给定框重叠面积超过阈值的框。

        同时考虑双向重叠比例（box 覆盖候选框 和 候选框覆盖 box），
        取两者都超过阈值且综合重叠最大的框。

        Args:
            box: 目标框
            boxes: 候选框列表
            thr: 重叠比例阈值，默认 0.3

        Returns:
            int 或 None: 匹配的框索引
        """
        if not boxes:
            return
        max_overlapped_i, max_overlapped, _max_overlapped = None, thr, 0
        s, e = 0, len(boxes)
        for i in range(s, e):
            ov = Recognizer.overlapped_area(box, boxes[i])
            _ov = Recognizer.overlapped_area(boxes[i], box)
            if (ov, _ov) < (max_overlapped, _max_overlapped):
                continue
            max_overlapped_i = i
            max_overlapped = ov
            _max_overlapped = _ov

        return max_overlapped_i

    def preprocess(self, image_list):
        """
        图像预处理。

        根据模型输入格式的不同，支持两种预处理路径：
        1. 若模型需要 scale_factor（如 PaddleDetection 导出模型）：
           使用 LinearResize → StandardizeImage → Permute → PadStride 流程
        2. 否则：直接缩放到模型输入尺寸，归一化到 [0, 1]，转换为 CHW 格式

        Args:
            image_list: PIL Image 或 numpy 图像列表

        Returns:
            list: 每张图像的模型输入字典
        """
        inputs = []
        if "scale_factor" in self.input_names:
            preprocess_ops = []
            for op_info in [
                {'interp': 2, 'keep_ratio': False, 'target_size': [800, 608], 'type': 'LinearResize'},
                {'is_scale': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'type': 'StandardizeImage'},
                {'type': 'Permute'},
                {'stride': 32, 'type': 'PadStride'}
            ]:
                new_op_info = op_info.copy()
                op_type = new_op_info.pop('type')
                preprocess_ops.append(getattr(operators, op_type)(**new_op_info))

            for im_path in image_list:
                im, im_info = preprocess(im_path, preprocess_ops)
                inputs.append({"image": np.array((im,)).astype('float32'),
                               "scale_factor": np.array((im_info["scale_factor"],)).astype('float32')})
        else:
            hh, ww = self.input_shape
            for img in image_list:
                h, w = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(np.array(img).astype('float32'), (ww, hh))
                # Scale input pixel values to 0 to 1
                img /= 255.0
                img = img.transpose(2, 0, 1)
                img = img[np.newaxis, :, :, :].astype(np.float32)
                inputs.append({self.input_names[0]: img, "scale_factor": [w/ww, h/hh]})
        return inputs

    def postprocess(self, boxes, inputs, thr):
        """
        模型输出后处理。

        将模型原始输出转换为标准检测结果格式。
        支持两种输出格式：
        1. PaddleDetection 风格：[class_id, score, x1, y1, x2, y2]
        2. YOLO 风格：[x, y, w, h, obj_conf, class_probs...]

        执行步骤：
        - 阈值过滤低分框
        - 坐标映射回原始图像尺寸（乘以 scale_factor）
        - 类别级别的 NMS 去重

        Args:
            boxes: 模型原始输出
            inputs: 预处理时记录的输入信息（含 scale_factor）
            thr: 置信度阈值

        Returns:
            list: [{"type": 类别名, "bbox": [x1,y1,x2,y2], "score": 分数}, ...]
        """
        if "scale_factor" in self.input_names:
            bb = []
            for b in boxes:
                clsid, bbox, score = int(b[0]), b[2:], b[1]
                if score < thr:
                    continue
                if clsid >= len(self.label_list):
                    continue
                bb.append({
                    "type": self.label_list[clsid].lower(),
                    "bbox": [float(t) for t in bbox.tolist()],
                    "score": float(score)
                })
            return bb

        def xywh2xyxy(x):
            # [x, y, w, h] to [x1, y1, x2, y2]
            y = np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        def compute_iou(box, boxes):
            # Compute xmin, ymin, xmax, ymax for both boxes
            xmin = np.maximum(box[0], boxes[:, 0])
            ymin = np.maximum(box[1], boxes[:, 1])
            xmax = np.minimum(box[2], boxes[:, 2])
            ymax = np.minimum(box[3], boxes[:, 3])

            # Compute intersection area
            intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

            # Compute union area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union_area = box_area + boxes_area - intersection_area

            # Compute IoU
            iou = intersection_area / union_area

            return iou

        def iou_filter(boxes, scores, iou_threshold):
            sorted_indices = np.argsort(scores)[::-1]

            keep_boxes = []
            while sorted_indices.size > 0:
                # Pick the last box
                box_id = sorted_indices[0]
                keep_boxes.append(box_id)

                # Compute IoU of the picked box with the rest
                ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

                # Remove boxes with IoU over the threshold
                keep_indices = np.where(ious < iou_threshold)[0]

                # print(keep_indices.shape, sorted_indices.shape)
                sorted_indices = sorted_indices[keep_indices + 1]

            return keep_boxes

        boxes = np.squeeze(boxes).T
        # Filter out object confidence scores below threshold
        scores = np.max(boxes[:, 4:], axis=1)
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0:
            return []

        # Get the class with the highest confidence
        class_ids = np.argmax(boxes[:, 4:], axis=1)
        boxes = boxes[:, :4]
        input_shape = np.array([inputs["scale_factor"][0], inputs["scale_factor"][1], inputs["scale_factor"][0], inputs["scale_factor"][1]])
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)
        boxes = xywh2xyxy(boxes)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = iou_filter(class_boxes, class_scores, 0.2)
            indices.extend(class_indices[class_keep_boxes])

        return [{
            "type": self.label_list[class_ids[i]].lower(),
            "bbox": [float(t) for t in boxes[i].tolist()],
            "score": float(scores[i])
        } for i in indices]

    def close(self):
        logging.info("Close recognizer.")
        if hasattr(self, "ort_sess"):
            del self.ort_sess
        gc.collect()

    def __call__(self, image_list, thr=0.7, batch_size=16):
        """
        执行识别推理。

        按 batch 处理图像列表，依次执行：
        preprocess → ONNX 推理 → postprocess

        Args:
            image_list: PIL Image 或 numpy 图像列表
            thr: 置信度阈值，低于此值的检测框将被过滤
            batch_size: 批量大小

        Returns:
            list: 每张图像的检测结果列表
        """
        res = []
        images = []
        for i in range(len(image_list)):
            if not isinstance(image_list[i], np.ndarray):
                images.append(np.array(image_list[i]))
            else:
                images.append(image_list[i])

        batch_loop_cnt = math.ceil(float(len(images)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(images))
            batch_image_list = images[start_index:end_index]
            inputs = self.preprocess(batch_image_list)
            logging.debug("preprocess")
            for ins in inputs:
                bb = self.postprocess(self.ort_sess.run(None, {k:v for k,v in ins.items() if k in self.input_names}, self.run_options)[0], ins, thr)
                res.append(bb)

        #seeit.save_results(image_list, res, self.label_list, threshold=thr)

        return res

    def __del__(self):
        self.close()


