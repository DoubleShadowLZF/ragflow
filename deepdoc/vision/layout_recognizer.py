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
版面布局识别模块

提供文档图像的版面布局分析功能，识别以下区域类型：
- 文本（Text）、标题（Title）
- 图片（Figure）、图片说明（Figure caption）
- 表格（Table）、表格说明（Table caption）
- 页眉（Header）、页脚（Footer）
- 参考文献（Reference）、公式（Equation）

支持三种识别器实现：
- LayoutRecognizer: 基类，使用 ONNX 模型 + 传统后处理
- LayoutRecognizer4YOLOv10: 基于 YOLOv10 的版面识别
- AscendLayoutRecognizer: 华为昇腾平台的版面识别
"""

import logging
import math
import os
import re
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from common.file_utils import get_project_base_directory
from deepdoc.vision import Recognizer
from deepdoc.vision.operators import nms


class LayoutRecognizer(Recognizer):
    """
    版面布局识别器。

    基于 ONNX 目标检测模型，识别文档中的版面元素（标题、文本、表格、图片等）。
    将模型检测结果与 OCR 文字块对齐，为每个文字块打上布局类型标签。

    支持通过 DEEPDOC_URL / TENSORRT_DLA_SVR 环境变量使用远程 DLA 推理服务。

    Attributes:
        labels: 版面类别标签列表
        garbage_layouts: 垃圾布局类型（页眉、页脚、参考文献），可选择丢弃
    """
    labels = [
        "_background_",
        "Text",
        "Title",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Header",
        "Footer",
        "Reference",
        "Equation",
    ]

    def __init__(self, domain):
        self.garbage_layouts = ["footer", "header", "reference"]
        self.client = None

        dla_url = os.environ.get("DEEPDOC_URL") or os.environ.get("TENSORRT_DLA_SVR")
        if dla_url:
            from deepdoc.vision.dla_cli import DLAClient

            self.client = DLAClient(dla_url)
            env_used = "DEEPDOC_URL" if os.environ.get("DEEPDOC_URL") else "TENSORRT_DLA_SVR"
            logging.info(f"LayoutRecognizer using remote DLA client at {dla_url} (via {env_used})")
            return

        try:
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            super().__init__(self.labels, domain, model_dir)
        except Exception:
            model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc", local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"), local_dir_use_symlinks=False)
            super().__init__(self.labels, domain, model_dir)

    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        """
        执行版面布局识别并与 OCR 结果融合。

        处理流程：
        1. 运行布局检测模型，获取版面区域
        2. 缩放区域坐标到原始图像尺寸（除以 scale_factor）
        3. 按 Y 坐标排序后清理重叠区域
        4. 将 OCR 文字块与检测到的版面区域进行匹配
        5. 为每个文字块打上 layout_type 标签
        6. 过滤页眉/页脚/参考文献中的重复垃圾文字

        Args:
            image_list: 原始图像列表（PIL Image）
            scale_factor: 缩放因子（PDF 渲染时的放大倍数）
            thr: 检测置信度阈值
            batch_size: 批处理大小
            drop: 是否丢弃页眉/页脚/参考文献区域的文字

        Returns:
            tuple: (ocr_res, page_layout)
                - ocr_res: 打上布局标签后的文字块列表
                - page_layout: 每页的布局区域信息
        """
        def __is_garbage(b):
            patt = [r"\(cid\s*:\s*\d+\s*\)"]
            return any([re.search(p, b.get("text", "")) for p in patt])

        if self.client:
            layouts = self.client.predict(image_list)
        else:
            layouts = super().__call__(image_list, thr, batch_size)
        # save_results(image_list, layouts, self.labels, output_dir='output/', threshold=0.7)
        assert len(image_list) == len(ocr_res)
        # Tag layout type
        boxes = []
        assert len(image_list) == len(layouts)
        garbages = {}
        page_layout = []
        for pn, lts in enumerate(layouts):
            bxs = ocr_res[pn]
            lts = [
                {
                    "type": b["type"],
                    "score": float(b["score"]),
                    "x0": b["bbox"][0] / scale_factor,
                    "x1": b["bbox"][2] / scale_factor,
                    "top": b["bbox"][1] / scale_factor,
                    "bottom": b["bbox"][-1] / scale_factor,
                    "page_number": pn,
                }
                for b in lts
                if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts
            ]
            lts = self.sort_Y_firstly(lts, np.mean([lt["bottom"] - lt["top"] for lt in lts]) / 2)
            lts = self.layouts_cleanup(bxs, lts)
            page_layout.append(lts)

            def findLayout(ty):
                nonlocal bxs, lts, self
                lts_ = [lt for lt in lts if lt["type"] == ty]
                i = 0
                while i < len(bxs):
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    if __is_garbage(bxs[i]):
                        bxs.pop(i)
                        continue

                    ii = self.find_overlapped_with_threshold(bxs[i], lts_, thr=0.4)
                    if ii is None:
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue
                    lts_[ii]["visited"] = True
                    keep_feats = [
                        lts_[ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9 / scale_factor,
                        lts_[ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].size[1] * 0.1 / scale_factor,
                    ]
                    if drop and lts_[ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        if lts_[ii]["type"] not in garbages:
                            garbages[lts_[ii]["type"]] = []
                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[ii]["type"] != "equation" else "figure"
                    i += 1

            for lt in ["footer", "header", "reference", "figure caption", "table caption", "title", "table", "text", "figure", "equation"]:
                findLayout(lt)

            # add box to figure layouts which has not text box
            for i, lt in enumerate([lt for lt in lts if lt["type"] in ["figure", "equation"]]):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                del lt["type"]
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                bxs.append(lt)

            boxes.extend(bxs)

        ocr_res = boxes

        garbag_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbag_set.add(g)

        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]
        return ocr_res, page_layout

    def forward(self, image_list, thr=0.7, batch_size=16):
        return super().__call__(image_list, thr, batch_size)


class LayoutRecognizer4YOLOv10(LayoutRecognizer):
    """
    基于 YOLOv10 的版面布局识别器。

    相比基类 LayoutRecognizer，使用 YOLOv10 模型架构，具有更高的检测精度和速度。
    预处理采用 letterbox 缩放策略（保持宽高比 + 填充），后处理按类别分别进行 NMS。

    Attributes:
        labels: YOLOv10 特定的类别标签映射
        auto: 是否使用自动锚框（默认 False）
        scaleFill: 是否使用缩放填充模式
        scaleup: 是否允许放大
        stride: 模型 stride（默认 32）
    """

    labels = [
        "title",
        "Text",
        "Reference",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Table caption",
        "Equation",
        "Figure caption",
    ]

    def __init__(self, domain):
        domain = "layout"
        super().__init__(domain)
        self.auto = False
        self.scaleFill = False
        self.scaleup = True
        self.stride = 32
        self.center = True

    def preprocess(self, image_list):
        inputs = []
        new_shape = self.input_shape  # height, width
        for img in image_list:
            shape = img.shape[:2]  # current shape [height, width]
            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            ww, hh = new_unpad
            img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.float32)
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
            img /= 255.0
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :, :, :].astype(np.float32)
            inputs.append({self.input_names[0]: img, "scale_factor": [shape[1] / ww, shape[0] / hh, dw, dh]})

        return inputs

    def postprocess(self, boxes, inputs, thr):
        thr = 0.08
        boxes = np.squeeze(boxes)
        scores = boxes[:, 4]
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0:
            return []
        class_ids = boxes[:, -1].astype(int)
        boxes = boxes[:, :4]
        boxes[:, 0] -= inputs["scale_factor"][2]
        boxes[:, 2] -= inputs["scale_factor"][2]
        boxes[:, 1] -= inputs["scale_factor"][3]
        boxes[:, 3] -= inputs["scale_factor"][3]
        input_shape = np.array([inputs["scale_factor"][0], inputs["scale_factor"][1], inputs["scale_factor"][0], inputs["scale_factor"][1]])
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = nms(class_boxes, class_scores, 0.45)
            indices.extend(class_indices[class_keep_boxes])

        return [{"type": self.label_list[class_ids[i]].lower(), "bbox": [float(t) for t in boxes[i].tolist()], "score": float(scores[i])} for i in indices]


class AscendLayoutRecognizer(Recognizer):
    """
    华为昇腾（Ascend）平台的版面布局识别器。

    使用昇腾推理引擎（ais_bench）加载 .om 格式模型进行推理。
    预处理和后处理逻辑与 LayoutRecognizer4YOLOv10 类似，
    但推理后端使用昇腾 NPU 而非 ONNX Runtime。

    通过环境变量 ASCEND_LAYOUT_RECOGNIZER_DEVICE_ID 指定昇腾设备 ID。

    Attributes:
        session: 昇腾推理会话
        input_shape: 模型输入尺寸 (H, W)
        garbage_layouts: 垃圾布局类型列表
    """

    labels = [
        "title",
        "Text",
        "Reference",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Table caption",
        "Equation",
        "Figure caption",
    ]

    def __init__(self, domain):
        from ais_bench.infer.interface import InferSession

        model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
        model_file_path = os.path.join(model_dir, domain + ".om")

        if not os.path.exists(model_file_path):
            raise ValueError(f"Model file not found: {model_file_path}")

        device_id = int(os.getenv("ASCEND_LAYOUT_RECOGNIZER_DEVICE_ID", 0))
        self.session = InferSession(device_id=device_id, model_path=model_file_path)
        self.input_shape = self.session.get_inputs()[0].shape[2:4]  # H,W
        self.garbage_layouts = ["footer", "header", "reference"]

    def preprocess(self, image_list):
        inputs = []
        H, W = self.input_shape
        for img in image_list:
            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            r = min(H / h, W / w)
            new_unpad = (int(round(w * r)), int(round(h * r)))
            dw, dh = (W - new_unpad[0]) / 2.0, (H - new_unpad[1]) / 2.0

            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            img /= 255.0
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            inputs.append(
                {
                    "image": img,
                    "scale_factor": [w / new_unpad[0], h / new_unpad[1]],
                    "pad": [dw, dh],
                    "orig_shape": [h, w],
                }
            )
        return inputs

    def postprocess(self, boxes, inputs, thr=0.25):
        arr = np.squeeze(boxes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results = []
        if arr.shape[1] == 6:
            # [x1,y1,x2,y2,score,cls]
            m = arr[:, 4] >= thr
            arr = arr[m]
            if arr.size == 0:
                return []
            xyxy = arr[:, :4].astype(np.float32)
            scores = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)

            if "pad" in inputs:
                dw, dh = inputs["pad"]
                sx, sy = inputs["scale_factor"]
                xyxy[:, [0, 2]] -= dw
                xyxy[:, [1, 3]] -= dh
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)
            else:
                # backup
                sx, sy = inputs["scale_factor"]
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)

            keep_indices = []
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                k = nms(xyxy[idx], scores[idx], 0.45)
                keep_indices.extend(idx[k])

            for i in keep_indices:
                cid = int(cls_ids[i])
                if 0 <= cid < len(self.labels):
                    results.append({"type": self.labels[cid].lower(), "bbox": [float(t) for t in xyxy[i].tolist()], "score": float(scores[i])})
            return results

        raise ValueError(f"Unexpected output shape: {arr.shape}")

    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        import re
        from collections import Counter

        assert len(image_list) == len(ocr_res)

        images = [np.array(im) if not isinstance(im, np.ndarray) else im for im in image_list]
        layouts_all_pages = []  # list of list[{"type","score","bbox":[x1,y1,x2,y2]}]

        conf_thr = max(thr, 0.08)

        batch_loop_cnt = math.ceil(float(len(images)) / batch_size)
        for bi in range(batch_loop_cnt):
            s = bi * batch_size
            e = min((bi + 1) * batch_size, len(images))
            batch_images = images[s:e]

            inputs_list = self.preprocess(batch_images)
            logging.debug("preprocess done")

            for ins in inputs_list:
                feeds = [ins["image"]]
                out_list = self.session.infer(feeds=feeds, mode="static")

                for out in out_list:
                    lts = self.postprocess(out, ins, conf_thr)

                    page_lts = []
                    for b in lts:
                        if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts:
                            x0, y0, x1, y1 = b["bbox"]
                            page_lts.append(
                                {
                                    "type": b["type"],
                                    "score": float(b["score"]),
                                    "x0": float(x0) / scale_factor,
                                    "x1": float(x1) / scale_factor,
                                    "top": float(y0) / scale_factor,
                                    "bottom": float(y1) / scale_factor,
                                    "page_number": len(layouts_all_pages),
                                }
                            )
                    layouts_all_pages.append(page_lts)

        def _is_garbage_text(box):
            patt = [r"^•+$", r"^[0-9]{1,2} / ?[0-9]{1,2}$", r"^[0-9]{1,2} of [0-9]{1,2}$", r"^http://[^ ]{12,}", r"\(cid *: *[0-9]+ *\)"]
            return any(re.search(p, box.get("text", "")) for p in patt)

        boxes_out = []
        page_layout = []
        garbages = {}

        for pn, lts in enumerate(layouts_all_pages):
            if lts:
                avg_h = np.mean([lt["bottom"] - lt["top"] for lt in lts])
                lts = self.sort_Y_firstly(lts, avg_h / 2 if avg_h > 0 else 0)

            bxs = ocr_res[pn]
            lts = self.layouts_cleanup(bxs, lts)
            page_layout.append(lts)

            def _tag_layout(ty):
                nonlocal bxs, lts
                lts_of_ty = [lt for lt in lts if lt["type"] == ty]
                i = 0
                while i < len(bxs):
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    if _is_garbage_text(bxs[i]):
                        bxs.pop(i)
                        continue

                    ii = self.find_overlapped_with_threshold(bxs[i], lts_of_ty, thr=0.4)
                    if ii is None:
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue

                    lts_of_ty[ii]["visited"] = True

                    keep_feats = [
                        lts_of_ty[ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].shape[0] * 0.9 / scale_factor,
                        lts_of_ty[ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].shape[0] * 0.1 / scale_factor,
                    ]
                    if drop and lts_of_ty[ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        garbages.setdefault(lts_of_ty[ii]["type"], []).append(bxs[i].get("text", ""))
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_of_ty[ii]["type"] if lts_of_ty[ii]["type"] != "equation" else "figure"
                    i += 1

            for ty in ["footer", "header", "reference", "figure caption", "table caption", "title", "table", "text", "figure", "equation"]:
                _tag_layout(ty)

            figs = [lt for lt in lts if lt["type"] in ["figure", "equation"]]
            for i, lt in enumerate(figs):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                lt.pop("type", None)
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                bxs.append(lt)

            boxes_out.extend(bxs)

        garbag_set = set()
        for k, lst in garbages.items():
            cnt = Counter(lst)
            for g, c in cnt.items():
                if c > 1:
                    garbag_set.add(g)

        ocr_res_new = [b for b in boxes_out if b["text"].strip() not in garbag_set]
        return ocr_res_new, page_layout
