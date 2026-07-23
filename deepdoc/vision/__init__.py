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
deepdoc.vision 模块

提供文档视觉处理的核心功能，包括：
- OCR 文字检测与识别
- 版面布局分析（文本、标题、表格、图片等区域识别）
- 表格结构识别（行列、表头、合并单元格）
- 图像预处理与后处理算子

主要导出类：
    OCR: 光学字符识别引擎
    Recognizer: 版面/结构识别的基类
    LayoutRecognizer: 版面布局识别器（基于 YOLOv10）
    AscendLayoutRecognizer: 华为昇腾平台的版面识别器
    TableStructureRecognizer: 表格结构识别器
"""
import io
import sys
import threading

import pdfplumber

from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import AscendLayoutRecognizer
from .layout_recognizer import LayoutRecognizer4YOLOv10 as LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer

# 全局锁，用于保护 pdfplumber 的并发访问
LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def init_in_out(args):
    """
    初始化输入图像和输出路径。

    支持 PDF 和图片格式输入：
    - PDF 文件会被转换为逐页图片（使用 pdfplumber）
    - 图片文件直接加载为 RGB PIL Image

    Args:
        args: 命令行参数对象，需包含：
            - inputs: 输入文件路径或目录
            - output_dir: 输出目录路径

    Returns:
        tuple: (images, outputs)
            - images: PIL Image 列表
            - outputs: 对应的输出文件路径列表
    """
    import os
    import traceback

    from PIL import Image

    from common.file_utils import traversal_files

    images = []
    outputs = []

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    def pdf_pages(fnm, zoomin=3):
        """将 PDF 文件转换为逐页图片列表"""
        nonlocal outputs, images
        with sys.modules[LOCK_KEY_pdfplumber]:
            pdf = pdfplumber.open(fnm)
            images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(pdf.pages)]

        for i, page in enumerate(images):
            outputs.append(os.path.split(fnm)[-1] + f"_{i}.jpg")
        pdf.close()

    def images_and_outputs(fnm):
        """处理单个输入文件，区分 PDF 和普通图片"""
        nonlocal outputs, images
        if fnm.split(".")[-1].lower() == "pdf":
            pdf_pages(fnm)
            return
        try:
            with open(fnm, "rb") as fp:
                binary = fp.read()
            images.append(Image.open(io.BytesIO(binary)).convert("RGB"))
            outputs.append(os.path.split(fnm)[-1])
        except Exception:
            traceback.print_exc()

    if os.path.isdir(args.inputs):
        for fnm in traversal_files(args.inputs):
            images_and_outputs(fnm)
    else:
        images_and_outputs(args.inputs)

    for i in range(len(outputs)):
        outputs[i] = os.path.join(args.output_dir, outputs[i])

    return images, outputs


__all__ = [
    "OCR",
    "Recognizer",
    "LayoutRecognizer",
    "AscendLayoutRecognizer",
    "TableStructureRecognizer",
    "init_in_out",
]
