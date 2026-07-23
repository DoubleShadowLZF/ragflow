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
可视化工具模块

提供检测结果的可视化绘制功能，用于在图像上绘制：
- 检测框（bbox）
- 类别标签
- 置信度分数

主要用于调试和结果展示。
"""

import logging
import os
import PIL
from PIL import ImageDraw


def save_results(image_list, results, labels, output_dir='output/', threshold=0.5):
    """
    将检测结果绘制到图像上并保存到指定目录。

    Args:
        image_list: PIL Image 列表，原始输入图像
        results: 检测结果列表，每个元素为一页的检测框列表
        labels: 类别标签列表
        output_dir: 输出目录路径
        threshold: 置信度阈值，低于此值的检测框不绘制
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, im in enumerate(image_list):
        im = draw_box(im, results[idx], labels, threshold=threshold)

        out_path = os.path.join(output_dir, f"{idx}.jpg")
        im.save(out_path, quality=95)
        logging.debug("save result to: " + out_path)


def draw_box(im, result, labels, threshold=0.5):
    """
    在图像上绘制检测框和标签。

    绘制内容包括：
    - 彩色矩形边框（颜色由类别决定）
    - 左上角的类别名称和置信度标签

    Args:
        im: PIL Image 对象
        result: 检测结果列表，每项包含 type, bbox, score
        labels: 类别标签列表
        threshold: 置信度阈值

    Returns:
        PIL Image: 绘制后的图像
    """
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    color_list = get_color_map_list(len(labels))
    clsid2color = {n.lower():color_list[i] for i,n in enumerate(labels)}
    result = [r for r in result if r["score"] >= threshold]

    for dt in result:
        color = tuple(clsid2color[dt["type"]])
        xmin, ymin, xmax, ymax = dt["bbox"]
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # 绘制标签文字和背景
        text = "{} {:.4f}".format(dt["type"], dt["score"])
        tw, th = imagedraw_textsize_c(draw, text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im


def get_color_map_list(num_classes):
    """
    为每个类别生成唯一的 RGB 颜色。

    使用位操作将类别索引映射到不同的颜色通道组合，
    确保相邻类别的颜色有较明显的区分度。

    Args:
        num_classes (int): 类别数量

    Returns:
        color_map (list): RGB 颜色列表，每个元素为 [R, G, B]
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def imagedraw_textsize_c(draw, text):
    """
    计算文本在图像中的绘制尺寸。

    兼容 Pillow 新旧版本 API：
    - Pillow < 10: 使用 draw.textsize()
    - Pillow >= 10: 使用 draw.textbbox()

    Args:
        draw: ImageDraw 对象
        text: 要测量的文本字符串

    Returns:
        tuple: (宽度, 高度)
    """
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tw, th = right - left, bottom - top

    return tw, th
