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
RAGFlow PDF 解析器核心模块。

本模块包含 RAGFlow 的 PDF 文档解析核心实现，是整个文档解析系统的基石。
提供了从底层 PDF 文本提取到高层版面分析的完整流水线。

核心类：
- RAGFlowPdfParser: 主解析器，实现完整的 PDF 解析流水线
  - 页面图片渲染（pdfplumber）
  - OCR 文本识别（PaddleOCR / ONNX）
  - 版面分析（LayoutRecognizer，ONNX 模型）
  - 表格结构识别（TableStructureRecognizer）
  - 文本合并与排序（XGBoost 模型判断上下行连接）
  - 阅读顺序重建（K-Means 分栏）
  - 乱码检测（PUA 字符 + 字体编码检测）
  - 表格方向自动纠正
  - 图片/表格裁剪与提取
- PlainParser: 轻量解析器，仅使用 pypdf 提取纯文本
- VisionParser: 视觉解析器，使用 VLM 进行页面描述

解析流水线（RAGFlowPdfParser.__call__）:
1. __images__: 渲染页面图片，提取字符级 PDF 文本
2. _layouts_rec: 版面分析，识别文本/表格/图片区域
3. _table_transformer_job: 表格结构识别
4. _text_merge: 水平合并相邻文本框
5. _concat_downward: 垂直合并（上下行连接）
6. _filter_forpages: 过滤目录页和无意义内容
7. _extract_table_figure: 提取表格和图片
8. __filterout_scraps: 过滤碎片文本，生成最终输出
"""

import asyncio
import logging
import math
import os
import random
import re
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer

import numpy as np
import pdfplumber
import xgboost as xgb
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader as pdf2_read
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common.constants import MAXIMUM_PAGE_NUMBER
from common.file_utils import get_project_base_directory
from deepdoc.vision import OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer
from rag.nlp import rag_tokenizer
from rag.prompts.generator import vision_llm_describe_prompt
from deepdoc.parser.utils import extract_pdf_outlines
from common import settings

from common.misc_utils import thread_pool_exec

# pdfplumber 全局锁，用于多线程环境下保护 pdfplumber 的 PDF 打开操作
LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class RAGFlowPdfParser:
    """
    RAGFlow PDF 解析器核心类。

    实现完整的 PDF 文档理解流水线，包括：
    - OCR 文本检测与识别
    - 深度学习版面分析（ONNX / Ascend 推理引擎）
    - 表格结构识别
    - 文本阅读顺序重建
    - 跨页段落连接（XGBoost 模型）
    - 目录检测与过滤
    - 表格/图片提取

    解析结果输出为包含位置标签的文本行和裁剪图片的列表。
    """

    def __init__(self, **kwargs):
        """
        初始化 PDF 解析器。

        加载 OCR 引擎、版面分析模型、表格识别模型和文本连接判断模型。

        HuggingFace 模型下载说明：
        - Linux: export HF_ENDPOINT=https://hf-mirror.com
        - Windows: 祝你好运 ^_-

        Keyword Args:
            model_species: 模型种类标识，影响版面分析模型的域名选择
        """
        # 初始化 OCR 引擎（PaddleOCR 或 ONNX）
        self.ocr = OCR()

        # 多 GPU 并行支持：为每个设备创建独立的信号量
        self.parallel_limiter = None
        if settings.PARALLEL_DEVICES > 1:
            self.parallel_limiter = [asyncio.Semaphore(1) for _ in range(settings.PARALLEL_DEVICES)]

        # 版面识别器类型选择：onnx（默认）或 ascend（华为昇腾）
        layout_recognizer_type = os.getenv("LAYOUT_RECOGNIZER_TYPE", "onnx").lower()
        if layout_recognizer_type not in ["onnx", "ascend"]:
            raise RuntimeError("Unsupported layout recognizer type.")

        if hasattr(self, "model_species"):
            recognizer_domain = "layout." + self.model_species
        else:
            recognizer_domain = "layout"

        if layout_recognizer_type == "ascend":
            logging.debug("Using Ascend LayoutRecognizer")
            self.layouter = AscendLayoutRecognizer(recognizer_domain)
        else:  # onnx
            logging.debug("Using Onnx LayoutRecognizer")
            self.layouter = LayoutRecognizer(recognizer_domain)

        # 初始化表格结构识别器
        self.tbl_det = TableStructureRecognizer()

        # 初始化 XGBoost 模型，用于判断上下两个文本行是否应连接
        self.updown_cnt_mdl = xgb.Booster()
        # xgboost 模型很小，显式使用 CPU 推理
        self.updown_cnt_mdl.set_param({"device": "cpu"})
        logging.info("updown_cnt_mdl initialized on CPU")
        try:
            # 优先从本地加载预下载的模型
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
        except Exception:
            # 本地加载失败时从 HuggingFace 下载
            model_dir = snapshot_download(repo_id="InfiniFlow/text_concat_xgb_v1.0", local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"), local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))

        self.page_from = 0
        self.column_num = 1

    # ==================== 几何与坐标计算工具方法 ====================

    def __char_width(self, c):
        """计算单个字符的平均宽度：(x1 - x0) / 字符数"""
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        """获取元素的高度"""
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        """
        计算两个元素之间的水平距离。

        取多种定义的最小值：右边界距离、左边界距离、中心点距离的一半。
        """
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(self, a, b):
        """
        计算两个元素之间的垂直距离。

        使用中心点的垂直偏差：(b.top + b.bottom - a.top - a.bottom) / 2
        """
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    # ==================== 文本模式匹配方法 ====================

    def _match_proj(self, b):
        """
        检查文本是否匹配项目符号/编号模式。

        支持中文序号（第X章、第X条）、数字编号（1. 2.）、
        括号编号（(1) (2)）、项目符号（⚫•➢）等。

        Args:
            b: 包含 "text" 字段的文本框字典

        Returns:
            bool: 是否匹配项目/编号模式
        """
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        """
        提取上下两个文本框的连接特征向量。

        生成 32 维特征向量，供 XGBoost 模型判断两个文本块是否应合并。
        特征包括：几何关系（宽高比、间距）、文本内容（标点、大小写）、
        版面类型、分页信息、分词结果等。

        Args:
            up: 上方文本框
            down: 下方文本框

        Returns:
            list: 32 维特征向量
        """
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = up["text"][-LEN:].strip() + (" " if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0]) else "") + down["text"][:LEN].strip()
        tks_all = rag_tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),           # 是否在同一表格行
            y_dis / h,                                        # 垂直间距归一化
            down["page_number"] - up["page_number"],          # 跨页数
            up["layout_type"] == down["layout_type"],         # 版面类型是否相同
            up["layout_type"] == "text",                      # 上方是否为文本
            down["layout_type"] == "text",                    # 下方是否为文本
            up["layout_type"] == "table",                     # 上方是否为表格
            down["layout_type"] == "table",                   # 下方是否为表格
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,  # 上方是否以句末标点结尾
            True if re.search(r"[，：'""、0-9（+-]$", up["text"]) else False,           # 上方是否以句中标点结尾
            True if re.search(r"(^.?[/,?;:\]，。；：'""》！？】）-])", down["text"]) else False,  # 下方是否以标点开头
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,       # 上方是否为括号包裹
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,                 # 上方是否以逗号结尾（未完成）
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,                 # 同上（重复特征，原设计保留）
            True if re.search(r"[\(（][^\)）]+$", up["text"]) and re.search(r"[\)）]", down["text"]) else False,  # 跨行括号匹配
            self._match_proj(down),                            # 下方是否为项目符号
            True if re.match(r"[A-Z]", down["text"]) else False,   # 下方是否以大写字母开头
            True if re.match(r"[A-Z]", up["text"][-1]) else False,  # 上方是否以大写字母结尾
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,  # 上方是否以小写字母/数字结尾
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,  # 下方是否全为数字/符号
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],                             # 上方是否完全在下方右侧
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up), self.__height(down)),  # 高度差异比
            self._x_dis(up, down) / max(w, 0.000001),          # 水平距离归一化
            (len(up["text"]) - len(down["text"])) / max(len(up["text"]), len(down["text"])),  # 文本长度差异比
            len(tks_all) - len(tks_up) - len(tks_down),        # 拼接后的新词数
            len(tks_down) - len(tks_up),                       # 下方比上方多的词数
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,  # 尾词是否相同
            max(down["in_row"], up["in_row"]),                 # 同行最大元素数
            abs(down["in_row"] - up["in_row"]),                # 同行元素数差异
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,  # 下方单名词
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0,      # 上方单名词
        ]
        return fea

    # ==================== 文本框排序方法 ====================

    @staticmethod
    def sort_X_by_page(arr, threshold):
        """
        按页面排序文本框：先按 x0，再按 top，并在阈值内修正顺序。

        先按 (page_number, x0, top) 排序，然后逆向冒泡修正：
        如果两个相邻元素 x0 差距小于阈值但 top 顺序反了，则交换。

        Args:
            arr: 文本框列表
            threshold: x0 差异阈值

        Returns:
            list: 排序后的文本框列表
        """
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold and arr[j + 1]["top"] < arr[j]["top"] and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        """
        检查 PDF 字符是否有可见颜色。

        忽略 DeviceGray 颜色空间中白色（值为1）的非内容字符
        （如内部标记、Tj 操作符字面量等）。

        Args:
            o: pdfplumber 字符对象

        Returns:
            bool: 有颜色返回 True（保留），纯白色无意义字符返回 False（丢弃）
        """
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    # ==================== 乱码检测方法 ====================

    # pdfminer 无法映射的 CID 字体字符的正则模式
    _CID_PATTERN = re.compile(r"\(cid\s*:\s*\d+\s*\)")

    @staticmethod
    def _is_garbled_char(ch):
        """
        检查单个字符是否为乱码（无法从 PDF 字体编码映射到有效 Unicode）。

        乱码判断条件：
        - Unicode 私有使用区 (PUA): U+E000-U+F8FF, U+F0000-U+FFFFF, U+100000-U+10FFFF
        - 替换字符: U+FFFD
        - C0/C1 控制字符（除 tab、换行、回车外）
        - Unicode 类别为 Cn（未分配）或 Cs（代理对）

        Args:
            ch: 单个字符

        Returns:
            bool: 是乱码返回 True
        """
        if not ch:
            return False
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            return True
        if 0xF0000 <= cp <= 0xFFFFF:
            return True
        if 0x100000 <= cp <= 0x10FFFF:
            return True
        if cp == 0xFFFD:
            return True
        if cp < 0x20 and ch not in ('\t', '\n', '\r'):
            return True
        if 0x80 <= cp <= 0x9F:
            return True
        cat = unicodedata.category(ch)
        if cat in ("Cn", "Cs"):
            return True
        return False

    @staticmethod
    def _is_garbled_text(text, threshold=0.5):
        """
        检查文本字符串中乱码字符的比例是否超过阈值。

        同时检测 pdfminer 的 CID 占位符模式 '(cid:123)'。

        Args:
            text: 待检查的文本
            threshold: 乱码比例阈值，默认 0.5

        Returns:
            bool: 乱码比例超过阈值返回 True
        """
        if not text or not text.strip():
            return False
        if RAGFlowPdfParser._CID_PATTERN.search(text):
            return True
        garbled_count = 0
        total = 0
        for ch in text:
            if ch.isspace():
                continue
            total += 1
            if RAGFlowPdfParser._is_garbled_char(ch):
                garbled_count += 1
        if total == 0:
            return False
        return garbled_count / total >= threshold

    @staticmethod
    def _has_subset_font_prefix(fontname):
        """
        检查字体名是否有子集前缀（如 'DY1+ZLQDm1-1'）。

        PDF 子集字体使用 2-6 个大写字母/数字标签加 '+' 前缀。

        Args:
            fontname: 字体名称

        Returns:
            bool: 有子集前缀返回 True
        """
        if not fontname:
            return False
        return bool(re.match(r"^[A-Z0-9]{2,6}\+", fontname))

    @staticmethod
    def _is_garbled_by_font_encoding(page_chars, min_chars=20):
        """
        检测因字体编码映射损坏导致的乱码。

        某些 PDF（尤其是较旧的中文标准文档）内嵌自定义字体，
        将 CJK 字形映射到 ASCII 码点，提取出的文本显示为随机
        ASCII 标点符号而非实际的中日韩字符。

        检测策略：如果大量字符来自子集嵌入字体，且页面产生
        压倒性多数 ASCII（标点、数字、符号），几乎没有 CJK/
        韩文/假名字符，则该页面可能因字体编码损坏而出现乱码。

        Args:
            page_chars: 页面字符列表
            min_chars: 最小字符数阈值，默认 20

        Returns:
            bool: 检测到字体编码乱码返回 True
        """
        if not page_chars or len(page_chars) < min_chars:
            return False

        subset_font_count = 0
        total_non_space = 0
        ascii_punct_sym = 0
        cjk_like = 0

        for c in page_chars:
            text = c.get("text", "")
            fontname = c.get("fontname", "")
            if not text or text.isspace():
                continue
            total_non_space += 1

            if RAGFlowPdfParser._has_subset_font_prefix(fontname):
                subset_font_count += 1

            cp = ord(text[0])
            # CJK / 韩文 / 日文 字符范围
            if (0x2E80 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF
                    or 0x20000 <= cp <= 0x2FA1F
                    or 0xAC00 <= cp <= 0xD7AF
                    or 0x3040 <= cp <= 0x30FF):
                cjk_like += 1
            # ASCII 标点符号范围
            elif (0x21 <= cp <= 0x2F or 0x3A <= cp <= 0x40
                    or 0x5B <= cp <= 0x60 or 0x7B <= cp <= 0x7E):
                ascii_punct_sym += 1

        if total_non_space < min_chars:
            return False

        # 子集字体占比低于 30% 则跳过
        subset_ratio = subset_font_count / total_non_space
        if subset_ratio < 0.3:
            return False

        # CJK 比例极低且标点比例极高 → 字体编码乱码
        cjk_ratio = cjk_like / total_non_space
        punct_ratio = ascii_punct_sym / total_non_space
        if cjk_ratio < 0.05 and punct_ratio > 0.4:
            return True

        return False

    # ==================== 表格方向检测与结构识别 ====================

    def _evaluate_table_orientation(self, table_img, sample_ratio=0.3):
        """
        评估表格图片的最佳旋转方向。

        测试 4 个旋转角度（0°/90°/180°/270°），使用 OCR 置信度分数
        确定最佳朝向。选择规则：
        - 通常选综合得分最高的角度
        - 绝对阈值规则：非 0° 选项必须比 0° 高 0.2 以上且 0° 得分低于 0.8

        Args:
            table_img: 表格区域的 PIL Image 对象
            sample_ratio: 快速评估的采样比例（当前未使用，保留接口）

        Returns:
            tuple: (最佳角度, 旋转后的最佳图片, 各角度置信度分数的字典)
        """
        rotations = [
            (0, "original"),        # 原始方向
            (90, "rotate_90"),      # 顺时针 90°
            (180, "rotate_180"),    # 180°
            (270, "rotate_270"),    # 顺时针 270°（逆时针 90°）
        ]

        results = {}
        best_score = -1
        best_angle = 0
        best_img = table_img
        score_0 = None

        for angle, name in rotations:
            # 旋转图片（PIL 的 rotate 是逆时针，用负角度实现顺时针）
            if angle == 0:
                rotated_img = table_img
            else:
                rotated_img = table_img.rotate(-angle, expand=True)

            # 转为 numpy 数组供 OCR 使用
            img_array = np.array(rotated_img)

            # OCR 检测与识别
            try:
                ocr_results = self.ocr(img_array)

                if ocr_results:
                    # 计算平均置信度
                    scores = [conf for _, (_, conf) in ocr_results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    total_regions = len(scores)

                    # 综合分数：同时考虑平均置信度和识别区域数
                    # 更多区域 + 更高置信度 = 更好的朝向
                    combined_score = avg_score * (1 + 0.1 * min(total_regions, 50) / 50)
                else:
                    avg_score = 0
                    total_regions = 0
                    combined_score = 0

            except Exception as e:
                logging.warning(f"OCR failed for angle {angle}: {e}")
                avg_score = 0
                total_regions = 0
                combined_score = 0

            results[angle] = {"avg_confidence": avg_score, "total_regions": total_regions, "combined_score": combined_score}
            if angle == 0:
                score_0 = combined_score

            logging.debug(f"Table orientation {angle}°: avg_conf={avg_score:.4f}, regions={total_regions}, combined={combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_angle = angle
                best_img = rotated_img

        # 绝对阈值规则：仅当非 0° 比 0° 高出 0.2 以上且 0° 得分低于 0.8 时才接受旋转
        if best_angle != 0 and score_0 is not None:
            if not (best_score - score_0 > 0.2 and score_0 < 0.8):
                best_angle = 0
                best_img = table_img
                best_score = score_0

        results[best_angle] = results.get(best_angle, {"avg_confidence": 0, "total_regions": 0, "combined_score": 0})

        logging.info(f"Best table orientation: {best_angle}° (score={best_score:.4f})")

        return best_angle, best_img, results

    def _table_transformer_job(self, ZM, auto_rotate=True):
        """
        执行表格结构识别（Table Structure Recognition, TSR）。

        当 auto_rotate=True 时的完整流程：
        1. 评估表格方向并选择最佳旋转角度
        2. 使用旋转后的图片进行表格结构识别
        3. 对旋转后的图片重新 OCR
        4. 将新 OCR 结果与 TSR 单元格坐标匹配

        识别结果（表头 headers、行 rows、列 clmns、合并单元格 spans）
        的信息被注入到 self.boxes 中相关表格文本框的元数据中。

        Args:
            ZM: 缩放因子
            auto_rotate: 是否启用自动方向纠正
        """
        logging.debug("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10  # 表格裁剪边距
        self.tb_cpns = []
        self.table_rotations = {}  # 存储每个表格的旋转信息
        self.rotated_table_imgs = {}  # 存储旋转后的表格图片

        assert len(self.page_layout) == len(self.page_images)

        # 收集所有表格的版面信息
        table_layouts = []  # [(page, table_layout, left, top, right, bott), ...]

        table_index = 0
        for p, tbls in enumerate(self.page_layout):  # 遍历每页
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            for tb in tbls:  # 遍历每个表格
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top, p, table_index))

                # 记录表格版面信息
                table_layouts.append({"page": p, "table_index": table_index, "layout": tb, "coords": (left, top, right, bott)})

                # 裁剪表格图片
                table_img = self.page_images[p].crop((left, top, right, bott))

                if auto_rotate:
                    # 评估表格方向
                    logging.debug(f"Evaluating orientation for table {table_index} on page {p}")
                    best_angle, rotated_img, rotation_scores = self._evaluate_table_orientation(table_img)

                    # 存储旋转信息
                    self.table_rotations[table_index] = {
                        "page": p,
                        "original_pos": (left, top, right, bott),
                        "best_angle": best_angle,
                        "scores": rotation_scores,
                        "rotated_size": rotated_img.size,
                    }

                    # 存储旋转后的图片
                    self.rotated_table_imgs[table_index] = rotated_img
                    imgs.append(rotated_img)

                else:
                    imgs.append(table_img)
                    self.table_rotations[table_index] = {"page": p, "original_pos": (left, top, right, bott), "best_angle": 0, "scores": {}, "rotated_size": table_img.size}
                    self.rotated_table_imgs[table_index] = table_img

                table_index += 1

        assert len(self.page_images) == len(tbcnt) - 1
        if not imgs:
            return

        # 执行表格结构识别（TSR）
        recos = self.tbl_det(imgs)

        # 如果表格曾被旋转，重新 OCR 旋转后的图片并替换表格框
        if auto_rotate:
            self._ocr_rotated_tables(ZM, table_layouts, recos, tbcnt)

        # 处理 TSR 结果：将识别出的表头/行/列/合并单元格信息注入 boxes
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # 遍历每页
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # 遍历每个表格
                poss = pos[tbcnt[i] : tbcnt[i + 1]]
                for it in tb_items:  # 遍历表格组件
                    # TSR 坐标相对于旋转后的图片，需要记录
                    it["x0_rotated"] = it["x0"]
                    it["x1_rotated"] = it["x1"]
                    it["top_rotated"] = it["top"]
                    it["bottom_rotated"] = it["bottom"]

                    it["pn"] = poss[j][2]  # 页码
                    it["layoutno"] = j
                    it["table_index"] = poss[j][3]  # 表格索引
                    pg.append(it)
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            """按标签模式筛选并排序 TSR 元素。"""
            eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # 收集并添加 R（行）、H（表头）、C（列）、SP（合并单元格）标签到 boxes
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0_rotated"] if "x0_rotated" in x else x["x0"]))
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)

        # 将表格结构信息注入到 boxes 中的表格文本框
        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def _ocr_rotated_tables(self, ZM, table_layouts, tsr_results, tbcnt):
        """
        对旋转后的表格图片重新 OCR 并更新 self.boxes。

        这是一个关键方法：当表格方向被纠正后，原始 pdfplumber 提取的
        文本坐标可能不准，需要重新 OCR 获取更准确的文本位置。

        工作流程：
        1. 遍历每个表格，定位其在 self.boxes 中的原始文本框
        2. 移除旧文本框
        3. 对旋转后的图片执行 OCR
        4. 将 OCR 结果坐标从旋转图片空间映射回原始表格坐标
        5. 将新文本框插入 self.boxes

        Args:
            ZM: 缩放因子
            table_layouts: 表格版面信息列表
            tsr_results: TSR 识别结果
            tbcnt: 每页表格累计数
        """
        tbcnt = np.cumsum(tbcnt)

        def _table_region(layout, page_index):
            """获取表格在原页面中的坐标和累积高度坐标。"""
            table_x0 = layout["x0"]
            table_top = layout["top"]
            table_x1 = layout["x1"]
            table_bottom = layout["bottom"]
            table_top_cum = table_top + self.page_cum_height[page_index]
            table_bottom_cum = table_bottom + self.page_cum_height[page_index]
            return table_x0, table_top, table_x1, table_bottom, table_top_cum, table_bottom_cum

        def _collect_table_boxes(page_index, table_x0, table_x1, table_top_cum, table_bottom_cum):
            """收集并移除表格区域内的旧文本框，返回旧框列表和插入位置。"""
            indices = [
                i
                for i, b in enumerate(self.boxes)
                if (
                    b.get("page_number") == page_index + self.page_from
                    and b.get("layout_type") == "table"
                    and b["x0"] >= table_x0 - 5
                    and b["x1"] <= table_x1 + 5
                    and b["top"] >= table_top_cum - 5
                    and b["bottom"] <= table_bottom_cum + 5
                )
            ]
            original_boxes = [self.boxes[i] for i in indices]
            insert_at = indices[0] if indices else len(self.boxes)
            for i in reversed(indices):
                self.boxes.pop(i)
            return original_boxes, insert_at

        def _restore_boxes(original_boxes, insert_at):
            """恢复旧文本框（OCR 失败时的回退操作）。"""
            for b in original_boxes:
                self.boxes.insert(insert_at, b)
                insert_at += 1
            return insert_at

        def _map_rotated_point(x, y, angle, width, height):
            """
            将旋转图片中的坐标映射回原始图片坐标。

            逆变换公式：
            - 0°:   恒等映射
            - 90°:  顺时针90° → (width - y, x)
            - 180°: 顺时针180° → (width - x, height - y)
            - 270°: 顺时针270° → (y, height - x)
            """
            if angle == 0:
                return x, y
            if angle == 90:
                return width - y, x
            if angle == 180:
                return width - x, height - y
            if angle == 270:
                return y, height - x
            return x, y

        def _insert_ocr_boxes(ocr_results, page_index, table_x0, table_top, insert_at, table_index, best_angle, table_w_px, table_h_px):
            """
            将 OCR 结果插入 self.boxes。

            OCR 坐标是相对于旋转后图片的，需要映射回原始表格坐标。
            同时向页面空间转换（加上表格偏移和页面累积高度）。
            """
            added = 0
            for bbox, (text, conf) in ocr_results:
                if conf < 0.5:  # 低置信度结果过滤
                    continue
                # 坐标映射：旋转图片空间 → 原始表格空间
                mapped = [_map_rotated_point(p[0], p[1], best_angle, table_w_px, table_h_px) for p in bbox]
                x_coords = [p[0] for p in mapped]
                y_coords = [p[1] for p in mapped]
                box_x0 = min(x_coords) / ZM
                box_x1 = max(x_coords) / ZM
                box_top = min(y_coords) / ZM
                box_bottom = max(y_coords) / ZM
                new_box = {
                    "text": text,
                    "x0": box_x0 + table_x0,
                    "x1": box_x1 + table_x0,
                    "top": box_top + table_top + self.page_cum_height[page_index],
                    "bottom": box_bottom + table_top + self.page_cum_height[page_index],
                    "page_number": page_index + self.page_from,
                    "layout_type": "table",
                    "layoutno": f"table-{table_index}",
                    "_rotated": True,
                    "_rotation_angle": best_angle,
                    "_table_index": table_index,
                    "_rotated_x0": box_x0,
                    "_rotated_x1": box_x1,
                    "_rotated_top": box_top,
                    "_rotated_bottom": box_bottom,
                }
                self.boxes.insert(insert_at, new_box)
                insert_at += 1
                added += 1
            return added

        # 遍历每个表格，执行重新 OCR 流程
        for tbl_info in table_layouts:
            table_index = tbl_info["table_index"]
            page = tbl_info["page"]
            layout = tbl_info["layout"]
            left, top, right, bott = tbl_info["coords"]

            rotation_info = self.table_rotations.get(table_index, {})
            best_angle = rotation_info.get("best_angle", 0)

            # 获取旋转后的表格图片
            rotated_img = self.rotated_table_imgs.get(table_index)
            if rotated_img is None:
                continue

            # 无旋转则保留原始 OCR 文本框不动
            if best_angle == 0:
                continue

            # 获取表格区域、收集旧文本框
            table_x0, table_top, table_x1, table_bottom, table_top_cum, table_bottom_cum = _table_region(layout, page)
            original_boxes, insert_at = _collect_table_boxes(page, table_x0, table_x1, table_top_cum, table_bottom_cum)

            logging.info(f"Re-OCR table {table_index} on page {page} with rotation {best_angle}°")

            # 对旋转后的图片执行 OCR
            img_array = np.array(rotated_img)
            ocr_results = self.ocr(img_array)

            if not ocr_results:
                logging.warning(f"No OCR results for rotated table {table_index}, restoring originals")
                _restore_boxes(original_boxes, insert_at)
                continue

            # 将 OCR 结果添加到 self.boxes
            table_w_px = right - left   # 表格在旋转前图片中的宽度（像素）
            table_h_px = bott - top     # 表格在旋转前图片中的高度（像素）
            added = _insert_ocr_boxes(
                ocr_results,
                page,
                table_x0,
                table_top,
                insert_at,
                table_index,
                best_angle,
                table_w_px,
                table_h_px,
            )

            logging.info(f"Added {added} OCR results from rotated table {table_index}")

    # ==================== OCR 文本检测与识别 ====================

    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        """
        对单页 PDF 图片执行 OCR 检测和识别。

        流程：
        1. OCR 检测：定位文本行边界框
        2. 排序边界框（按 Y 坐标，阈值 = 平均字符高度的 1/3）
        3. 将 pdfplumber 的字符级文本合并到 OCR 检测到的边界框中
        4. 乱码检测：如果 pdfplumber 文本乱码比例过高，清空文本以触发 OCR 识别
        5. 对于 pdfplumber 无法提取文本的边界框，使用 OCR 识别

        Args:
            pagenum: 页码（从 1 开始）
            img: 页面 PIL Image
            chars: pdfplumber 提取的字符列表
            ZM: 缩放因子
            device_id: GPU 设备 ID（多 GPU 并行时使用）
        """
        start = timer()
        bxs = self.ocr.detect(np.array(img), device_id)
        logging.info(f"__ocr detecting boxes of an image cost ({timer() - start}s)")

        start = timer()
        if not bxs:
            self.boxes.append([])
            return
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [
                {"x0": b[0][0] / ZM, "x1": b[1][0] / ZM, "top": b[0][1] / ZM, "text": "", "txt": t, "bottom": b[-1][1] / ZM, "chars": [], "page_number": pagenum}
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[pagenum - 1] / 3,
        )

        # 将 pdfplumber 字符合并到 OCR 检测到的边界框中
        for c in chars:
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            bxs[ii]["chars"].append(c)

        for b in bxs:
            if not b["chars"]:
                del b["chars"]
                continue
            box_chars = b["chars"]
            m_ht = np.mean([c["height"] for c in box_chars])
            garbled_count = 0
            total_count = 0
            # 按 Y 坐标排序字符并拼接文本
            for c in Recognizer.sort_Y_firstly(box_chars, m_ht):
                if c["text"] == " " and b["text"]:
                    if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                        b["text"] += " "
                else:
                    b["text"] += c["text"]
                    for ch in c["text"]:
                        if not ch.isspace():
                            total_count += 1
                            if self._is_garbled_char(ch):
                                garbled_count += 1
            del b["chars"]

            # 策略 1：PUA / 无法映射的 CID 字符检测
            if total_count > 0 and garbled_count / total_count >= 0.5:
                logging.info(
                    "Page %d: detected garbled pdfplumber text (garbled=%d/%d), falling back to OCR for box at (%.1f, %.1f)",
                    pagenum, garbled_count, total_count, b["x0"], b["top"],
                )
                b["text"] = ""
                continue
            # 策略 2：字体编码乱码 — 所有字符都是子集字体的 ASCII 标点（无 CJK 输出）
            if total_count > 0 and self._is_garbled_by_font_encoding(box_chars, min_chars=5):
                logging.info(
                    "Page %d: detected font-encoding garbled text (%d chars), falling back to OCR for box at (%.1f, %.1f)",
                    pagenum, total_count, b["x0"], b["top"],
                )
                b["text"] = ""

        logging.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        start = timer()
        boxes_to_reg = []
        img_np = None
        # 对于 pdfplumber 无法提取文本的框，使用 OCR 识别
        for b in bxs:
            if not b["text"]:
                if img_np is None:
                    img_np = np.asarray(img)
                left, right, top, bott = b["x0"] * ZM, b["x1"] * ZM, b["top"] * ZM, b["bottom"] * ZM
                b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
                boxes_to_reg.append(b)
            del b["txt"]
        texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
        logging.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[pagenum - 1] == 0:
            self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])
        self.boxes.append(bxs)

    # ==================== 版面分析与文本合并 ====================

    def _layouts_rec(self, ZM, drop=True):
        """
        版面分析：使用 LayoutRecognizer 识别每页的版面结构。

        将页面划分为文本/表格/图片/标题等区域，并更新累积 Y 坐标。

        Args:
            ZM: 缩放因子
            drop: 是否丢弃低置信度检测结果
        """
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
        # 更新累积 Y 坐标
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _assign_column(self, boxes, zoomin=3):
        """
        为文本框分配列 ID（col_id），用于多栏文档的阅读顺序重建。

        使用 K-Means 聚类根据 x0 坐标将文本框分配到不同的列。
        每页独立聚类，全局列数取所有页面的众数。

        Args:
            boxes: 文本框列表
            zoomin: 缩放因子

        Returns:
            list: 带有 col_id 字段的文本框列表
        """
        if not boxes:
            return boxes
        if all("col_id" in b for b in boxes):
            return boxes

        by_page = defaultdict(list)
        for b in boxes:
            by_page[b["page_number"]].append(b)

        page_cols = {}

        # 每页独立聚类确定最佳列数（通过轮廓系数评估）
        for pg, bxs in by_page.items():
            if not bxs:
                page_cols[pg] = 1
                continue

            x0s_raw = np.array([b["x0"] for b in bxs], dtype=float)

            min_x0 = np.min(x0s_raw)
            max_x1 = np.max([b["x1"] for b in bxs])
            width = max_x1 - min_x0

            INDENT_TOL = width * 0.12  # 缩进容忍度：页面宽度的 12%
            x0s = []
            for x in x0s_raw:
                if abs(x - min_x0) < INDENT_TOL:
                    x0s.append([min_x0])
                else:
                    x0s.append([x])
            x0s = np.array(x0s, dtype=float)

            max_try = min(4, len(bxs))
            if max_try < 2:
                max_try = 1
            best_k = 1
            best_score = -1

            # 尝试 1 到 max_try 个聚类，选择轮廓系数最高的
            for k in range(1, max_try + 1):
                km = KMeans(n_clusters=k, n_init="auto")
                labels = km.fit_predict(x0s)

                centers = np.sort(km.cluster_centers_.flatten())
                if len(centers) > 1:
                    try:
                        score = silhouette_score(x0s, labels)
                    except ValueError:
                        continue
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_k = k

            page_cols[pg] = best_k
            logging.info(f"[Page {pg}] best_score={best_score:.2f}, best_k={best_k}")

        # 全局列数取众数
        global_cols = Counter(page_cols.values()).most_common(1)[0][0]
        logging.info(f"Global column_num decided by majority: {global_cols}")

        # 按每页的最佳列数执行聚类并分配 col_id
        for pg, bxs in by_page.items():
            if not bxs:
                continue
            k = page_cols[pg]
            if len(bxs) < k:
                k = 1
            x0s = np.array([[b["x0"]] for b in bxs], dtype=float)
            km = KMeans(n_clusters=k, n_init="auto")
            labels = km.fit_predict(x0s)

            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)  # 按 x 坐标排序列

            remap = {orig: new for new, orig in enumerate(order)}

            for b, lb in zip(bxs, labels):
                b["col_id"] = remap[lb]

        return boxes

    def _text_merge(self, zoomin=3):
        """
        水平合并同一版面区域内相邻的文本框。

        当两个文本框在同一列、同一版面区域、垂直距离小于平均行高的 1/3 时合并。
        表格、图片、公式区域不参与合并。
        """
        bxs = self._assign_column(self.boxes, zoomin)

        def end_with(b, txt):
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]

            # 不同页或不同列的不合并
            if b["page_number"] != b_["page_number"] or b.get("col_id") != b_.get("col_id"):
                i += 1
                continue

            # 不同版面区域或非文本类型不合并
            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue

            # 垂直距离小于 1/3 行高时合并
            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    def _naive_vertical_merge(self, zoomin=3):
        """
        简单的垂直合并：将同一页、重叠度足够、且不以句末标点结束的相邻文本框合并。

        当两个文本框垂直间距小于 1.5 倍行高、水平重叠度大于 30%、
        且上方不以句末标点结束时，将其合并。
        """
        bxs = self.boxes

        grouped = defaultdict(list)
        for b in bxs:
            grouped[(b["page_number"], "x")].append(b)

        merged_boxes = []
        for (pg, col), bxs in grouped.items():
            bxs = sorted(bxs, key=lambda x: (x["top"], x["x0"]))
            if not bxs:
                continue

            mh = self.mean_height[pg - 1] if self.mean_height else np.median([b["bottom"] - b["top"] for b in bxs]) or 10

            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]

                # 跨页且以数字/序号结尾的可能是页码，移除
                if b["page_number"] < b_["page_number"] and re.match(r"[0-9  •一—-]+$", b["text"]):
                    bxs.pop(i)
                    continue

                if not b["text"].strip():
                    bxs.pop(i)
                    continue

                if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
                    i += 1
                    continue

                # 垂直间距过大不合并
                if b_["top"] - b["bottom"] > mh * 1.5:
                    i += 1
                    continue

                # 水平重叠度不足 30% 不合并
                overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
                if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
                    i += 1
                    continue

                # 连接特征（倾向于合并）
                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、'";：-",
                    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，'";：",
                    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?？）),，、：",
                ]
                # 分离特征（倾向于不合并）
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
                ]
                # 完全不相交的分离特征
                detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    logging.debug(
                        "{} {} {} {}".format(
                            b["text"],
                            b_["text"],
                            any(feats),
                            any(concatting_feats),
                        )
                    )
                    i += 1
                    continue

                # 执行合并
                b["text"] = (b["text"].rstrip() + " " + b_["text"].lstrip()).strip()
                b["bottom"] = b_["bottom"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)

            merged_boxes.extend(bxs)

        self.boxes = merged_boxes

    def _final_reading_order_merge(self, zoomin=3):
        """
        按阅读顺序重新排列文本框。

        先分配列 ID，然后按 (页码, 列, top, x0) 排序，
        确保输出顺序符合人类阅读习惯（从上到下，从左到右）。
        """
        if not self.boxes:
            return

        self.boxes = self._assign_column(self.boxes, zoomin=zoomin)

        pages = defaultdict(lambda: defaultdict(list))
        for b in self.boxes:
            pg = b["page_number"]
            col = b.get("col_id", 0)
            pages[pg][col].append(b)

        for pg in pages:
            for col in pages[pg]:
                pages[pg][col].sort(key=lambda x: (x["top"], x["x0"]))

        new_boxes = []
        for pg in sorted(pages.keys()):
            for col in sorted(pages[pg].keys()):
                new_boxes.extend(pages[pg][col])

        self.boxes = new_boxes

    def _concat_downward(self, concat_between_pages=True):
        """
        向下连接：使用 XGBoost 模型判断上下两个文本框是否应合并。

        当前版本简化了处理——仅按 Y 坐标排序，不执行实际的模型推理。
        完整的 DFS+模型连接逻辑保留在注释代码中，可根据需要启用。

        Args:
            concat_between_pages: 是否允许跨页连接
        """
        self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
        return

        # ---- 以下为完整版本的 DFS + XGBoost 模型跨段落连接逻辑 ----
        # 计算同行文本框数量（用于特征提取）
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                """DFS 搜索可连接的文本框序列。"""
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if not concat_between_pages and down["page_number"] > up["page_number"]:
                        break

                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"]) or not down["text"].strip():
                        i += 1
                        continue

                    if not down["text"].strip() or not up["text"].strip():
                        i += 1
                        continue

                    if up["x1"] < down["x0"] - 10 * mw or up["x0"] > down["x1"] + 10 * mw:
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # 合并每个 block 内的文本框
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    # ==================== 目录页过滤与碎片清理 ====================

    def _filter_forpages(self):
        """
        过滤目录页和无意义内容。

        两种检测策略：
        1. 检测"目录/目次/Table of Contents"等关键词，删除目录页内容
        2. 检测包含大量点线（···）的页面（通常为目录页），删除相关内容
        """
        if not self.boxes:
            return
        findit = False
        i = 0
        # 策略 1：检测目录关键词
        while i < len(self.boxes):
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", re.sub(r"( | |　)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        # 策略 2：检测包含大量点线（目录引导符）的页面
        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        """
        合并相邻且以相同项目符号开头的文本框。

        用于处理被错误分割的列表项，将同一项目的多行合并。
        """
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            # 首字符相同且非字母、非中文、且上方不在下方之后的才合并
            if (
                b["text"].strip()[0] != b_["text"].strip()[0]
                or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or rag_tokenizer.is_chinese(b["text"].strip()[0])
                or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    # ==================== 表格与图片提取 ====================

    def _extract_table_figure(self, need_image, ZM, return_html, need_position, separate_tables_figures=False):
        """
        从解析结果中提取表格和图片。

        流程：
        1. 从 self.boxes 中分离出表格和图片类型的文本框
        2. 跨页合并被分割的表格
        3. 将标题（caption）与对应的表格/图片关联
        4. 裁剪表格/图片区域生成缩略图
        5. 将表格转换为 HTML 格式（可选）

        Args:
            need_image: 是否需要提取图片
            ZM: 缩放因子
            return_html: 是否将表格转为 HTML 格式
            need_position: 是否返回位置信息
            separate_tables_figures: 是否分开返回表格和图片

        Returns:
            list 或 tuple: 裁剪后的 (图片, 文本) 列表
        """
        tables = {}
        figures = {}
        # 分离表格和图片框
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            # 标题类型的版面区域标记为不跨页合并
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # 跨页合并表格
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            """检查两个框在水平方向是否重叠。"""
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # 将标题与最近的表格/图片关联
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            def nearest(tbls):
                """找到距离标题最近的表格/图片。"""
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c)
                logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            """
            裁剪表格/图片区域的缩略图。

            支持单页和多页裁剪。多页时垂直拼接各页的裁剪结果。
            """
            nonlocal ZM
            max_page_index = len(self.page_images) - 1

            def local_page_index(page_number):
                """将全局页码转为本地页面索引。"""
                idx = page_number - 1 if page_number > 0 else 0
                if idx > max_page_index and self.page_from:
                    idx = page_number - 1 - self.page_from
                return idx

            pn = set()
            for b in bxs:
                idx = local_page_index(b["page_number"])
                if 0 <= idx <= max_page_index:
                    pn.add(idx)
                else:
                    logging.warning(
                        "Skip out-of-range page_number %s (page_from=%s, pages=%s)",
                        b.get("page_number"),
                        self.page_from,
                        len(self.page_images),
                    )

            if not pn:
                return None

            if len(pn) < 2:
                # 单页情况：直接裁剪
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {"x0": np.min([b["x0"] for b in bxs]), "top": np.min([b["top"] for b in bxs]) - ht, "x1": np.max([b["x1"] for b in bxs]), "bottom": np.max([b["bottom"] for b in bxs]) - ht}
                louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logging.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM))

            # 多页情况：分组裁剪后垂直拼接
            pn = {}
            for b in bxs:
                p = local_page_index(b["page_number"])
                if 0 <= p <= max_page_index:
                    if p not in pn:
                        pn[p] = []
                    pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            imgs = [img for img in imgs if img is not None]
            if not imgs:
                return None
            pic = Image.new("RGB", (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))), (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # 裁剪图片
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            if separate_tables_figures:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                figure_results.append((img, [txt]))
                figure_positions.append(poss)
            else:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                res.append((img, [txt]))
                positions.append(poss)

        # 裁剪表格
        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            img = cropout(bxs, "table", poss)
            if img is None:
                continue
            res.append((img, self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    # ==================== 项目符号匹配与位置标签生成 ====================

    def proj_match(self, line):
        """
        检查文本行是否匹配项目符号/标题/编号模式。

        返回匹配的优先级编号（数字越大优先级越高）。

        Args:
            line: 文本行

        Returns:
            int 或 None: 匹配优先级，不匹配返回 None
        """
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        """
        为文本框生成 RAGFlow 位置标签。

        格式：@@页码\tx0\tx1\ttop\tbottom##

        支持跨页文本框（当文本框跨越多个 PDF 页面时，页码用 '-' 连接）。

        Args:
            bx: 文本框字典
            ZM: 缩放因子

        Returns:
            str: 位置标签字符串
        """
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        # 处理跨页情况：逐页减去页面高度
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):
        """
        过滤碎片文本，生成最终输出。

        使用递归深度优先搜索（DFS）收集连续的、有意义的文本行，
        过滤掉宽度过窄、高度过低的碎片文本。

        保留标准：
        - 有版面类型标签（layout_type）的文本
        - 宽度超过页面 1/3 的文本
        - 高度超过平均行高的文本

        Args:
            boxes: 文本框列表
            ZM: 缩放因子

        Returns:
            str: 带有位置标签的最终文本输出
        """
        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            """判断文本框是否有意义。"""
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(boxes[0]["text"]) or boxes[0].get("layout_type", "") == "title"

            def dfs(line, st):
                """DFS 收集连续的有意义文本行。"""
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                mmj = self.proj_match(line["text"]) or line.get("layout_type", "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (self._x_dis(boxes[i], line) < pw / 10):
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            # 保留条件：项目符号/标题 或 宽文本 或 平均宽度占页面 35% 以上
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append("\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    # ==================== PDF 页面分析与渲染主流程 ====================

    @staticmethod
    def total_page_number(fnm, binary=None):
        """
        获取 PDF 的总页数。

        使用 pdfplumber 打开 PDF 并统计页数。加全局锁以避免多线程竞态。

        Args:
            fnm: PDF 文件路径或二进制数据
            binary: 文件的二进制内容

        Returns:
            int 或 None: 总页数
        """
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = pdfplumber.open(fnm) if not binary else pdfplumber.open(BytesIO(binary))
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logging.exception("total_page_number")

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        """
        渲染 PDF 页面图片并提取字符级文本（解析流水线第一步）。

        使用 pdfplumber 将每页渲染为高分辨率图片，同时提取字符级
        文本信息（位置、字体、颜色等）。

        乱码检测：在提取字符后，检测两种乱码模式并清空乱码页面的字符，
        强制后续使用 OCR 路径：
        1. PUA / CID 乱码（阈值 30%）
        2. 字体编码乱码（子集字体 + 无 CJK 输出）

        Args:
            fnm: PDF 文件路径或二进制数据
            zoomin: 缩放因子（默认 3 = 216 DPI）
            page_from: 起始页码
            page_to: 结束页码
            callback: 进度回调
        """
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        start = timer()
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                with pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm)) as pdf:
                    self.pdf = pdf
                    self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]

                    try:
                        self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
                    except Exception as e:
                        logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                        self.page_chars = [[] for _ in range(len(self.page_images))]  # 提取失败时使用空列表

                    # 乱码检测：两种策略
                    for pi, page_ch in enumerate(self.page_chars):
                        if not page_ch:
                            continue
                        # 策略 1：PUA / CID 乱码
                        sample = page_ch if len(page_ch) <= 200 else page_ch[:200]
                        sample_text = "".join(c.get("text", "") for c in sample)
                        if self._is_garbled_text(sample_text, threshold=0.3):
                            logging.warning(
                                "Page %d: pdfplumber extracted mostly garbled characters (%d chars), "
                                "clearing to use OCR fallback.",
                                page_from + pi + 1, len(page_ch),
                            )
                            self.page_chars[pi] = []
                            continue
                        # 策略 2：字体编码乱码（CJK 映射到 ASCII）
                        if self._is_garbled_by_font_encoding(page_ch):
                            logging.warning(
                                "Page %d: detected font-encoding garbled text "
                                "(subset fonts with no CJK output, %d chars), "
                                "clearing to use OCR fallback.",
                                page_from + pi + 1, len(page_ch),
                            )
                            self.page_chars[pi] = []

                    self.total_page = len(self.pdf.pages)

        except Exception as e:
            logging.exception(f"RAGFlowPdfParser __images__, exception: {e}")
        logging.info(f"__images__ dedupe_chars cost {timer() - start}s")

        logging.debug("Images converted.")
        # 语言检测：判断 PDF 是否为英文
        self.is_english = [
            re.search(r"[ a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i])))))
            for i in range(len(self.page_chars))
        ]
        if sum([1 if e else 0 for e in self.is_english]) > len(self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        async def __img_ocr(i, id, img, chars, limiter):
            """异步 OCR 单页图片。"""
            j = 0
            while j + 1 < len(chars):
                if (
                    chars[j]["text"]
                    and chars[j + 1]["text"]
                    and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                ):
                    chars[j]["text"] += " "
                j += 1

            if limiter:
                async with limiter:
                    await thread_pool_exec(self.__ocr, i + 1, img, chars, zoomin, id)
            else:
                self.__ocr(i + 1, img, chars, zoomin, id)

            if callback and i % 6 == 5:
                callback((i + 1) * 0.6 / len(self.page_images))

        async def __img_ocr_launcher():
            """启动所有页面的 OCR 任务。"""
            def __ocr_preprocess():
                chars = self.page_chars[i] if not self.is_english else []
                self.mean_height.append(np.median(sorted([c["height"] for c in chars])) if chars else 0)
                self.mean_width.append(np.median(sorted([c["width"] for c in chars])) if chars else 8)
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars

            if self.parallel_limiter:
                tasks = []

                for i, img in enumerate(self.page_images):
                    chars = __ocr_preprocess()

                    semaphore = self.parallel_limiter[i % settings.PARALLEL_DEVICES]

                    async def wrapper(i=i, img=img, chars=chars, semaphore=semaphore):
                        await __img_ocr(
                            i,
                            i % settings.PARALLEL_DEVICES,
                            img,
                            chars,
                            semaphore,
                        )

                    tasks.append(asyncio.create_task(wrapper()))
                    await asyncio.sleep(0)

                try:
                    await asyncio.gather(*tasks, return_exceptions=False)
                except Exception as e:
                    logging.error(f"Error in OCR: {e}")
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

            else:
                for i, img in enumerate(self.page_images):
                    chars = __ocr_preprocess()
                    await __img_ocr(i, 0, img, chars, None)

        start = timer()

        asyncio.run(__img_ocr_launcher())

        logging.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")

        # OCR 后重新检测语言
        if not self.is_english and not any([c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[ \na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        logging.debug(f"Is it English: {self.is_english}")

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        # 如果没有检测到任何文本框，增大缩放因子重试
        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)

    # ==================== 主解析入口 ====================

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False, auto_rotate_tables=None):
        """
        解析 PDF 文件（主入口方法，完整流水线）。

        执行完整的 PDF 文档理解流水线：
        1. __images__: 渲染页面，OCR 文本
        2. _layouts_rec: 版面分析
        3. _table_transformer_job: 表格结构识别
        4. _text_merge: 水平合并文本框
        5. _concat_downward: 垂直连接
        6. _filter_forpages: 过滤目录页
        7. _extract_table_figure: 提取表格和图片
        8. __filterout_scraps: 过滤碎片，生成最终输出

        Args:
            fnm: PDF 文件路径或二进制内容
            need_image: 是否提取图片
            zoomin: 缩放因子（默认 3 = 216 DPI）
            return_html: 是否以 HTML 格式返回表格
            auto_rotate_tables: 是否启用表格自动方向纠正。
                               None: 使用 TABLE_AUTO_ROTATE 环境变量（默认 True）
                               True: 启用自动方向纠正
                               False: 禁用自动方向纠正

        Returns:
            tuple: (文本行列表, 表格/图片列表)
        """
        if auto_rotate_tables is None:
            auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

        self.outlines = extract_pdf_outlines(fnm)
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    # ==================== 分块解析模式 ====================

    def parse_into_bboxes(self, fnm, callback=None, zoomin=3, from_page=0, to_page=MAXIMUM_PAGE_NUMBER):
        """
        分块解析 PDF，返回所有文本框的边界框信息。

        对于页数较多的 PDF，分批处理以控制内存使用。
        批次大小由 PDF_PARSER_PAGE_BATCH_SIZE 环境变量控制（默认 50 页）。

        Args:
            fnm: PDF 文件路径或二进制内容
            callback: 进度回调
            zoomin: 缩放因子
            from_page: 起始页码
            to_page: 结束页码

        Returns:
            list: 所有文本框的边界框列表
        """
        self.outlines = extract_pdf_outlines(fnm)
        batch_size = max(1, int(os.getenv("PDF_PARSER_PAGE_BATCH_SIZE", "50")))
        if isinstance(fnm, str):
            total_pages = self.total_page_number(fnm)
        else:
            total_pages = self.total_page_number(fnm, binary=fnm)

        if total_pages is None:
            effective_to_page = to_page
            logging.warning(
                "parse_into_bboxes: total_page_number returned None; using caller-supplied to_page=%s",
                to_page,
            )
        else:
            effective_to_page = min(to_page, total_pages)

        if effective_to_page - from_page <= batch_size:
            self.__images__(fnm, zoomin, page_from=from_page, page_to=effective_to_page, callback=callback)
            return self._parse_loaded_window_into_bboxes(zoomin, callback=callback)

        # 分批处理模式
        logging.info(
            "parse_into_bboxes uses chunk mode: from_page=%s, effective_to_page=%s, batch_size=%s",
            from_page,
            effective_to_page,
            batch_size,
        )
        all_boxes = []
        start = timer()
        for page_from in range(from_page, effective_to_page, batch_size):
            page_to = min(page_from + batch_size, effective_to_page)
            self.__images__(fnm, zoomin, page_from=page_from, page_to=page_to, callback=None)
            chunk_boxes = self._parse_loaded_window_into_bboxes(zoomin)
            all_boxes.extend(self._to_global_boxes(chunk_boxes))
            if callback:
                callback((page_to - from_page) / max(1, effective_to_page - from_page), f"Structured: {page_to}/{effective_to_page} pages")

        logging.info("parse_into_bboxes chunk mode cost %.2fs", timer() - start)
        return all_boxes

    def _parse_loaded_window_into_bboxes(self, zoomin=3, callback=None):
        """
        对已加载的页面窗口执行版面分析和表格识别。

        处理步骤：
        1. 版面分析（layouts_rec）
        2. 表格结构识别（table_transformer_job）
        3. 文本合并
        4. 表格/图片提取
        5. 将表格/图片插入到文本框列表的适当位置

        Args:
            zoomin: 缩放因子
            callback: 进度回调

        Returns:
            list: 包含表格/图片的完整文本框列表
        """
        start = timer()
        self._layouts_rec(zoomin)
        if callback:
            callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

        start = timer()
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        if callback:
            callback(0.83, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        self._concat_downward()
        self._naive_vertical_merge(zoomin)
        if callback:
            callback(0.92, "Text merged ({:.2f}s)".format(timer() - start))

        start = timer()
        tbls, figs = self._extract_table_figure(True, zoomin, True, True, True)

        def insert_table_figures(tbls_or_figs, layout_type):
            """
            将表格/图片插入到文本框列表的适当位置。

            使用最小矩形距离找到最近的文本框，将表格/图片
            插入到该文本框之后（遵循阅读顺序）。
            """
            def min_rectangle_distance(rect1, rect2):
                """计算两个矩形之间的最小距离。"""
                pn1, left1, right1, top1, bottom1 = rect1
                pn2, left2, right2, top2, bottom2 = rect2
                if right1 >= left2 and right2 >= left1 and bottom1 >= top2 and bottom2 >= top1:
                    return 0
                if right1 < left2:
                    dx = left2 - right1
                elif right2 < left1:
                    dx = left1 - right2
                else:
                    dx = 0
                if bottom1 < top2:
                    dy = top2 - bottom1
                elif bottom2 < top1:
                    dy = top1 - bottom2
                else:
                    dy = 0
                return math.sqrt(dx * dx + dy * dy)

            for (img, txt), poss in tbls_or_figs:
                local_poss = []
                for pn, left, right, top, bott in poss:
                    local_pn = pn - self.page_from
                    if 0 <= local_pn < len(self.page_cum_height) - 1:
                        local_poss.append((local_pn, left, right, top, bott))
                    else:
                        logging.debug(f"Skip out-of-range table/figure position pn={pn}, page_from={self.page_from}")
                if not local_poss:
                    logging.debug("No valid local positions for table/figure; skip insertion.")
                    continue

                if isinstance(txt, list):
                    txt = "\n".join(txt)
                pn, left, right, top, bott = local_poss[0]
                insert_at = len(self.boxes)
                bboxes = [(i, (b["page_number"], b["x0"], b["x1"], b["top"], b["bottom"])) for i, b in enumerate(self.boxes)]
                if bboxes:
                    dists = [
                        (min_rectangle_distance((cand_pn, cand_left, cand_right, cand_top + self.page_cum_height[cand_pn], cand_bott + self.page_cum_height[cand_pn]), rect), i)
                        for i, rect in bboxes
                        for cand_pn, cand_left, cand_right, cand_top, cand_bott in local_poss
                    ]
                    if dists:
                        nearest_bbox_idx = int(np.argmin([dist for dist, _ in dists]))
                        insert_at, _ = bboxes[dists[nearest_bbox_idx][-1]]
                        if self.boxes[insert_at]["bottom"] < top + self.page_cum_height[pn]:
                            insert_at += 1
                else:
                    logging.debug("No text boxes available; append %s block directly.", layout_type)
                self.boxes.insert(
                    insert_at,
                    {
                        "page_number": pn + 1,
                        "x0": left,
                        "x1": right,
                        "top": top + self.page_cum_height[pn],
                        "bottom": bott + self.page_cum_height[pn],
                        "layout_type": layout_type,
                        "text": txt,
                        "image": img,
                        "positions": [[pn + 1, int(left), int(right), int(top), int(bott)]],
                    },
                )

        # 为每个文本框生成位置标签和裁剪图片
        for b in self.boxes:
            b["position_tag"] = self._line_tag(b, zoomin)
            b["image"] = self.crop(b["position_tag"], zoomin)
            b["positions"] = [[pos[0][-1] + 1, *pos[1:]] for pos in RAGFlowPdfParser.extract_positions(b["position_tag"])]

        insert_table_figures(tbls, "table")
        insert_table_figures(figs, "figure")
        if callback:
            callback(1, "Structured ({:.2f}s)".format(timer() - start))
        return deepcopy(self.boxes)

    # ==================== 辅助方法 ====================

    @staticmethod
    def _offset_position_tag(text, page_offset):
        """
        将位置标签中的页码偏移指定数量。

        用于分块解析模式中，将局部页码转换为全局页码。

        Args:
            text: 包含 @@页码## 标签的文本
            page_offset: 页码偏移量

        Returns:
            str: 偏移后的文本
        """
        if not text or page_offset <= 0:
            return text

        def _replace(match):
            pages = [str(int(p) + page_offset) for p in match.group(1).split("-")]
            return f"@@{'-'.join(pages)}\t"

        return re.sub(r"@@([0-9-]+)\t", _replace, text)

    def _to_global_boxes(self, boxes):
        """
        将局部页码转换为全局页码。

        在分块解析模式中，每批次的页码从 0 开始，需要加回 page_from
        得到原始 PDF 的页码。

        Args:
            boxes: 文本框列表

        Returns:
            list: 全局页码的文本框列表
        """
        if self.page_from <= 0:
            return boxes

        for box in boxes:
            box["page_number"] = int(box.get("page_number", 1)) + self.page_from
            if isinstance(box.get("position_tag"), str):
                box["position_tag"] = self._offset_position_tag(box["position_tag"], self.page_from)
            if isinstance(box.get("positions"), list):
                box["positions"] = [
                    [int(pos[0]) + self.page_from, *pos[1:]]
                    if isinstance(pos, list) and len(pos) > 0 and isinstance(pos[0], (int, float))
                    else pos
                    for pos in box["positions"]
                ]
        return boxes

    @staticmethod
    def remove_tag(txt):
        """从文本中去除位置标签。"""
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    @staticmethod
    def extract_positions(txt):
        """
        从文本中提取位置标签信息。

        解析 @@页码\t左\t右\t上\t下## 格式的标签。

        Args:
            txt: 包含位置标签的文本

        Returns:
            list: [(页码列表, left, right, top, bottom), ...]
        """
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text, ZM=3, need_position=False):
        """
        根据位置标签从 PDF 页面裁剪对应区域的图片。

        支持跨页裁剪，将多页裁剪结果垂直拼接为一张图片。
        首尾区域添加半透明遮罩以标记上下文填充内容。

        Args:
            text: 包含位置标签的文本
            ZM: 缩放因子
            need_position: 是否同时返回位置信息

        Returns:
            PIL.Image 或 (PIL.Image, positions) 或 None
        """
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            logging.warning("crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                logging.warning("Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                logging.warning(f"All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            logging.warning("No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        # 在首尾插入上下文填充区域（120px 高度）
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            logging.warning(f"Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1] / ZM
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if 0 < ii < len(poss) - 1:
                right = max(left + 10, right)
            else:
                right = left + max_width
            bottom *= ZM
            # 累加跨页高度
            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    logging.warning(f"Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                logging.warning(f"Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            imgs.append(self.page_images[pns[0]].crop((left * ZM, top * ZM, right * ZM, min(bottom, self.page_images[pns[0]].size[1]))))
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    logging.warning(f"Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                imgs.append(self.page_images[pn].crop((left * ZM, 0, right * ZM, min(bottom, self.page_images[pn].size[1]))))
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        # 垂直拼接所有裁剪图片
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            # 首尾图片（上下文填充区域）添加半透明遮罩
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        """
        获取文本框的页面位置信息。

        Args:
            bx: 文本框字典
            ZM: 缩放因子

        Returns:
            list: [(页码, x0, x1, top, bottom), ...]
        """
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss


class PlainParser:
    """
    轻量级 PDF 解析器。

    使用 pypdf（PyPDF2 的继任者）提取纯文本，不做版面分析或 OCR。
    适用于已有文本层且不需要结构化信息的简单 PDF 文档。

    输出为 (文本行, 空位置标签) 的列表。
    """

    def __call__(self, filename, from_page=0, to_page=MAXIMUM_PAGE_NUMBER, **kwargs):
        """
        解析 PDF 文件，提取纯文本。

        Args:
            filename: PDF 文件路径或二进制内容
            from_page: 起始页码
            to_page: 结束页码

        Returns:
            tuple: (文本行列表, 空表格列表)
        """
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])
        except Exception:
            logging.exception("Outlines exception")
        self.outlines = extract_pdf_outlines(filename)

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        """PlainParser 不支持图片裁剪。"""
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        """PlainParser 不支持标签移除。"""
        raise NotImplementedError


class VisionParser(RAGFlowPdfParser):
    """
    视觉语言模型（VLM）解析器。

    使用视觉模型（如 GPT-4 Vision、Qwen-VL 等）对 PDF 每一页进行
    端到端的视觉理解，生成自然语言描述文本。

    与 RAGFlowPdfParser 的区别：
    - 不做 OCR 文字检测/识别
    - 不做版面分析
    - 不做表格结构识别
    - 直接调用 VLM 对整页图片进行描述

    适用场景：图片型 PDF、复杂排版 PDF、需要语义理解的文档。
    """

    def __init__(self, vision_model, *args, **kwargs):
        """
        初始化视觉解析器。

        Args:
            vision_model: 视觉语言模型实例
        """
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        self.outlines = []

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        """
        渲染 PDF 页面图片（简化版，不做字符提取）。

        Args:
            fnm: PDF 文件路径或二进制内容
            zoomin: 缩放因子
            page_from: 起始页码
            page_to: 结束页码
            callback: 进度回调
        """
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=MAXIMUM_PAGE_NUMBER, **kwargs):
        """
        使用视觉模型解析 PDF 文件。

        对每一页调用 VLM 生成自然语言描述，然后将描述文本与位置标签
        组合为最终输出。

        Args:
            filename: PDF 文件路径
            from_page: 起始页码
            to_page: 结束页码
            **kwargs: 可包含 callback、zoomin 等参数

        Returns:
            tuple: (文档描述列表, 空表格列表)
        """
        callback = kwargs.get("callback", lambda prog, msg: None)
        zoomin = kwargs.get("zoomin", 3)
        self.__images__(fnm=filename, zoomin=zoomin, page_from=from_page, page_to=to_page, callback=callback)

        total_pdf_pages = self.total_page

        start_page = max(0, from_page)
        end_page = min(to_page, total_pdf_pages)

        all_docs = []

        for idx, img_binary in enumerate(self.page_images or []):
            pdf_page_num = idx  # 0-based
            if pdf_page_num < start_page or pdf_page_num >= end_page:
                continue

            from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk

            # 调用 VLM 生成页面描述
            text = picture_vision_llm_chunk(
                binary=img_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_describe_prompt(page=pdf_page_num + 1),
                callback=callback,
            )

            if kwargs.get("callback"):
                kwargs["callback"](idx * 1.0 / len(self.page_images), f"Processed: {idx + 1}/{len(self.page_images)}")

            if text:
                width, height = self.page_images[idx].size
                all_docs.append((text, f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{width / zoomin:.1f}\t{0.0:.1f}\t{height / zoomin:.1f}##"))
        return all_docs, []


if __name__ == "__main__":
    pass
