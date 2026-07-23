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
OCR 独立测试脚本

用于对输入图像或 PDF 文件执行 OCR 文字检测与识别，并将结果可视化输出。
支持多 GPU 并行加速。

用法:
    python t_ocr.py --inputs <图片/PDF路径或目录> --output_dir <输出目录>

GPU 配置:
    - 单 GPU:    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    - 多 GPU:    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    - CPU 模式:  os.environ['CUDA_VISIBLE_DEVICES'] = ''

输出:
    - 绘制了 OCR 检测框的图片
    - 对应的 .txt 文本文件
"""

import asyncio
import logging
import os
import sys


from common.misc_utils import thread_pool_exec

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.seeit import draw_box
from deepdoc.vision import OCR, init_in_out
import argparse
import numpy as np

# GPU 设备配置：取消对应行注释即可切换模式
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' # 多 GPU（不连续）
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 单 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU 模式


def main(args):
    """
    主函数：执行 OCR 检测与识别。

    流程：
    1. 初始化 OCR 引擎
    2. 加载输入图像/PDF
    3. 使用异步协程并行处理所有页面（多 GPU 时使用信号量控制并发）
    4. 将检测结果绘制到图像并保存

    Args:
        args: 命令行参数
    """
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    # 多 GPU 时，每个设备分配一个信号量控制并发
    limiter = [asyncio.Semaphore(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    ocr = OCR()
    images, outputs = init_in_out(args)

    def __ocr(i, id, img):
        """对单张图像执行 OCR，绘制结果并保存"""
        print("Task {} start".format(i))
        bxs = ocr(np.array(img), id)
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{
            "text": t,
            "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
            "type": "ocr",
            "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(images[i], bxs, ["ocr"], 1.)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join([o["text"] for o in bxs]))

        print("Task {} done".format(i))

    async def __ocr_thread(i, id, img, limiter = None):
        """异步 OCR 任务包装器，支持信号量限流"""
        if limiter:
            async with limiter:
                print(f"Task {i} use device {id}")
                await thread_pool_exec(__ocr, i, id, img)
        else:
            await thread_pool_exec(__ocr, i, id, img)


    async def __ocr_launcher():
        """启动所有异步 OCR 任务"""
        tasks = []
        for i, img in enumerate(images):
            dev_id = i % cuda_devices if cuda_devices > 1 else 0
            semaphore = limiter[dev_id] if limiter else None
            tasks.append(asyncio.create_task(__ocr_thread(i, dev_id, img, semaphore)))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("OCR tasks failed: {}".format(e))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    asyncio.run(__ocr_launcher())

    print("OCR tasks are all done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir', help="Directory where to store the output images. Default: './ocr_outputs'",
                        default="./ocr_outputs")
    args = parser.parse_args()
    main(args)
