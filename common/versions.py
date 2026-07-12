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
版本信息模块

提供 RAGFlow 版本号的获取功能，支持多种版本来源：
- 优先读取项目根目录下的 VERSION 文件（正式发布版本）
- 若 VERSION 文件不存在，通过 git describe 命令获取最近的 tag 和提交数（开发版本）
- 以上两种方式都失败时返回 "unknown"
"""

import os
import subprocess

# 全局版本号缓存，避免重复读取文件或执行 git 命令
RAGFLOW_VERSION_INFO = "unknown"


def get_ragflow_version() -> str:
    """获取 RAGFlow 当前版本号。

    读取策略（按优先级）：
    1. 若已缓存且不为 "unknown"，直接返回缓存值
    2. 尝试读取项目根目录的 VERSION 文件
    3. VERSION 文件不存在时，通过 git describe 获取开发版本信息

    Returns:
        str: 版本号字符串，如 "v0.17.0" 或 "v0.17.0-15-g1a2b3c4"
    """
    global RAGFLOW_VERSION_INFO
    if RAGFLOW_VERSION_INFO != "unknown":
        return RAGFLOW_VERSION_INFO
    version_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, "VERSION"
        )
    )
    if os.path.exists(version_path):
        with open(version_path, "r") as f:
            RAGFLOW_VERSION_INFO = f.read().strip()
    else:
        RAGFLOW_VERSION_INFO = get_closest_tag_and_count()
    return RAGFLOW_VERSION_INFO


def get_closest_tag_and_count():
    """通过 git describe 获取最近的 tag 和当前提交信息。

    使用 --first-parent 仅沿第一父提交回溯，避免合并提交干扰；
    --always 确保在没有 tag 时仍返回 commit hash 短名。

    Returns:
        str: git describe 输出，如 "v0.17.0-15-g1a2b3c4"；失败时返回 "unknown"
    """
    try:
        version_info = (
            subprocess.check_output(["git", "describe", "--tags", "--match=v*", "--first-parent", "--always"])
            .strip()
            .decode("utf-8")
        )
        return version_info
    except Exception:
        return "unknown"
