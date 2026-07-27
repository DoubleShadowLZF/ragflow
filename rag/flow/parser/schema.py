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

"""解析器模块的数据模型定义。

定义 Parser 组件从上游（文件上传组件）接收的输入数据结构，
使用 Pydantic 进行数据校验和序列化。
"""

from pydantic import BaseModel, ConfigDict, Field


class ParserFromUpstream(BaseModel):
    """Parser 组件从上游接收的输入数据模型。

    表示一个待解析的文件，包含文件元信息和可选的处理标志。
    由 Pipeline 中的上游组件（如 File 组件）传递而来。
    """

    # 文件创建时间戳（由上游组件设置）
    created_time: float | None = Field(default=None, alias="_created_time")

    # 上游组件处理耗时（秒），由上游组件设置
    elapsed_time: float | None = Field(default=None, alias="_elapsed_time")

    # 文件名（含扩展名），用于识别文件类型并选择对应的解析器
    name: str

    # 文件元信息字典，包含 id、created_by、存储路径等
    file: dict | None = Field(default=None)

    # 是否提取摘要信息
    abstract: bool = False

    # 是否提取作者信息
    author: bool = False

    # Pydantic 配置：允许通过字段别名（如 _created_time）赋值，禁止额外字段
    model_config = ConfigDict(populate_by_name=True, extra="forbid")