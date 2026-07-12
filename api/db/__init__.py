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

from enum import IntEnum
from enum import StrEnum

from common.constants import PipelineTaskType


class UserTenantRole(StrEnum):
    """用户-租户关系中的角色枚举。

    定义用户在租户（团队）中的四种角色：
    - OWNER:  租户所有者，拥有最高权限
    - ADMIN:  管理员
    - NORMAL: 普通成员，已接受邀请
    - INVITE: 受邀状态，尚未接受邀请
    """
    OWNER = 'owner'
    ADMIN = 'admin'
    NORMAL = 'normal'
    INVITE = 'invite'


class TenantPermission(StrEnum):
    """知识库的可见权限范围。

    - ME:   仅自己可见
    - TEAM: 团队内所有成员可见
    """
    ME = 'me'
    TEAM = 'team'


class SerializedType(IntEnum):
    """序列化方式枚举。"""
    PICKLE = 1
    JSON = 2


class FileType(StrEnum):
    """文件类型枚举。

    涵盖 RAGFlow 支持的所有文档类型：
    - PDF:     PDF 文档
    - DOC:     Office 文档（DOCX, XLSX, PPT 等）
    - VISUAL:  图片文件
    - AURAL:   音频文件
    - VIRTUAL: 知识库级别的处理结果（如手动添加的条目）
    - FOLDER:  文件夹
    - OTHER:   其他类型
    """
    PDF = 'pdf'
    DOC = 'doc'
    VISUAL = 'visual'
    AURAL = 'aural'
    VIRTUAL = 'virtual'
    FOLDER = 'folder'
    OTHER = "other"

# 所有有效的文件类型集合，用于快速类型校验
VALID_FILE_TYPES = {FileType.PDF, FileType.DOC, FileType.VISUAL, FileType.AURAL, FileType.VIRTUAL, FileType.FOLDER, FileType.OTHER}


class InputType(StrEnum):
    """数据源的输入方式枚举。

    定义 connector 从外部数据源拉取数据的方式：
    - LOAD_STATE:     加载完整的当前状态或保存的快照（如从文件加载）
    - POLL:           定期轮询（如每小时调用 API 获取新文档）
    - EVENT:          事件驱动（注册监听端点，处理 connector 事件）
    - SLIM_RETRIEVAL: 轻量检索模式
    """
    LOAD_STATE = "load_state"  # 加载当前完整状态或保存的快照，例如从文件加载
    POLL = "poll"  # 轮询模式，例如定时调用 API 获取最近一小时的所有文档
    EVENT = "event"  # 事件模式，例如注册端点作为监听器，处理 connector 事件
    SLIM_RETRIEVAL = "slim_retrieval"  # 轻量检索


class CanvasCategory(StrEnum):
    """画布（工作流）类别枚举。

    - Agent:    智能体画布，用于构建 LLM Agent 工作流
    - DataFlow: 数据流画布，用于构建文档处理流水线
    """
    Agent = "agent_canvas"
    DataFlow = "dataflow_canvas"


# 有效的流水线任务类型集合，包含解析、下载、RAPTOR、图谱RAG、思维导图
VALID_PIPELINE_TASK_TYPES = {PipelineTaskType.PARSE, PipelineTaskType.DOWNLOAD, PipelineTaskType.RAPTOR, PipelineTaskType.GRAPH_RAG, PipelineTaskType.MINDMAP}

# 进度冻结类任务类型：这些任务在进度计算中有特殊处理逻辑，不会被常规进度更新所覆盖
PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES = {PipelineTaskType.RAPTOR.lower(), PipelineTaskType.GRAPH_RAG.lower(), PipelineTaskType.MINDMAP.lower()}

# 知识库元数据文件夹名称（隐藏文件夹，存放知识库配置）
KNOWLEDGEBASE_FOLDER_NAME=".knowledgebase"
# 技能文件夹名称
SKILLS_FOLDER_NAME="skills"
