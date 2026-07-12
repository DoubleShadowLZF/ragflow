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
RSA 加密/解密工具模块 —— 用于密码等敏感数据的非对称加密传输。

本模块提供基于 RSA + PKCS#1 v1.5 填充的加密解密函数，供前端和
ragflow_cli 命令行工具使用。加密流程：

    明文 → Base64 → RSA 公钥加密 → Base64 → 密文字符串

即 `decrypt(crypt(plaintext)) == base64(plaintext)`。

密钥文件位置：项目根目录下的 ``conf/public.pem``（公钥）和
``conf/private.pem``（私钥），密码短语为 ``"Welcome"``。

注意：模块中导入了两套加密库：
- ``Cryptodome`` —— 来自 PyCryptodome，供 ``crypt()`` / ``decrypt()`` 使用
- ``Crypto`` —— 旧版 pycrypto，供 ``decrypt2()`` 兼容使用（处理历史密文）
"""

import base64  # Base64 编解码
import os  # 文件路径拼接
import sys  # 命令行参数读取
from pathlib import Path  # 文件内容读取
from Cryptodome.PublicKey import RSA  # RSA 公钥/私钥导入（PyCryptodome 版本）
from Cryptodome.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5  # PKCS#1 v1.5 填充加密（PyCryptodome）
from common.file_utils import get_project_base_directory  # 获取项目根目录路径


def crypt(line):
    """使用 RSA 公钥加密字符串。

    加密流程：
    1. 读取 ``conf/public.pem`` 公钥文件（密码短语 "Welcome"）
    2. 明文 → UTF-8 编码 → Base64 编码
    3. 用 PKCS#1 v1.5 进行 RSA 公钥加密
    4. 密文 → Base64 编码 → 返回可传输的字符串

    该函数满足恒等式：``decrypt(crypt(input_string)) == base64(input_string)``。
    前端和 ragflow_cli 用此函数将密码加密后再传输到后端。

    Args:
        line: 待加密的明文字符串（如原始密码）。

    Returns:
        str —— 经过 RSA 加密 + Base64 编码的密文字符串。
    """
    file_path = os.path.join(get_project_base_directory(), "conf", "public.pem")
    rsa_key = RSA.importKey(Path(file_path).read_text(), "Welcome")
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)
    # 第一步：明文 → Base64（解密时的预期还原结果是 base64(原始明文)）
    password_base64 = base64.b64encode(line.encode('utf-8')).decode("utf-8")
    # 第二步：RSA 公钥加密
    encrypted_password = cipher.encrypt(password_base64.encode())
    # 第三步：密文 → Base64（便于字符串传输）
    return base64.b64encode(encrypted_password).decode('utf-8')


def decrypt(line):
    """使用 RSA 私钥解密由 ``crypt()`` 生成的密文字符串。

    解密流程：
    1. 读取 ``conf/private.pem`` 私钥文件（密码短语 "Welcome"）
    2. Base64 解码 → RSA 私钥解密（PKCS#1 v1.5）
    3. 返回 UTF-8 解码后的明文

    Args:
        line: 由 ``crypt()`` 生成的 Base64 + RSA 密文字符串。

    Returns:
        str —— 解密后的 Base64 编码字符串，即 ``base64(原始明文)``。

    Raises:
        ValueError: 解密失败时抛出，提示 "Fail to decrypt password!"。
    """
    file_path = os.path.join(get_project_base_directory(), "conf", "private.pem")
    rsa_key = RSA.importKey(Path(file_path).read_text(), "Welcome")
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)
    return cipher.decrypt(base64.b64decode(line), "Fail to decrypt password!").decode('utf-8')


def decrypt2(crypt_text):
    """兼容旧版 pycrypto 的 RSA 解密函数（处理历史密文格式）。

    与 ``decrypt()`` 的区别：
    - 使用旧版 ``Crypto`` 库（而非 ``Cryptodome``）
    - 额外处理 127 字节的密文（Java 端 RSA 加密有时会缺少前置 0x00 字节，
      导致密文长度少一位，需手动补 0x00）

    适用场景：解密由旧版本客户端或 Java 端生成的密文。

    Args:
        crypt_text: Base64 编码的 RSA 密文字符串。

    Returns:
        str —— 解密后的明文字符串。
    """
    from base64 import b64decode, b16decode
    from Crypto.Cipher import PKCS1_v1_5 as Cipher_PKCS1_v1_5
    from Crypto.PublicKey import RSA
    decode_data = b64decode(crypt_text)
    # 修复 Java 端 RSA 加密可能缺少前置 0x00 的问题（1024位密钥加密结果应为128字节）
    if len(decode_data) == 127:
        hex_fixed = '00' + decode_data.hex()
        decode_data = b16decode(hex_fixed.upper())

    file_path = os.path.join(get_project_base_directory(), "conf", "private.pem")
    pem = Path(file_path).read_text()
    rsa_key = RSA.importKey(pem, "Welcome")
    cipher = Cipher_PKCS1_v1_5.new(rsa_key)
    decrypt_text = cipher.decrypt(decode_data, None)
    return (b64decode(decrypt_text)).decode()


# 命令行入口 —— 用于手动加密密码并验证解密结果
if __name__ == "__main__":
    passwd = crypt(sys.argv[1])
    print(passwd)
    print(decrypt(passwd))
