"""共享 pytest 配置：把 code/ 加入 sys.path，让测试能 import shared.*。

学生可以从这个文件里看到一个最小的 pytest 配置示例。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
