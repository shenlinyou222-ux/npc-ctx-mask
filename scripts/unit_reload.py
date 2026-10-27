# -*- coding: utf-8 -*-
"""单元测试：_npc_vis 的配置热重载。"""
import json
import os
import sys

p = "/tmp/npc_vis_unit.json"
json.dump({"live": [[0, 2]]}, open(p, "w"))
os.environ["NPC_VIS_BLOCKS"] = p

sys.path.insert(0, "/home/user/vllm-venv/lib/python3.12/site-packages")
from vllm import _npc_vis  # noqa: E402

import torch  # noqa: E402


def vis():
    return _npc_vis.bitmap(["scope=live|x"], 8, "cpu")[0].tolist()


a = vis()
print("deny [0,2) ->", a[:6])
assert a[:2] == [False, False] and a[2] is True or a[2] == True, "首次配置未生效"

json.dump({"live": []}, open(p, "w"))
import time
time.sleep(0.01)
b = vis()
print("放开后     ->", b[:6])
assert all(b), "热重载未生效"

json.dump({"live": [[4, 6]]}, open(p, "w"))
time.sleep(0.01)
c = vis()
print("换到 [4,6) ->", c[:8])
assert c[4] == False and c[5] == False and c[0] == True, "二次重载未生效"  # noqa: E712

print("\n热重载逻辑正常 ✔")
