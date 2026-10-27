# -*- coding: utf-8 -*-
"""验证 start_position 钩子：同一问题，不同位置偏移。"""
import json
import urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
SYS = ("You answer using only the provided records. Reply with the shortest exact answer.")
CTX = ("Record A: the access code is VIS-3355.\n"
       "Record B: the access code is MSK-9902.\n")


def post(rid):
    body = {"model": "q17b-flex",
            "messages": [{"role": "system", "content": SYS + "\n\n" + CTX},
                         {"role": "user", "content": "What is the access code in Record B?"}],
            "temperature": 0.0, "max_tokens": 20,
            "chat_template_kwargs": {"enable_thinking": False}, "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


for rid in ("scope=plain|x", "scope=off1@1000|x", "scope=off2@20000|x", "scope=plain|y"):
    try:
        print("%-22s -> %r" % (rid, post(rid)[:60]))
    except Exception as e:  # noqa: BLE001
        print("%-22s -> ERROR %s" % (rid, str(e)[:100]))
