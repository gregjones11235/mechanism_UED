"""Faithful copy of launch_d052_pure_dynamic_enhanced.py LLM client (api/extj/PROV).
Copied verbatim (not imported) so the launcher training main never executes.
Never prints keys. temperature=0, hard-fail after 48 tries, no silent fallback.
"""
import json, os, re, time, urllib.error, urllib.parse

PROV = {
    "ds": {"url": "https://api.deepseek.com/v1/chat/completions", "key": "DEEPSEEK_API_KEY"},
    "qw": {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "key": "DASHSCOPE_API_KEY"},
    "gl": {"url": "https://open.bigmodel.cn/api/paas/v4/chat/completions", "key": "ZHIPUAI_API_KEY"},
}
ROLE_MODEL_MAP = {"tutor": "qwen-flash-2025-07-28", "critic": "deepseek-v4-pro", "explorer": "glm-4-flash"}
MODELER_PROVIDER = "ds"
MODELER_MODEL = "deepseek-v4-pro"  # new role; strong integrative reasoner; independent of the 3 role models


def api(prov, model, msgs, mtok=512):
    import socket, http.client, ssl
    p = PROV[prov]
    key = os.environ[p["key"]]
    pl = json.dumps({"model": model, "messages": msgs, "max_tokens": mtok, "temperature": 0.0})
    itok = max(1, sum(len(m.get("content", "")) for m in msgs) // 3)
    MAXA = int(os.environ.get("D052_MAX_ATTEMPTS", "48"))  # default 48 = faithful; env override bounds reruns
    HTT = int(os.environ.get("D052_HTTP_TIMEOUT", "30"))   # default 30s = faithful; raise for large batched TTFB
    for a in range(MAXA):
        if a > 0:
            time.sleep(60 if (a % 16 == 0) else 1.5)
        try:
            parsed = urllib.parse.urlparse(p["url"])
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=HTT, context=ctx)
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
            conn.request("POST", parsed.path or "/", body=pl.encode(), headers=headers)
            resp = conn.getresponse()
            status = resp.status
            conn.sock.settimeout(300)
            body = resp.read().decode()
            conn.close()
            if status != 200:
                if status == 429 or 500 <= status < 600:
                    time.sleep(min(10 * (a + 1), 60) if status == 429 else min(5 * (a + 1), 60))
                    continue
                return {"ok": False, "err": "HTTP %d: %s" % (status, body[:200])}
            try:
                res = json.loads(body)
            except Exception:
                time.sleep(min(5 * (a + 1), 60))
                continue
            if "choices" not in res:
                time.sleep(min(10 * (a + 1), 60))
                continue
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, ssl.SSLError,
                ConnectionError, http.client.HTTPException, OSError):
            time.sleep(min(5 * (a + 1), 60))
            continue
        except Exception as e:
            if any(k in str(e).lower() for k in ["timeout", "ssl", "eof", "reset", "connection",
                                                  "incomplete", "peer", "broken pipe", "refused"]):
                time.sleep(min(5 * (a + 1), 60))
                continue
            return {"ok": False, "err": str(e)[:200]}
        try:
            msg = res["choices"][0]["message"]
            c = msg.get("content", "") or ""
            rm = res.get("model", "?")
            if isinstance(c, list):
                c = " ".join(str(x) for x in c)
            return {"ok": True, "content": c, "mrq": model, "mrt": rm, "itok": itok, "otok": max(1, len(c) // 3)}
        except Exception:
            if a < MAXA - 1:
                time.sleep(min(5 * (a + 1), 60))
                continue
            return {"ok": False, "err": "Bad format"}
    return {"ok": False, "err": "max retries"}


def extj(c):
    """Extract and parse JSON (object or array) from an LLM response."""
    if not c or not c.strip():
        return None
    c = c.strip()
    c = re.sub(r"```(?:json)?\s*", "", c)
    c = re.sub(r"```\s*$", "", c)
    s_obj = c.find("{")
    s_arr = c.find("[")
    if s_arr >= 0 and (s_obj < 0 or s_arr < s_obj):
        s = s_arr
        e = c.rfind("]")
    else:
        s = s_obj
        e = c.rfind("}")
    raw = c[s:e + 1] if s >= 0 and e > s else c
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    raw = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', raw)
    for _ in range(3):
        try:
            return json.loads(raw)
        except Exception:
            raw = re.sub(r"'([^']*)':", r'"\1":', raw)
            raw = re.sub(r":\s*'([^']*)'", r': "\1"', raw)
    return None


def call_json(prov, model, prompt, mtok, retries=3):
    """One logical call with up to `retries` JSON-repair attempts (escalating prompt).
    Returns (parsed, meta). parsed is None on hard failure (NO silent fallback)."""
    meta = {"provider": prov, "model_rq": model, "attempts": 0, "itok": 0, "otok": 0, "err": None}
    last = None
    for i in range(retries):
        meta["attempts"] = i + 1
        if i == 0:
            p = prompt
        else:
            p = prompt + ("\n\n[retry %d] Previous output was not valid JSON. "
                          "Return ONLY valid JSON, no prose, no markdown fences." % (i + 1))
        r = api(prov, model, [{"role": "user", "content": p}], mtok)
        meta["itok"] += r.get("itok", 0)
        meta["otok"] += r.get("otok", 0)
        meta["model_rt"] = r.get("mrt")
        if not r.get("ok"):
            meta["err"] = r.get("err")
            time.sleep(2)
            continue
        parsed = extj(r["content"])
        if parsed is not None:
            meta["err"] = None
            return parsed, meta
        last = r["content"]
        meta["err"] = "json_parse_failed"
    meta["last_raw_head"] = (last or "")[:300]
    return None, meta
