# -*- coding: utf-8 -*-
import json
import sys
import argparse
import hashlib
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_human_text(inst: str, inp: str) -> str:
    inst = (inst or "").strip()
    inp = (inp or "").strip()
    if inst and inp:
        return f"{inst}\n{inp}"
    return inst or inp


def to_conv(obj: dict):
    # 兼容字段名大小写或别名
    inst = obj.get("instruction") or obj.get(
        "Instruction") or obj.get("query") or ""
    inp = obj.get("input") or obj.get("Input") or ""
    out = obj.get("output") or obj.get("Output") or obj.get("response") or ""

    human = build_human_text(str(inst), str(inp)).strip()
    gpt = str(out).strip()
    if not human or not gpt:
        return None

    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt",   "value": gpt}
        ]
    }


def make_key(sample: dict) -> str:
    conv = sample["conversations"]
    h = conv[0]["value"].strip()
    a = conv[1]["value"].strip()
    return hashlib.md5((h + "\n###\n" + a).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="多个 JSONL 输入文件")
    ap.add_argument("--output", required=True,
                    help="输出 JSONL（conversations 格式）")
    ap.add_argument("--drop_short", type=int, default=0, help="丢弃问或答长度小于该值的样本")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_in = n_ok = n_skip = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for p in args.inputs:
            path = Path(p)
            for obj in read_jsonl(path):
                n_in += 1
                conv = to_conv(obj)
                if conv is None:
                    n_skip += 1
                    continue
                q = conv["conversations"][0]["value"]
                a = conv["conversations"][1]["value"]
                if args.drop_short and (len(q) < args.drop_short or len(a) < args.drop_short):
                    n_skip += 1
                    continue
                key = make_key(conv)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n_ok += 1

    print(f"读入 {n_in} 行，写出 {n_ok} 行，跳过 {n_skip} 行 → {out_path}")


if __name__ == "__main__":
    main()
