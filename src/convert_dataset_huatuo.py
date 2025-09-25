import json
import re
import sys
import argparse
import hashlib
from pathlib import Path


def strip_qa_prefix(s: str) -> str:
    # 去掉“问：/答：/Q:/A:”等前缀，顺便清空多余空白
    s = s.strip()
    s = re.sub(r'^(问[:：]\s*|Q[:：]\s*)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(答[:：]\s*|A[:：]\s*)', '', s, flags=re.IGNORECASE)
    return s.strip()


def normalize_text(s: str) -> str:
    # 轻量清洗：替换 Windows 换行、去掉不可见空白
    return s.replace('\r\n', '\n').replace('\r', '\n').strip()


def convert_line(obj):
    """
    支持两种情况：
    1) {"data": ["问：...","答：..."]}
    2) {"q": "...", "a": "..."}  (可选支持)
    """
    if "data" in obj and isinstance(obj["data"], list) and len(obj["data"]) >= 2:
        q_raw, a_raw = obj["data"][0], obj["data"][1]
    elif "q" in obj and "a" in obj:
        q_raw, a_raw = obj["q"], obj["a"]
    else:
        raise ValueError("无法识别该行格式，需要 data=[问,答] 或 q/a 字段")

    q = normalize_text(strip_qa_prefix(str(q_raw)))
    a = normalize_text(strip_qa_prefix(str(a_raw)))

    # 生成目标行
    return {
        "conversations": [
            {"from": "human", "value": q},
            {"from": "gpt",   "value": a}
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="原始问答 JSONL 文件（每行形如 {\"data\":[\"问：...\",\"答：...\"]}）")
    ap.add_argument("--output", required=True,
                    help="输出的 conversations JSONL 文件")
    ap.add_argument("--drop_short", type=int, default=0,
                    help="丢弃过短样本（问题或答案少于N字符则跳过），默认0不过滤")
    args = ap.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    n_in, n_ok, n_skip = 0, 0, 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
                ex = convert_line(obj)
                q = ex["conversations"][0]["value"]
                a = ex["conversations"][1]["value"]
                if args.drop_short and (len(q) < args.drop_short or len(a) < args.drop_short):
                    n_skip += 1
                    continue
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_skip += 1
                # 也可把错误行写到日志，这里简单略过
                continue

    print(f"输入 {n_in} 行，成功 {n_ok} 行，跳过 {n_skip} 行 -> {out_path}")


if __name__ == "__main__":
    main()
