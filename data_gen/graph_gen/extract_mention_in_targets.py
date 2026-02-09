#!/usr/bin/env python3
"""
从 gpickle 图中提取：
1）所有被 mention_in（图中存为 "mention in"）指向的节点；
2）所有孤立节点（入度+出度=0）。
分别导出为 JSON 文件。
使用方式: python extract_mention_in_targets.py [--input mcp_tragectory.gpickle] [--output ...] [--isolated ...]
"""
import pickle
import json
import argparse
from pathlib import Path

# 图中边的 relation 为 "mention in"（带空格）
REL_MENTION_IN = "mention in"

DEFAULT_INPUT = "graph_buffer/mcp_tragectory.gpickle"
DEFAULT_OUTPUT = "graph_buffer/mention_in_target_nodes.json"
DEFAULT_ISOLATED_OUTPUT = "graph_buffer/isolated_nodes.json"


def _node_to_row(g, nid):
    """将图节点转为与 Neo4j 导出一致的 dict：id, type, concept"""
    attrs = g.nodes[nid]
    row = {
        "id": attrs.get("id", nid),
        "type": attrs.get("type", ""),
        "concept": attrs.get("concept", []),
    }
    if isinstance(row["concept"], str):
        row["concept"] = [c.strip() for c in row["concept"].split(",") if c.strip()]
    return row


def main():
    parser = argparse.ArgumentParser(description="从 gpickle 提取 mention_in 目标节点与孤立节点")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="gpickle 图文件路径")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="mention_in 目标节点 JSON 路径")
    parser.add_argument("--isolated", default=DEFAULT_ISOLATED_OUTPUT, help="孤立节点 JSON 路径")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        return 1

    with open(input_path, "rb") as f:
        g = pickle.load(f)

    # 1. 所有作为 "mention in" 边目标的节点 id（去重）
    target_ids = set()
    for u, v, data in g.edges(data=True):
        if data.get("relation") == REL_MENTION_IN:
            target_ids.add(v)

    mention_in_rows = []
    for nid in sorted(target_ids):
        if nid not in g.nodes:
            continue
        mention_in_rows.append(_node_to_row(g, nid))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = script_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mention_in_rows, f, ensure_ascii=False, indent=2)
    print(f"已从 {input_path.name} 提取 {len(mention_in_rows)} 个被 mention_in 指向的节点 -> {out_path}")

    # 2. 孤立节点：入度=0 且 出度=0
    isolated_ids = [n for n in g.nodes if g.in_degree(n) == 0 and g.out_degree(n) == 0]
    isolated_rows = [_node_to_row(g, nid) for nid in sorted(isolated_ids)]

    iso_path = Path(args.isolated)
    if not iso_path.is_absolute():
        iso_path = script_dir / iso_path
    iso_path.parent.mkdir(parents=True, exist_ok=True)
    with open(iso_path, "w", encoding="utf-8") as f:
        json.dump(isolated_rows, f, ensure_ascii=False, indent=2)
    print(f"已从 {input_path.name} 提取 {len(isolated_rows)} 个孤立节点 -> {iso_path}")

    return 0


if __name__ == "__main__":
    exit(main())
