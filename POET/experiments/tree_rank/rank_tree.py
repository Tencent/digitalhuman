#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poetry-Tree ranking evaluation (§4.2 / Tables 2–4).

Ranks poems along a degradation path against depth-order gold.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

try:
    import pandas as pd
    import openpyxl
except ImportError:
    pd = None
    openpyxl = None

from common.llm_client import call_LLM, format_token_cost_summary, get_token_cost_summary

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_leaf_paths(tree: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    从一棵诗词树中提取所有“叶子到根”的路径（内部实现为根->叶，再丢弃根节点）。
    
    返回值为若干条路径，每条路径是从根到叶的节点列表。
    在调用处会丢弃根节点，仅保留削弱后的诗词序列。
    """
    paths: List[List[Dict[str, Any]]] = []

    def dfs(node: Dict[str, Any], path: List[Dict[str, Any]]):
        new_path = path + [node]
        children = node.get("children") or []
        if not children:
            # 叶子节点，一条完整路径
            paths.append(new_path)
            return
        for child in children:
            if isinstance(child, dict):
                dfs(child, new_path)

    if isinstance(tree, dict):
        dfs(tree, [])
    return paths


def expand_poetry_tree_to_poems(
    tree_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将诗词树结构（final_results_xxx.json）展开为 step3 使用的平铺诗歌列表。
    
    每条从叶子到根的路径（不包含根节点）作为一个 round：
    - round_id: f\"{root_id}-{path_index}\"
    - 每个节点对应一首待评审的诗：使用节点中的 \"poem\" 文本
    - 使用节点上的 \"poet_id\" 作为真实ID，写入 entry[\"id\"]
    
    同时返回 tree_results，第二个返回值预留给后续可能的扩展（目前直接返回空列表，保持接口一致性）。
    """
    poem_entries: List[Dict[str, Any]] = []

    for root in tree_results:
        root_id = root.get("id")
        root_title = root.get("标题") or root.get("title") or f"ID_{root_id}"
        tree = root.get("诗词树") or root.get("tree")
        if not isinstance(tree, dict):
            continue

        all_paths = collect_leaf_paths(tree)
        # 为该根节点下的每条路径创建一个 round
        for path_index, path in enumerate(all_paths, 1):
            # path 是 [root, ..., leaf]，需要去掉根节点，仅保留削弱后的诗词
            if len(path) <= 1:
                continue
            non_root_nodes = path[1:]  # 丢弃根节点
            round_id = f"{root_id}-{path_index}"

            for pos, node in enumerate(non_root_nodes, 1):
                poem_text = node.get("poem", "").strip()
                poet_id = node.get("poet_id")  # 树节点中的分层 poet_id（如 2-1-1）
                if not poem_text or not poet_id:
                    continue
                entry: Dict[str, Any] = {
                    "id": str(poet_id),          # 真实 poet_id，后续排序结果会映射回这里
                    "root_id": root_id,          # 根节点对应的原始 ID（hongloumeng.json 中的 id）
                    "round": round_id,           # 每条路径作为一个 round
                    "position": pos,             # 在该路径中的顺序（1 开始）
                    "title": f"{root_title}",
                    "poem": poem_text,
                    "anonymous_name": "",        # 不暴露真实作者
                    "model": "",                 # 可选字段
                    # background 字段在后续用于生成统一背景文本
                    "background": {
                        "root_id": root_id,
                        "title": root_title,
                    },
                }
                poem_entries.append(entry)

    return poem_entries, []


def build_rounds_from_flat(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    针对已展开的平铺路径（含 poet_id、group_id、is_end 等），
    将每条叶子路径恢复为 round，返回带 round 字段的 poem_entries。
    """
    if not entries:
        return []

    # 确保每条记录都有 id 字段
    for e in entries:
        if "id" not in e or not e.get("id"):
            e["id"] = str(e.get("poet_id", ""))

    poem_entries: List[Dict[str, Any]] = []

    # Prefer an existing group_id on each record
    has_group_id = any(e.get("group_id") for e in entries)
    if has_group_id:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            gid = e.get("group_id")
            if not gid:
                continue
            groups.setdefault(gid, []).append(e)

        for gid, items in groups.items():
            # 仅保留包含叶子节点的组
            if not any(i.get("is_end") for i in items):
                continue
            ordered = sorted(items, key=lambda x: x.get("position", 0))
            for item in ordered:
                new_item = dict(item)
                new_item["round"] = gid
                poem_entries.append(new_item)
        return poem_entries

    # 如果没有 group_id，退化为依据 poet_id/父节点路径恢复
    id_map: Dict[str, Dict[str, Any]] = {str(e.get("id")): e for e in entries if e.get("id")}

    for leaf in entries:
        if not leaf.get("is_end"):
            continue
        pid = str(leaf.get("id", ""))
        if not pid:
            continue
        parts = pid.split("-")
        ancestor_ids = []
        # 生成 2段以上的层级（跳过根），直至叶子
        for i in range(2, len(parts) + 1):
            ancestor_ids.append("-".join(parts[:i]))

        seen = set()
        nodes: List[Dict[str, Any]] = []
        for aid in ancestor_ids:
            if aid in seen:
                continue
            seen.add(aid)
            node = id_map.get(aid)
            if node:
                nodes.append(node)

        if not nodes:
            continue

        ordered = sorted(
            nodes,
            key=lambda x: x.get(
                "position",
                len(str(x.get("id", "")).split("-")),
            ),
        )
        round_id = pid
        for item in ordered:
            new_item = dict(item)
            new_item["round"] = round_id
            poem_entries.append(new_item)

    return poem_entries


def extract_json_payload(response: str) -> dict:
    """
    从模型响应中提取 JSON，增强兼容性处理控制字符。
    使用与 step2 相同的容错机制。
    """
    response = response.replace('e ',' ') #针对Gemini的修改
    match = re.search(r"\{.*\}", response, flags=re.S)
    if not match:
        LOGGER.warning("未找到 JSON 块，原始响应：%s", response[:200])
        raise ValueError("模型未输出 JSON。")
    
    json_str = match.group(0)
    
    # 策略1：直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        LOGGER.debug("直接解析失败，尝试修复控制字符：%s", str(e)[:100])
    
    # 策略2：修复字符串值中的控制字符
    try:
        fixed_chars = []
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                fixed_chars.append(char)
                escape_next = False
            elif char == '\\':
                fixed_chars.append(char)
                escape_next = True
            elif char == '"':
                fixed_chars.append(char)
                in_string = not in_string
            elif in_string:
                if char == '\n':
                    fixed_chars.append('\\n')
                elif char == '\r':
                    fixed_chars.append('\\r')
                elif char == '\t':
                    fixed_chars.append('\\t')
                elif ord(char) < 32:
                    fixed_chars.append(f'\\u{ord(char):04x}')
                else:
                    fixed_chars.append(char)
            else:
                fixed_chars.append(char)
            
            i += 1
        
        fixed_json = ''.join(fixed_chars)
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        LOGGER.warning("修复后解析仍失败：%s", str(e)[:100])
        print(fixed_json)
        raise ValueError(f"JSON 解析失败：{e}")


def summarize_poem(entry: Dict[str, Any], virtual_id: str = None) -> str:
    """
    总结一首参赛作品的信息。
    
    Args:
        entry: 诗歌条目
        virtual_id: 虚拟ID（用于匿名化，避免暴露真实poet_id的规律）
    """
    title = entry.get("title") or "无题"
    anonymous_name = entry.get("anonymous_name", "")  # 诗会中的匿名名称
    poem_text = entry.get("poem", "").strip()
    notes = entry.get("notes", "")
    
    # 使用虚拟ID或真实ID
    display_id = virtual_id if virtual_id is not None else entry.get('id', '')
    
    summary = []
    
    #目前取消显示作者了，后面有需要可以加回去
    
    # # 如果有匿名名称，显示匿名名称；否则显示"匿名作者"
    # if anonymous_name:
    #     summary.append(f"作者:{anonymous_name}（匿名）")
    # else:
    #     summary.append("作者:匿名作者")
    
    summary.append(f"诗歌ID:{display_id}\n诗歌:\n{poem_text}")
    
    if notes:
        summary.append(f"作者自述:{notes}")
    
    return "\n".join(summary)


def summarize_gold(entry: Dict[str, Any], idx: int) -> str:
    """总结金标准参考信息"""
    title = entry.get("title") or entry.get("提取的诗名") or f"金标准{idx}"
    poem = entry.get("poet") or entry.get("内容", "")
    # 优先使用 poet_analyse（新格式），兼容旧格式
    analyse = entry.get("poet_analyse") or entry.get("analyse") or entry.get("评解", "")
    return (
        f"参考ID:{idx}\n题目:{title}\n原诗:\n{poem}\n评点:{analyse}\n"
    )


def build_prompt_for_single_poem(
    poem_entry: Dict[str, Any],
    gold_samples: List[Dict[str, Any]],
    background_text: str,
    gold_samples_count: int = 4,
    use_gold: bool = True,
    virtual_id: str = None,
    root_gold_entry: Dict[str, Any] = None,
) -> List[dict]:
    """
    为单首诗构建专家评审 prompt。
    
    Args:
        poem_entry: 待评审的诗作
        gold_samples: 金标准参考样本列表
        background_text: 统一背景
        gold_samples_count: 使用的样本数量
        use_gold: 是否使用金标准参考
        virtual_id: 虚拟ID（用于匿名化）
        root_gold_entry: 根节点的原诗和专家评估（从 hongloumeng.json 获取）
    """
    poem_summary = summarize_poem(poem_entry, virtual_id=virtual_id)
    
    if use_gold:
        # 构建专家样本
        gold_section_parts = []
        
        # 第一个：根节点原诗（背景诗）+ 真实解读
        if root_gold_entry:
            gold_section_parts.append(f"【参考样本1】（背景诗）\n{summarize_gold(root_gold_entry, 1)}")
        
        # 另外 N 个随机样本
        start_idx = 2 if root_gold_entry else 1
        for idx, sample in enumerate(gold_samples[:gold_samples_count], start_idx):
            gold_section_parts.append(f"【参考样本{idx}】\n{summarize_gold(sample, idx)}")
        
        gold_section = "\n\n".join(gold_section_parts)
        num_samples = len(gold_section_parts)
        
        user_text = f'''你是金标准参考样本中那位专家。请先深入研读以下 {num_samples} 个金标准评点案例，全面学习并内化专家的评审风格、评审结构、评审思路、语言表达、专业术语、分析角度、论述逻辑等所有特征，然后以完全相同的专家身份和口吻去评审下面这首参赛诗词。

**核心要求：**
1. **身份认同**：你就是那位专家本人，不是模仿者，而是专家本身。你的每一句话都应该让人以为是专家亲笔所写。
2. **全面学习**：仔细分析参考样本中的以下方面：
   - 语言风格：用词习惯、句式特点、修辞手法、专业术语的使用
   - 评审结构：如何开头、如何展开、如何收尾、段落组织方式
   - 分析角度：从哪些维度切入（思想内容、艺术手法、格律技巧、意象营造、情感表达等）
   - 论述逻辑：论证思路、观点展开方式、例证引用方式
   - 专业表述：古典诗词评论的专业术语、典故引用、比较分析等
3. **评审原则**：
   - 参考样本1（背景诗）的评点风格是核心模板，必须严格遵循其风格特征
   - 其余样本用于补充理解专家的语气变化、结构多样性、思路灵活性
   - 不要机械罗列维度，要像专家那样自然流畅地组织论述
   - 使用与专家一致的专业术语和表达方式
   - 保持与专家相同的学术深度和鉴赏水准
4. **输出要求**：必须输出有效的 JSON 格式，仅包含 review 字段。

输出格式：
{{
  "review": "以专家身份和风格撰写的专业评述，让人一读即知是专家手笔"
}}

【统一背景】
{background_text or '无背景说明，仅按题意与诗作内容评审'}

【待评审作品】
{poem_summary}

【金标准参考样本】（共{num_samples}个，请深入研读，全面学习专家的所有评审特征）
{gold_section}'''
        
        system_content = "你是金标准参考样本中那位古典诗词评论专家的数字人化身。你的任务是全面学习并内化专家的评审风格、语言表达、分析角度、论述逻辑、专业术语等所有特征，然后以专家的身份和口吻对参赛作品进行专业评析。你的评述应该让人一读即知是专家亲笔所写，而非模仿之作。必须输出有效的 JSON 格式，只包含 review 字段。"
    else:
        # 不使用金标准，直接评审
        user_text = f'''请作为古典诗词评论专家，对以下参赛作品进行专业评析。

**评审要求：**
1. 从思想内容、艺术手法、格律技巧、意象营造、情感表达等多个维度进行深入分析
2. 使用专业的古典诗词评论术语和表达方式
3. 评述要自然流畅，不要机械罗列维度
4. 必须输出有效的 JSON 格式，仅包含 review 字段

输出格式：
{{
  "review": "专业评述"
}}

【统一背景】
{background_text or '无背景说明，仅按题意与诗作内容评审'}

【待评审作品】
{poem_summary}'''
        
        system_content = "你是古典诗词评论专家，需要对参赛作品进行专业评析。必须输出有效的 JSON 格式，只包含 review 字段。"
    
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": user_text},
    ]


def build_ranking_prompt(poems_with_reviews: List[Dict[str, Any]]) -> tuple[List[dict], Dict[str, str]]:
    """
    构建最终排序 prompt，基于诗和点评进行排序。
    
    Returns:
        (prompt, id_mapping): prompt 列表和 "作品{idx}" -> 实际 id 的映射
    """
    items_text = []
    id_mapping: Dict[str, str] = {}  # {"作品1": "1-1", "作品2": "1-2", ...}
    random.shuffle(poems_with_reviews)
    
    for idx, item in enumerate(poems_with_reviews, 1):
        actual_id = item.get("id", "")
        work_id = f"作品{idx}"
        id_mapping[work_id] = actual_id
        
        title = item.get("title", "无题")
        poem = item.get("poem", "")
        review = item.get("review", "")
        author = item.get("anonymous_name", "") or item.get("author", "匿名作者")
        notes = item.get("notes", "")
        item_text = (
            f"作品{idx}：\n"
            f"标题：{title}\n"
            f"作者：{author}\n"
            f"诗歌：\n{poem}\n"
        )
        if notes:
            item_text += f"作者自述：{notes}\n"
        item_text += f"专家点评：\n{review}\n"
        items_text.append(item_text)
    
    items_section = "\n\n".join(items_text)
    user_text = f'''请作为诗评大会主审，基于以下各首诗及其专家点评，对这组诗词进行专业排名。

**要求：**
1. 综合考虑诗歌的艺术水准、思想内涵和专家点评的质量。
2. 给出专业排名（从1开始，1为最佳）。
3. 可以简要说明排名的理由。

输出格式（必须是有效的 JSON）：
{{
  "ranking": [
    {{
      "id": "作品ID（如作品1）",
      "rank": 排名（从1开始，1为最佳）
    }},
    ...
  ],
  "overall": "对这组诗词的总体观察和比较分析"
}}

【参赛作品及点评】（共 {len(poems_with_reviews)} 首）

{items_section}'''
    
    prompt = [
        {
            "role": "system",
            "content": "你是诗评大会主审，需要基于诗歌和专家点评进行专业排名。必须输出有效的 JSON 格式。",
        },
        {"role": "user", "content": user_text},
    ]
    return prompt, id_mapping


def build_batch_prompt(
    poems: List[Dict[str, Any]],
    gold_samples: List[Dict[str, Any]],
    background_text: str,
    gold_samples_count: int = 4,
    use_gold: bool = True,
    root_gold_entry: Dict[str, Any] = None,
) -> Tuple[List[dict], Dict[str, str]]:
    """
    构建"一次性评论+排序"模式的 prompt：
    - 输入一整个 round 的诗作列表
    - 输出每首诗的 review 以及整体 ranking 和 overall
    - 返回 (prompt, virtual_to_real_id_mapping): 虚拟ID到真实ID的映射
    
    注意：诗歌顺序会被打乱，并分配虚拟编号（1、2、3...）以避免暴露真实poet_id的规律
    
    Args:
        root_gold_entry: 根节点的原诗和专家评估（从 hongloumeng.json 获取）
    """
    # 打乱诗歌顺序并分配虚拟编号
    shuffled_poems = poems.copy()
    random.shuffle(shuffled_poems)
    
    virtual_to_real: Dict[str, str] = {}  # {"1": "116-1-1-3", "2": "116-1-2", ...}
    poems_text = []
    for idx, item in enumerate(shuffled_poems, 1):
        virtual_id = str(idx)
        real_id = item.get('id', '')
        virtual_to_real[virtual_id] = real_id
        
        title = item.get("title", "无题")
        poem_info = f"ID:{virtual_id}\n标题:{title}\n诗歌:\n{item['poem']}\n"
        notes = item.get("notes", "")
        if notes:
            poem_info += f"作者自述：{notes}\n"
        poems_text.append(poem_info)
    poems_section = "\n\n".join(poems_text)

    if use_gold:
        gold_section_parts = []
        
        # 第一个：根节点原诗（背景诗）+ 真实解读
        if root_gold_entry:
            gold_section_parts.append(f"【参考样本1】（背景诗）\n{summarize_gold(root_gold_entry, 1)}")
        
        # 另外 N 个随机样本
        start_idx = 2 if root_gold_entry else 1
        for idx, sample in enumerate(gold_samples[:gold_samples_count], start_idx): 
            gold_section_parts.append(f"【参考样本{idx}】\n{summarize_gold(sample, idx)}")
        gold_section = "\n\n".join(gold_section_parts)
        num_samples = len(gold_section_parts)
        
        user_text = f'''你是金标准参考样本中那位专家。请先深入研读以下 {num_samples} 个金标准评点案例，全面学习并内化专家的评审风格、评审结构、评审思路、语言表达、专业术语、分析角度、论述逻辑等所有特征，然后以完全相同的专家身份和口吻对下列所有作品一次性给出"专家点评"和"整体排名"。

**核心要求：**
1. **身份认同**：你就是那位专家本人，不是模仿者，而是专家本身。你对**每一首诗**的鉴赏都应该让人以为是专家亲笔所写。
2. **全面学习**：仔细分析参考样本中的以下方面：
   - 语言风格：用词习惯、句式特点、修辞手法、专业术语的使用
   - 评审结构：如何开头、如何展开、如何收尾、段落组织方式
   - 分析角度：从哪些维度切入（思想内容、艺术手法、格律技巧、意象营造、情感表达等）
   - 论述逻辑：论证思路、观点展开方式、例证引用方式
   - 专业表述：古典诗词评论的专业术语、典故引用、比较分析等
3. **评审原则**：
   - 参考样本1（背景诗）的评点风格是核心模板，必须严格遵循其风格特征
   - 其余样本用于补充理解专家的语气变化、结构多样性、思路灵活性
   - 不要机械罗列维度，要像专家那样自然流畅地组织论述
   - 使用与专家一致的专业术语和表达方式
   - 保持与专家相同的学术深度和鉴赏水准
4. **输出要求**：必须输出有效的 JSON 格式，字段：reviews（数组）、ranking（数组）、overall。

输出格式示例：
{{
  "reviews": [
    {{"id": "x-x", "review": "以专家身份和风格撰写的专业评述，让人一读即知是专家手笔" }},
    ...
  ],
  "ranking": [
    {{"id": "x-x", "rank": 1}},
    ...
  ],
  "overall": "总体观察"
}}

【统一背景】
{background_text or '无背景说明，仅按题意与诗作内容评审'}

【待评审作品列表】（请按上面的 ID 输出对应 review 和 rank）
{poems_section}

【参考样本】（请模仿其评点风格）
{gold_section}'''

        system_content = "你是金标准参考样本中那位古典诗词评论专家。你的任务是全面学习并内化专家的评审风格、语言表达、分析角度、论述逻辑、专业术语等所有特征，然后以专家的身份和口吻一次性给出多首诗的点评与排名。你的评述应该让人一读即知是专家亲笔所写。必须输出有效的 JSON 格式。"
    else:
        user_text = f'''请作为古典诗词评论专家，对以下所有作品一次性给出"专家点评"和"整体排名"。

**评审要求：**
1. 从思想内容、艺术手法、格律技巧、意象营造、情感表达等多个维度进行深入分析
2. 使用专业的古典诗词评论术语和表达方式
3. 评述要自然流畅，不要机械罗列维度
4. 必须输出有效的 JSON 格式，可以直接被代码读取，字段：reviews（数组）、ranking（数组）、overall

输出格式示例：
{{
  "reviews": [
    {{"id": "x-x", "review": "专业评述" }},
    ...
  ],
  "ranking": [
    {{"id": "x-x", "rank": 1}},
    ...
  ],
  "overall": "总体观察"
}}

【统一背景】
{background_text or '无背景说明，仅按题意与诗作内容评审'}

【待评审作品列表】（请按上面的 ID 输出对应 review 和 rank）
{poems_section}'''

        system_content = "你是古典诗词评论专家，需要对多首参赛作品进行专业评析和排名。必须输出有效的 JSON 格式。"
    
    prompt = [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": user_text},
    ]
    return prompt, virtual_to_real


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def calculate_spearman_correlation(rank1: List[int], rank2: List[int]) -> float:
    """计算斯皮尔曼相关系数"""
    if len(rank1) != len(rank2) or len(rank1) == 0:
        return 0.0
    
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(rank1, rank2)
        return float(corr) if not (corr != corr) else 0.0  # 处理NaN
    except ImportError:
        # 如果没有scipy，使用简化计算
        n = len(rank1)
        d_squared = sum((r1 - r2) ** 2 for r1, r2 in zip(rank1, rank2))
        return 1.0 - (6 * d_squared) / (n * (n * n - 1)) if n > 1 else 0.0


def calculate_accuracy(gold_ranks: List[int], pred_ranks: List[int]) -> float:
    """计算准确率（完全匹配的比例）"""
    if len(gold_ranks) != len(pred_ranks) or len(gold_ranks) == 0:
        return 0.0
    correct = sum(1 for g, p in zip(gold_ranks, pred_ranks) if g == p)
    return correct / len(gold_ranks)


def generate_excel_and_statistics(final_results: List[Dict[str, Any]], output_path: Path, rank_mode: str):
    """生成Excel文件和统计报告"""
    if pd is None or openpyxl is None:
        LOGGER.warning("pandas 或 openpyxl 未安装，跳过Excel生成")
        return
    
    # 准备数据
    all_poems = []  # 所有诗的详细信息
    depth_stats = defaultdict(list)  # {depth: [rank1, rank2, ...]}
    
    for round_result in final_results:
        reviews = round_result.get("reviews", [])
        if not reviews:
            continue
        
        # 按position排序（position越小=越深，position越大=越浅）
        sorted_reviews = sorted(reviews, key=lambda x: x.get("position", 0))
        
        # 提取真实排名（position）和模型排名（rank）
        gold_ranks = [r.get("position", 0) for r in sorted_reviews]
        # 确保rank是整数类型
        model_ranks = []
        for r in sorted_reviews:
            rank = r.get("rank", 0)
            # 如果rank是字符串，尝试转换为整数
            if isinstance(rank, str):
                try:
                    rank = int(rank)
                except (ValueError, TypeError):
                    rank = 0
            elif not isinstance(rank, (int, float)):
                rank = 0
            model_ranks.append(int(rank))
        
        # 计算该组的准确率和斯皮尔曼系数
        accuracy = calculate_accuracy(gold_ranks, model_ranks)
        spearman = calculate_spearman_correlation(gold_ranks, model_ranks)
        
        # 记录每首诗的详细信息
        for r in sorted_reviews:
            depth = r.get("position", 0)
            rank = r.get("rank", 0)
            # 确保rank是整数类型
            if isinstance(rank, str):
                try:
                    rank = int(rank)
                except (ValueError, TypeError):
                    rank = 0
            elif not isinstance(rank, (int, float)):
                rank = 0
            rank = int(rank)
            
            all_poems.append({
                "round": round_result.get("round", ""),
                "depth": depth,  # position越小=越深
                "gold_rank": depth,  # 真实排名（position）
                "model_rank": rank,
                "poet_id": r.get("id", ""),
                "title": r.get("title", ""),
                "poem": r.get("poem", ""),
                "review": r.get("review", ""),
                "accuracy": accuracy,
                "spearman": spearman,
            })
            
            # 统计每个深度的排名
            depth_stats[depth].append(rank)
    
    if not all_poems:
        LOGGER.warning("没有数据可生成Excel")
        return
    
    # Sheet1: 从浅到深的平均排名
    depth_avg_ranks = []
    for depth in sorted(depth_stats.keys(), reverse=True):  # 从浅到深（depth越大=越浅）
        ranks = depth_stats[depth]
        # 确保所有rank都是数字类型
        numeric_ranks = []
        for rank in ranks:
            if isinstance(rank, str):
                try:
                    numeric_ranks.append(int(rank))
                except (ValueError, TypeError):
                    continue
            elif isinstance(rank, (int, float)):
                numeric_ranks.append(int(rank))
        
        depth_avg_ranks.append({
            "depth": depth,
            "avg_model_rank": sum(numeric_ranks) / len(numeric_ranks) if numeric_ranks else 0,
            "count": len(numeric_ranks),
        })
    
    df_depth = pd.DataFrame(depth_avg_ranks)
    
    # Sheet2: 详细的每一组数据
    df_detail = pd.DataFrame(all_poems)
    
    # 写入Excel
    excel_path = output_path.with_suffix(f'.{rank_mode}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_depth.to_excel(writer, sheet_name='深度平均排名', index=False)
        df_detail.to_excel(writer, sheet_name='详细数据', index=False)
    
    LOGGER.info("Excel文件已生成：%s", excel_path)
    
    # 计算并输出统计指标
    # 1. 从浅到深的平均排名
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("统计报告")
    LOGGER.info("=" * 60)
    LOGGER.info("\n从浅到深的平均排名（depth越大=越浅）：")
    for row in depth_avg_ranks:
        LOGGER.info("  Depth %d: 平均排名 %.2f (样本数: %d)", row["depth"], row["avg_model_rank"], row["count"])
    
    # 2. 每一组诗的准确率和斯皮尔曼系数的均值
    round_accuracies = []
    round_spearmans = []
    for round_result in final_results:
        reviews = round_result.get("reviews", [])
        if not reviews:
            continue
        sorted_reviews = sorted(reviews, key=lambda x: x.get("position", 0))
        gold_ranks = [r.get("position", 0) for r in sorted_reviews]
        # 确保rank是整数类型
        model_ranks = []
        for r in sorted_reviews:
            rank = r.get("rank", 0)
            if isinstance(rank, str):
                try:
                    rank = int(rank)
                except (ValueError, TypeError):
                    rank = 0
            elif not isinstance(rank, (int, float)):
                rank = 0
            model_ranks.append(int(rank))
        accuracy = calculate_accuracy(gold_ranks, model_ranks)
        spearman = calculate_spearman_correlation(gold_ranks, model_ranks)
        round_accuracies.append(accuracy)
        round_spearmans.append(spearman)
    
    if round_accuracies:
        avg_accuracy = sum(round_accuracies) / len(round_accuracies)
        avg_spearman = sum(round_spearmans) / len(round_spearmans)
        LOGGER.info("\n每一组诗的统计（共 %d 组）：", len(round_accuracies))
        LOGGER.info("  平均准确率: %.4f", avg_accuracy)
        LOGGER.info("  平均斯皮尔曼系数: %.4f", avg_spearman)
    
    # 3. 所有诗的平均rank的准确率和斯皮尔曼系数
    all_gold_ranks = [p["gold_rank"] for p in all_poems]
    # 确保model_rank是整数类型
    all_model_ranks = []
    for p in all_poems:
        rank = p.get("model_rank", 0)
        if isinstance(rank, str):
            try:
                rank = int(rank)
            except (ValueError, TypeError):
                rank = 0
        elif not isinstance(rank, (int, float)):
            rank = 0
        all_model_ranks.append(int(rank))
    overall_accuracy = calculate_accuracy(all_gold_ranks, all_model_ranks)
    overall_spearman = calculate_spearman_correlation(all_gold_ranks, all_model_ranks)
    
    LOGGER.info("\n所有诗的统计（共 %d 首）：", len(all_poems))
    LOGGER.info("  总体准确率: %.4f", overall_accuracy)
    LOGGER.info("  总体斯皮尔曼系数: %.4f", overall_spearman)
    LOGGER.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Rank Poetry-Tree path groups against depth-order gold (§4.2).")
    parser.add_argument("--input", "--poems", dest="poems", required=True, help="Poetry-tree JSON (list of roots with 诗词树 or tree)")
    parser.add_argument(
        "--gold",
        nargs="+",
        default=None,
        help="金标准数据（可指定多个 JSON，如 tangshixiaozha.json hongloumeng.json）",
    )
    parser.add_argument("--output", default=None, help="输出路径（默认自动生成，包含rank-mode）")
    parser.add_argument("--model", default="gpt-4o", help="API model id (sent as-is)")
    parser.add_argument(
        "--rank-mode",
        choices=["separate", "batch"],
        default="batch",
        help="batch: rank the whole path at once (paper); separate: review one poem at a time",
    )
    parser.add_argument(
        "--gold-samples",
        type=int,
        default=4,
        help="Number of extra expert demonstrations besides the background poem (paper n=5 = 1 background + 4 random)",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发线程数（用于并发处理不同 round，默认 1）")
    parser.add_argument("--use-gold", action="store_true", default=False, help="是否使用金标准参考数据")
    parser.add_argument("--use-root-gold", action="store_true", default=False, help="是否使用根节点的原诗作为金标准参考（需要从 hongloumeng.json 中获取）")
    args = parser.parse_args()
    
    # 如果指定了--use-gold但没有提供--gold，报错
    if args.use_gold and not args.gold:
        parser.error("--use-gold 需要同时提供 --gold 参数")
    
    # 只有明确传入 --use-gold 时才使用金标准
    use_gold = args.use_gold

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    poem_path = Path(args.poems).expanduser().resolve()
    raw_data = load_json(poem_path)
    if not isinstance(raw_data, list):
        raise ValueError("poems 文件必须是数组")

    # 如果检测到是诗词树结果（包含"诗词树"字段），先展开为平铺的诗歌列表
    if raw_data and isinstance(raw_data[0], dict) and ("诗词树" in raw_data[0] or "tree" in raw_data[0]):
        LOGGER.info("检测到诗词树结构输入，将按叶子到根路径展开为 round")
        poem_entries, _ = expand_poetry_tree_to_poems(raw_data)
    # 如果是已展开的平铺路径（带 is_end/group_id），恢复 round
    elif raw_data and isinstance(raw_data[0], dict) and ("is_end" in raw_data[0] or "group_id" in raw_data[0]):
        LOGGER.info("检测到平铺的路径数据（test/train），按叶子节点反推父节点组成 round")
        poem_entries = build_rounds_from_flat(raw_data)
    else:
        poem_entries = raw_data

    # 加载《红楼梦》源数据，用于根据 root_id 提取统一背景（background + poet_analyse）
    root_source_meta: Dict[str, Dict[str, Any]] = {}
    try:
        # 默认认为 hongloumeng.json 与本脚本在同一工程根目录下
        project_root = Path(__file__).resolve().parent
        hlm_path = project_root / "hongloumeng.json"
        if hlm_path.exists():
            hlm_records = load_json(hlm_path)
            if isinstance(hlm_records, list):
                for rec in hlm_records:
                    if isinstance(rec, dict) and "id" in rec:
                        root_id_str = str(rec.get("id"))
                        root_source_meta[root_id_str] = rec
            LOGGER.info("已从 hongloumeng.json 加载 %d 条源诗信息用于背景匹配", len(root_source_meta))
            if args.use_root_gold:
                LOGGER.info("已启用根节点原诗作为金标准参考功能")
        else:
            LOGGER.warning("未找到 hongloumeng.json（期望路径：%s），将回退使用 poems 中的 background 字段", hlm_path)
            if args.use_root_gold:
                LOGGER.warning("--use-root-gold 已启用，但未找到 hongloumeng.json，将无法使用根节点原诗作为金标准参考")
    except Exception as e:
        LOGGER.warning("加载 hongloumeng.json 失败，将回退使用 poems 中的 background 字段：%s", e)
        if args.use_root_gold:
            LOGGER.warning("--use-root-gold 已启用，但加载 hongloumeng.json 失败，将无法使用根节点原诗作为金标准参考")

    def build_background_text_from_root(root_id: Any, poem_background: Any) -> str:
        """
        根据 root_id 从 hongloumeng.json 中提取背景信息（background + poet_analyse），
        若获取失败则回退使用 poem 中自带的 background 字段。
        """
        text_parts: List[str] = []
        if root_id is not None:
            rec = root_source_meta.get(str(root_id))
            if isinstance(rec, dict):
                bg = rec.get("background") or ""
                if isinstance(bg, str) and bg.strip():
                    text_parts.append(bg.strip())

        # 回退：使用原有 poem.background
        if not text_parts and poem_background:
            if isinstance(poem_background, dict):
                bg = poem_background.get("background") or poem_background.get("topic") or ""
                if isinstance(bg, str) and bg.strip():
                    text_parts.append(bg.strip())
            else:
                text_parts.append(str(poem_background))
        return "\n\n".join(text_parts)

    # 加载全部金标准数据（如果使用金标准）
    all_gold_entries: List[Dict[str, Any]] = []
    if use_gold and args.gold:
        for path_str in args.gold:
            records = load_json(Path(path_str).expanduser().resolve())
            if isinstance(records, list):
                all_gold_entries.extend(records)
        LOGGER.info("加载了 %d 首金标准参考作品", len(all_gold_entries))
    else:
        LOGGER.info("未使用金标准参考数据")
    
    # 按 round 分组处理
    from collections import defaultdict
    poems_by_round = defaultdict(list)
    for entry in poem_entries:
        round_id = entry.get("round", "unknown")
        poems_by_round[round_id].append(entry)
    
    LOGGER.info("共发现 %d 个 round，将分别进行评审", len(poems_by_round))
    
    # 确定输出路径（如果未指定，自动生成包含rank-mode的文件名）
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # 自动生成文件名：poetry_ranking_{rank_mode}.json
        base_path = Path(args.poems).expanduser().resolve()
        output_path = base_path.parent / f"poetry_ranking_{args.rank_mode}_{use_gold}_{args.model}_{args.use_root_gold}.json"
    
    # 读取已存在的输出，用于断点续跑和评审续作
    results_map: Dict[str, Dict[str, Any]] = {}
    # 记录每个 round 已评审的模型 ID（通过 id 字段识别，如 "1-1", "1-2"）
    existing_reviewed_ids: Dict[str, set] = {}  # {round_id: set of reviewed_ids}
    # 全局评审缓存：id -> review 结果（便于多个组复用同一节点的评审）
    review_cache: Dict[str, Dict[str, Any]] = {}
    
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    rid = item.get("round")
                    if rid:
                        results_map[rid] = item
                        # 记录已评审的 ID
                        if rid not in existing_reviewed_ids:
                            existing_reviewed_ids[rid] = set()
                        reviews = item.get("reviews", [])
                        for review in reviews:
                            review_id = review.get("id", "")
                            if review_id:
                                existing_reviewed_ids[rid].add(review_id)
                                # 缓存评审结果，供其它 round 复用
                                review_cache[review_id] = review
                
                LOGGER.info("检测到已有结果文件，已加载 %d 个 round", len(results_map))
                for rid, reviewed_ids in existing_reviewed_ids.items():
                    LOGGER.info("  Round %s: 已评审 %d 个作品", rid, len(reviewed_ids))
        except Exception as e:
            LOGGER.warning("读取已有结果失败，忽略继续：%s", e)

    existing_rounds = set(results_map.keys())
    workers = max(1, args.workers)
    lock = threading.Lock()

    def write_output():
        # 将 results_map 转成按 round 排序的列表写回磁盘
        sorted_results = sorted(results_map.values(), key=lambda x: x.get("round", ""))
        output_path.write_text(json.dumps(sorted_results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def process_one_poem(
        poem_entry: Dict[str, Any],
        round_id: str,
        gold_samples: List[Dict[str, Any]],
        background_text: str,
        gold_samples_count: int = 4,
        use_gold: bool = True,
        virtual_id: str = None,
        root_gold_entry: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """处理单首诗，返回带review的结果
        
        Args:
            virtual_id: 虚拟ID（用于匿名化，避免暴露真实poet_id的规律）
            root_gold_entry: 根节点的原诗和专家评估（从 hongloumeng.json 获取）
        """
        # 使用原始的 poet_id，不要重新生成
        poem_id = poem_entry.get("id", "")
        if poem_id and poem_id in review_cache:
            return review_cache[poem_id]
        title = poem_entry.get("title", "无题")
        
        prompt = build_prompt_for_single_poem(poem_entry, gold_samples, background_text, gold_samples_count, use_gold, virtual_id=virtual_id, root_gold_entry=root_gold_entry)
        
        last_error = None
        for attempt in range(1, 11):
            try:
                response, _ = call_LLM(prompt, model_name=args.model)
                payload = extract_json_payload(response)
                
                if "review" not in payload:
                    raise ValueError("输出 JSON 缺少 'review' 字段")
                
                result = {
                    "id": poem_id,
                    "source_id": poem_entry.get("source_id", ""),
                    "round": round_id,
                    "position": poem_entry.get("position", 0),
                    "title": title,
                    "poem": poem_entry.get("poem", ""),
                    "author": poem_entry.get("anonymous_name", "") or "匿名作者",
                    "model": poem_entry.get("model", ""),
                    "review": payload.get("review", ""),
                }
                
                LOGGER.info("✓ %s 评审完成", poem_id)
                if poem_id:
                    review_cache[poem_id] = result
                return result
            
            except Exception as e:
                last_error = e
                LOGGER.warning("%s 第 %d 次解析失败：%s", poem_id, attempt, e)
                if attempt >= 10:
                    break
                continue
        
        LOGGER.error("✗ %s 评审失败：%s", poem_id, last_error)
        failed = {
            "id": poem_id,
            "round": round_id,
            "position": poem_entry.get("position", 0),
            "title": title,
            "poem": poem_entry.get("poem", ""),
            "author": poem_entry.get("anonymous_name", "") or "匿名作者",
            "model": poem_entry.get("model", ""),
            "review": f"评审失败：{last_error}",
            "error": str(last_error) if last_error else "未知错误",
        }
        if poem_id:
            review_cache[poem_id] = failed
        return failed
    
    def evaluate_missing_poems(poems: List[Dict[str, Any]], use_gold: bool = True):
        """
        先对所有待评审的诗进行专家评估（去重，便于在多个组中复用），
        仅对缺少评审缓存的诗调用模型。
        
        在 separate 模式下，这一步也使用线程池并发执行，以提高整体速度。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pending = [p for p in poems if p.get("id") and p.get("id") not in review_cache]
        if not pending:
            LOGGER.info("所有诗均已存在评审缓存，跳过单诗评估")
            return

        LOGGER.info("开始单诗评估（去重后 %d 首，使用并发 %d）", len(pending), workers)

        def build_task_args(poem_entry: Dict[str, Any]):
            """为每首诗构建调用 process_one_poem 所需的参数元组。"""
            round_id = poem_entry.get("round", "")
            # 使用随机虚拟ID（1-1000之间的随机数），避免暴露真实ID的层级规律
            virtual_id = str(random.randint(1, 1000))

            # 背景信息：根据 root_id 从 hongloumeng.json 提取
            root_id_single = poem_entry.get("root_id")
            poem_background = poem_entry.get("background") if poem_entry else {}
            background_text = build_background_text_from_root(root_id_single, poem_background)

            # 获取根节点原诗和专家评估（如果启用）
            root_gold_entry = None
            if args.use_root_gold and root_id_single is not None:
                root_gold_entry = root_source_meta.get(str(root_id_single))
                if not isinstance(root_gold_entry, dict):
                    root_gold_entry = None

            # 金标准样本（按 root_id 过滤）
            gold_samples: List[Dict[str, Any]] = []
            if use_gold:
                root_id = poem_entry.get("root_id")
                candidates = [
                    g for g in all_gold_entries
                    if isinstance(g, dict) and g.get("id") is not None and g.get("id") != root_id
                ]
                if not candidates:
                    candidates = [g for g in all_gold_entries if isinstance(g, dict)]
                gold_samples_count = max(1, args.gold_samples)
                gold_samples = random.sample(candidates, min(gold_samples_count, len(candidates))) if candidates else []
            else:
                gold_samples_count = max(1, args.gold_samples)

            return (poem_entry, round_id, gold_samples, background_text, gold_samples_count, use_gold, virtual_id, root_gold_entry)

        # 并发执行单诗评估
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_poem = {
                executor.submit(process_one_poem, *build_task_args(poem_entry)): poem_entry
                for poem_entry in pending
            }
            for future in as_completed(future_to_poem):
                poem_entry = future_to_poem[future]
                poem_id = poem_entry.get("id", "")
                try:
                    _ = future.result()
                except Exception as e:
                    LOGGER.error("单诗评估异常（id=%s）：%s", poem_id, e)

        LOGGER.info("单诗评估完成，缓存评审数：%d", len(review_cache))

    def process_one_round(round_id: str, poems: List[Dict[str, Any]], existing_result: Dict[str, Any] = None, use_gold: bool = True) -> Dict[str, Any]:
        """处理单个 round 的所有作品
        
        Args:
            round_id: round ID
            poems: 所有待评审的诗作（包括已有的和新增的）
            existing_result: 已有的评审结果（用于续作）
            use_gold: 是否使用金标准参考
        """
        # 检查哪些作品需要评审（续作模式）
        existing_reviewed = existing_reviewed_ids.get(round_id, set())
        poems_to_review = [p for p in poems if p.get("id", "") not in existing_reviewed]
        existing_reviews = []
        
        if existing_result and existing_reviewed:
            existing_reviews = existing_result.get("reviews", [])
            LOGGER.info("开始评审 round: %s（续作模式，已有 %d 首，新增 %d 首，模式：%s）", 
                      round_id, len(existing_reviews), len(poems_to_review), args.rank_mode)
        else:
            LOGGER.info("开始评审 round: %s（共 %d 首作品，模式：%s）", round_id, len(poems), args.rank_mode)
            poems_to_review = poems
        
        # 获取统一背景：优先根据 root_id 从 hongloumeng.json 提取
        root_id_for_round = poems[0].get("root_id") if poems else None
        poem_background = poems[0].get("background") if poems else {}
        background_text = build_background_text_from_root(root_id_for_round, poem_background)
        
        # 获取根节点原诗和专家评估（如果启用）
        root_gold_entry = None
        if args.use_root_gold and root_id_for_round is not None:
            root_gold_entry = root_source_meta.get(str(root_id_for_round))
            if not isinstance(root_gold_entry, dict):
                root_gold_entry = None
        
        # 选择金标准样本（排除与当前 root_id 相同的作品）
        gold_samples = []
        gold_samples_count = max(1, args.gold_samples)  # 至少1个样本（即使不使用金标准也要初始化）
        if use_gold and poems:
            root_id = poems[0].get("root_id")
            # 过滤掉与当前 root_id 相同的作品
            candidates = [
                g for g in all_gold_entries
                if isinstance(g, dict) and g.get("id") is not None and g.get("id") != root_id
            ]
            if not candidates:
                # 兜底：如果过滤后为空，则使用全部金标准列表
                candidates = [g for g in all_gold_entries if isinstance(g, dict)]
            
            gold_samples = random.sample(candidates, min(gold_samples_count, len(candidates))) if candidates else []

        # 模式1：逐首点评，再统一排序（现有默认）
        if args.rank_mode == "separate":
            # 使用全局缓存的评审结果
            poems_with_reviews = []
            for poem_entry in poems:
                pid = poem_entry.get("id", "")
                cached = review_cache.get(pid)
                if cached:
                    poems_with_reviews.append(cached)
                else:
                    # 兜底：如果缓存缺失，现场评审（使用随机虚拟ID）
                    virtual_id = str(random.randint(1, 1000))
                    review_result = process_one_poem(
                        poem_entry, round_id, gold_samples, background_text, gold_samples_count, use_gold, virtual_id=virtual_id, root_gold_entry=root_gold_entry
                    )
                    poems_with_reviews.append(review_result)
            LOGGER.info("Round %s 使用缓存评审共 %d 首，开始统一排序", round_id, len(poems_with_reviews))
            
            ranking_prompt, id_mapping = build_ranking_prompt(poems_with_reviews)
            
            last_error = None
            ranking_result = None
            for attempt in range(1, 11):
                try:
                    response, _ = call_LLM(ranking_prompt, model_name=args.model)
                    ranking_payload = extract_json_payload(response)
                    
                    if "ranking" not in ranking_payload:
                        raise ValueError("排序输出 JSON 缺少 'ranking' 字段")
                    
                    ranking_result = ranking_payload
                    break
                
                except Exception as e:
                    last_error = e
                    LOGGER.warning("Round %s 排序第 %d 次失败：%s", round_id, attempt, e)
                    if attempt >= 10:
                        break
                    continue
            
            if ranking_result is None:
                LOGGER.error("Round %s 排序失败，使用默认排序", round_id)
                # 默认按position排序
                for idx, item in enumerate(poems_with_reviews, 1):
                    item["rank"] = idx
                ranking_result = {
                    "ranking": [{"id": item["id"], "rank": item["rank"]} for item in poems_with_reviews],
                    "overall": f"排序失败：{last_error}"
                }
            else:
                # 将排名信息合并到poems_with_reviews中
                # ranking_result 中的 id 是 "作品1" 格式，需要通过 id_mapping 转换为实际 id
                rank_map: Dict[str, int] = {}
                for rank_item in ranking_result.get("ranking", []):
                    work_id = rank_item.get("id", "")  # 如 "作品1"
                    actual_id = id_mapping.get(work_id, "")  # 转换为 "1-1"
                    rank = rank_item.get("rank", 0)
                    if actual_id:
                        rank_map[actual_id] = rank
                
                for item in poems_with_reviews:
                    item["rank"] = rank_map.get(item.get("id", ""), 0)
            
        else:
            # 模式2：一次性点评+排序
            # 只处理需要评审的新作品
            poems_base: List[Dict[str, Any]] = []
            for poem_entry in poems_to_review:
                # 使用原始的 poet_id，不要重新生成
                poem_id = poem_entry.get("id", "")
                poems_base.append(
                    {
                        "id": poem_id,
                        "source_id": poem_entry.get("source_id", ""),
                        "round": round_id,
                        "position": poem_entry.get("position", 0),
                        "title": poem_entry.get("title", "无题"),
                        "poem": poem_entry.get("poem", ""),
                        "author": poem_entry.get("anonymous_name", "") or "匿名作者",
                        "model": poem_entry.get("model", ""),
                        "notes": poem_entry.get("notes", ""),
                    }
                )
            
            if not poems_base:
                # 如果没有新作品需要评审，直接使用已有结果（已有结果应该已经有排名）
                poems_with_reviews = existing_reviews.copy()
                LOGGER.info("Round %s 无新作品需要评审，使用已有 %d 首作品的排名", round_id, len(poems_with_reviews))
                
                # 检查是否已有排名，如果没有则使用默认排序
                has_rank = any(item.get("rank", 0) > 0 for item in poems_with_reviews)
                if not has_rank:
                    LOGGER.warning("Round %s 已有评审没有排名信息，使用默认排序", round_id)
                    for idx, item in enumerate(poems_with_reviews, 1):
                        item["rank"] = idx
                
                ranking_result = {
                    "ranking": [{"id": item.get("id", ""), "rank": item.get("rank", 0)} for item in poems_with_reviews],
                    "overall": "无新作品，使用已有排名"
                }
            else:
                # 处理新作品
                batch_prompt, virtual_to_real = build_batch_prompt(poems_base, gold_samples, background_text, gold_samples_count, use_gold, root_gold_entry=root_gold_entry)
                real_to_virtual = {v: k for k, v in virtual_to_real.items()}  # 反向映射：真实ID -> 虚拟ID
                last_error = None
                ranking_result = None
                for attempt in range(1, 11):
                    try:
                        response, _ = call_LLM(batch_prompt, model_name=args.model)
                        ranking_payload = extract_json_payload(response)
                        if "reviews" not in ranking_payload or "ranking" not in ranking_payload:
                            raise ValueError("输出 JSON 缺少 'reviews' 或 'ranking' 字段")
                        ranking_result = ranking_payload
                        break
                    except Exception as e:
                        last_error = e
                        LOGGER.warning("Round %s 一次性点评/排序第 %d 次失败：%s", round_id, attempt, e)
                        if attempt >= 10:
                            break
                        continue

                new_reviews: List[Dict[str, Any]] = []
                # 模型返回的 reviews 中的 id 是虚拟ID，需要映射回真实ID
                review_map = {}
                for item in (ranking_result.get("reviews", []) if ranking_result else []):
                    if isinstance(item, dict):
                        virtual_id = str(item.get("id", ""))
                        real_id = virtual_to_real.get(virtual_id, "")
                        if real_id:
                            review_map[real_id] = item.get("review", "")
                
                for base in poems_base:
                    pid = base["id"]  # 真实ID
                    merged = dict(base)
                    merged["review"] = review_map.get(pid, f"未返回点评（{last_error}）" if last_error else "")
                    new_reviews.append(merged)
                
                # 将 ranking_result 中的虚拟ID映射回真实ID，并直接使用第一步的排名
                final_ranking = []
                rank_map: Dict[str, int] = {}  # 真实ID -> 排名
                
                if ranking_result:
                    for rank_item in ranking_result.get("ranking", []):
                        virtual_id = str(rank_item.get("id", ""))
                        real_id = virtual_to_real.get(virtual_id, "")
                        rank = rank_item.get("rank", 0)
                        if real_id:
                            rank_item["id"] = real_id  # 更新为真实ID
                            rank_map[real_id] = rank
                            final_ranking.append(rank_item)
                
                # 将排名信息合并到 new_reviews 中
                for review_item in new_reviews:
                    real_id = review_item.get("id", "")
                    review_item["rank"] = rank_map.get(real_id, 0)
                
                # 合并已有评审和新评审
                poems_with_reviews = existing_reviews + new_reviews
                if new_reviews:
                    LOGGER.info("Round %s 新增 %d 首评审完成，合并后共 %d 首，使用第一步的排名结果", 
                              round_id, len(new_reviews), len(poems_with_reviews))
                
                # 构建最终的 ranking_result（包含所有诗歌的排名）
                if not final_ranking:
                    # 如果没有排名结果，使用默认排序
                    for idx, item in enumerate(new_reviews, 1):
                        item["rank"] = idx
                    final_ranking = [{"id": item.get("id", ""), "rank": item.get("rank", 0)} for item in new_reviews]
                
                # 添加已有评审的排名（如果它们有排名的话）
                for existing_item in existing_reviews:
                    existing_id = existing_item.get("id", "")
                    existing_rank = existing_item.get("rank", 0)
                    if existing_rank > 0:
                        # 检查是否已经在 final_ranking 中
                        if not any(r.get("id") == existing_id for r in final_ranking):
                            final_ranking.append({"id": existing_id, "rank": existing_rank})
                
                ranking_result = {
                    "ranking": final_ranking,
                    "overall": ranking_result.get("overall", "") if ranking_result else ""
                }
        
        result = {
            "round": round_id,
            "source_id": poems[0].get("source_id", "") if poems else "",
            "total_poems": len(poems),
            "reviews": poems_with_reviews,
            "ranking": ranking_result.get("ranking", []),
            "overall": ranking_result.get("overall", ""),
        }
        
        LOGGER.info("✓ Round %s 完成，共 %d 首作品", round_id, len(poems_with_reviews))
        return result
    
    # 检查哪些 round 需要处理（包括新 round 和需要续作的 round）
    pending_rounds = {}
    for round_id, poems in poems_by_round.items():
        existing_reviewed = existing_reviewed_ids.get(round_id, set())
        poems_to_review = [p for p in poems if p.get("id", "") not in existing_reviewed]
        if poems_to_review:
            pending_rounds[round_id] = poems
    
    if not pending_rounds:
        LOGGER.info("所有 round 的所有作品均已评审")
        # 即使没有新数据需要处理，也要输出统计结果
        final_results = sorted(results_map.values(), key=lambda x: x.get("round", ""))
        if final_results:
            LOGGER.info("")
            LOGGER.info("=" * 60)
            LOGGER.info("全部评审结果（共 %d 个 round）", len(final_results))
            LOGGER.info("=" * 60)
            
            # 生成Excel和统计报告
            try:
                generate_excel_and_statistics(final_results, output_path, args.rank_mode)
            except Exception as e:
                LOGGER.error("生成Excel和统计报告失败：%s", e, exc_info=True)
        return

    # 若为 separate 模式，先对所有涉及的诗去重评审，便于复用
    if args.rank_mode == "separate":
        unique_poems = {}
        for poems in pending_rounds.values():
            for p in poems:
                pid = p.get("id")
                if pid and pid not in unique_poems:
                    unique_poems[pid] = p
        evaluate_missing_poems(list(unique_poems.values()), use_gold)

    # 并发或串行处理各个 round，边生成边写入
    if len(pending_rounds) == 1 or workers == 1:
        LOGGER.info("串行处理 %d 个待处理 round", len(pending_rounds))
        for round_id, poems in sorted(pending_rounds.items()):
            existing_result = results_map.get(round_id)
            result = process_one_round(round_id, poems, existing_result, use_gold)
            with lock:
                results_map[round_id] = result
                write_output()
    else:
        LOGGER.info("使用并发线程数: %d 处理 %d 个待处理 round", workers, len(pending_rounds))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_round = {
                executor.submit(process_one_round, round_id, poems, results_map.get(round_id), use_gold): round_id
                for round_id, poems in pending_rounds.items()
            }
            for future in as_completed(future_to_round):
                round_id = future_to_round[future]
                try:
                    result = future.result()
                except Exception as e:
                    LOGGER.error("Round %s 处理异常：%s", round_id, e)
                    result = {
                        "round": round_id,
                        "total_poems": len(poems_by_round.get(round_id, [])),
                        "reviews": [],
                        "ranking": [],
                        "overall": f"评审失败：{e}",
                        "error": str(e),
                    }
                with lock:
                    results_map[round_id] = result
                    write_output()

    final_results = sorted(results_map.values(), key=lambda x: x.get("round", ""))
    output_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("全部评审结果已写入：%s（共 %d 个 round）", output_path, len(final_results))
    LOGGER.info("=" * 60)
    
    # 输出 TOKEN_COST 统计
    LOGGER.info("")
    LOGGER.info(format_token_cost_summary())
    
    # 将 TOKEN_COST 写入单独的元数据文件
    try:
        token_cost_data = get_token_cost_summary()
        if token_cost_data:
            metadata_path = output_path.with_suffix('.token_cost.json')
            metadata_path.write_text(
                json.dumps(token_cost_data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            LOGGER.info("TOKEN_COST 统计已写入：%s", metadata_path)
    except Exception as e:
        LOGGER.warning("写入 TOKEN_COST 元数据失败：%s", e)
    
    # 生成Excel和统计报告
    try:
        generate_excel_and_statistics(final_results, output_path, args.rank_mode)
    except Exception as e:
        LOGGER.error("生成Excel和统计报告失败：%s", e, exc_info=True)


if __name__ == "__main__":
    main()
