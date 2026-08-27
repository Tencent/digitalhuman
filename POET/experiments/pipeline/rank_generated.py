#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expert-persona ranking of generated poems (§3.2 / Table 1 judges).

Batch mode (default in the paper) vs separate; n=5 ICL demonstrations.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from common.llm_client import call_LLM, format_token_cost_summary, get_token_cost_summary

LOGGER = logging.getLogger(__name__)


def extract_json_payload(response: str) -> dict:
    """
    从模型响应中提取 JSON，增强兼容性处理控制字符。
    使用与 step2 相同的容错机制。
    """
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
        raise ValueError(f"JSON 解析失败：{e}")


def summarize_poem(entry: Dict[str, Any]) -> str:
    """
    总结一首参赛作品的信息。
    """
    title = entry.get("title") or "无题"
    anonymous_name = entry.get("anonymous_name", "")  # 诗会中的匿名名称
    poem_text = entry.get("poem", "").strip()
    notes = entry.get("notes", "")
    
    summary = [
        f"标题:{title}",
    ]
    
    #目前取消显示作者了，后面有需要可以加回去
    
    # # 如果有匿名名称，显示匿名名称；否则显示"匿名作者"
    # if anonymous_name:
    #     summary.append(f"作者:{anonymous_name}（匿名）")
    # else:
    #     summary.append("作者:匿名作者")
    
    summary.append(f"诗歌ID:{entry.get('id','')}\n诗歌:\n{poem_text}")
    
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
    reference_gold: Dict[str, Any],
    gold_samples: List[Dict[str, Any]],
    background_text: str,
    gold_samples_count: int = 4,
) -> List[dict]:
    """
    为单首诗构建专家评审 prompt。
    
    Args:
        poem_entry: 待评审的诗作
        reference_gold: 对应的背景诗（金标准）
        gold_samples: 5个专家样本（1个背景诗+4个随机）
        background_text: 统一背景
    """
    poem_summary = summarize_poem(poem_entry)
    
    # 构建5个专家样本
    gold_section_parts = []
    # 第一个：背景诗+真实解读
    if reference_gold:
        gold_section_parts.append(f"【参考样本1】（背景诗）\n{summarize_gold(reference_gold, 1)}")
    
    # 另外 N 个随机样本
    for idx, sample in enumerate(gold_samples[:gold_samples_count], 2):
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
    
    return [
        {
            "role": "system",
            "content": "你是金标准参考样本中那位古典诗词评论专家的数字人化身。你的任务是全面学习并内化专家的评审风格、语言表达、分析角度、论述逻辑、专业术语等所有特征，然后以专家的身份和口吻对参赛作品进行专业评析。你的评述应该让人一读即知是专家亲笔所写，而非模仿之作。必须输出有效的 JSON 格式，只包含 review 字段。",
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
    reference_gold: Dict[str, Any],
    gold_samples: List[Dict[str, Any]],
    background_text: str,
    gold_samples_count: int = 4,
) -> List[dict]:
    """
    构建“一次性评论+排序”模式的 prompt：
    - 输入一整个 round 的诗作列表
    - 输出每首诗的 review 以及整体 ranking 和 overall
    """
    gold_section_parts = []
    if reference_gold:
        gold_section_parts.append(f"【参考样本1】（背景诗）\n{summarize_gold(reference_gold, 1)}")
    for idx, sample in enumerate(gold_samples[:gold_samples_count], 2): 
        gold_section_parts.append(f"【参考样本{idx}】\n{summarize_gold(sample, idx)}")
    gold_section = "\n\n".join(gold_section_parts)

    poems_text = []
    for item in poems:
        poem_info = f"ID:{item['id']}\n标题:{item['title']}\n诗歌:\n{item['poem']}\n"
        notes = item.get("notes", "")
        if notes:
            poem_info += f"作者自述：{notes}\n"
        poems_text.append(poem_info)
    poems_section = "\n\n".join(poems_text)

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

    return [
        {
            "role": "system",
            "content": "你是金标准参考样本中那位古典诗词评论专家。你的任务是全面学习并内化专家的评审风格、语言表达、分析角度、论述逻辑、专业术语等所有特征，然后以专家的身份和口吻一次性给出多首诗的点评与排名。你的评述应该让人一读即知是专家亲笔所写。必须输出有效的 JSON 格式。",
        },
        {"role": "user", "content": user_text},
    ]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Expert-persona ranking of generated poems (§3.2)")
    parser.add_argument("--poems", required=True, help="诗会结果 JSON（step2 输出）")
    parser.add_argument(
        "--gold",
        nargs="+",
        required=False,
        default=[],
        help="Optional JSON of expert critiques for ICL (title / poet / poet_analyse). Omit to rank without demonstrations.",
    )
    parser.add_argument("--output", default=str(Path(__file__).with_name("poetry_ranking.json")), help="输出路径")
    parser.add_argument("--model", default="gpt-4o", help="API model id (sent as-is)")
    parser.add_argument(
        "--rank-mode",
        choices=["separate", "batch"],
        default="batch",
        help="batch: rank the whole group at once (paper); separate: review one poem at a time",
    )
    parser.add_argument(
        "--gold-samples",
        type=int,
        default=4,
        help="金标准参考样本数量（不包括背景诗，默认 4 个，总共 5 个样本：1 个背景诗 + N 个随机样本）",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发线程数（用于并发处理不同 round，默认 1）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    poem_path = Path(args.poems).expanduser().resolve()
    poem_entries = load_json(poem_path)
    if not isinstance(poem_entries, list):
        raise ValueError("poems 文件必须是数组")

    # 加载全部金标准数据
    gold_paths = args.gold or []
    all_gold_entries: List[Dict[str, Any]] = []
    for path_str in gold_paths:
        records = load_json(Path(path_str).expanduser().resolve())
        if isinstance(records, list):
            all_gold_entries.extend(records)
    
    LOGGER.info("加载了 %d 首金标准参考作品", len(all_gold_entries))
    
    # 按 round 分组处理
    from collections import defaultdict
    poems_by_round = defaultdict(list)
    for entry in poem_entries:
        round_id = entry.get("round", "unknown")
        poems_by_round[round_id].append(entry)
    
    LOGGER.info("共发现 %d 个 round，将分别进行评审", len(poems_by_round))
    
    # 读取已存在的输出，用于断点续跑和评审续作
    output_path = Path(args.output).expanduser().resolve()
    results_map: Dict[str, Dict[str, Any]] = {}
    # 记录每个 round 已评审的模型 ID（通过 id 字段识别，如 "1-1", "1-2"）
    existing_reviewed_ids: Dict[str, set] = {}  # {round_id: set of reviewed_ids}
    
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
    
    def find_reference_gold(poem_entry: Dict[str, Any], all_gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据诗会背景找到对应的金标准（背景诗）"""
        background = poem_entry.get("background") or {}
        if isinstance(background, dict):
            topic = background.get("topic") or background.get("title") or ""
            # 尝试通过标题匹配
            for gold in all_gold:
                gold_title = gold.get("title") or gold.get("提取的诗名") or ""
                if topic and gold_title and topic.strip() in gold_title or gold_title.strip() in topic:
                    return gold
        # 如果匹配不上，返回第一个作为默认
        return all_gold[0] if all_gold else {}
    
    def process_one_poem(
        poem_entry: Dict[str, Any],
        round_id: str,
        reference_gold: Dict[str, Any],
        gold_samples: List[Dict[str, Any]],
        background_text: str,
        gold_samples_count: int = 4,
    ) -> Dict[str, Any]:
        """处理单首诗，返回带review的结果"""
        poem_id = f"{round_id}-{poem_entry.get('position', 0)}"
        title = poem_entry.get("title", "无题")
        
        prompt = build_prompt_for_single_poem(poem_entry, reference_gold, gold_samples, background_text, gold_samples_count)
        
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
                return result
            
            except Exception as e:
                last_error = e
                LOGGER.warning("%s 第 %d 次解析失败：%s", poem_id, attempt, e)
                if attempt >= 10:
                    break
                continue
        
        LOGGER.error("✗ %s 评审失败：%s", poem_id, last_error)
        return {
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
    
    def process_one_round(round_id: str, poems: List[Dict[str, Any]], existing_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理单个 round 的所有作品
        
        Args:
            round_id: round ID
            poems: 所有待评审的诗作（包括已有的和新增的）
            existing_result: 已有的评审结果（用于续作）
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
        
        # 获取统一背景
        background = poems[0].get("background") if poems else {}
        background_text = ""
        if isinstance(background, dict):
            background_text = background.get("background") or background.get("topic") or ""
        elif background:
            background_text = str(background)
        
        # 找到对应的背景诗（金标准）
        reference_gold = find_reference_gold(poems[0], all_gold_entries) if poems else {}
        gold_title = reference_gold.get("title") or reference_gold.get("提取的诗名") or ""
        
        # 选择 N 个随机样本（排除背景诗）
        other_gold = [g for g in all_gold_entries if g != reference_gold]
        gold_samples_count = max(1, args.gold_samples)  # 至少1个样本
        random_samples = random.sample(other_gold, min(gold_samples_count, len(other_gold))) if other_gold else []

        # 模式1：逐首点评，再统一排序（现有默认）
        if args.rank_mode == "separate":
            # 先处理需要评审的新作品
            new_reviews: List[Dict[str, Any]] = []
            for poem_entry in poems_to_review:
                review_result = process_one_poem(
                    poem_entry, round_id, reference_gold, random_samples, background_text, gold_samples_count
                )
                new_reviews.append(review_result)
            
            # 合并已有评审和新评审
            if not existing_reviews:
                poems_with_reviews = new_reviews
            else:
                poems_with_reviews = existing_reviews + new_reviews
            
            if new_reviews:
                LOGGER.info("Round %s 新增 %d 首评审完成，合并后共 %d 首，开始统一排序", 
                          round_id, len(new_reviews), len(poems_with_reviews))
            elif existing_reviews:
                LOGGER.info("Round %s 无新作品需要评审，将重新排序已有 %d 首作品", round_id, len(poems_with_reviews))
            else:
                LOGGER.info("Round %s 所有诗评审完成，开始统一排序", round_id)
            
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
                poem_id = f"{round_id}-{poem_entry.get('position', 0)}"
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
                # 如果没有新作品需要评审，直接使用已有结果并重新排序
                poems_with_reviews = existing_reviews.copy()
                LOGGER.info("Round %s 无新作品需要评审，将重新排序已有 %d 首作品", round_id, len(poems_with_reviews))
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
                        LOGGER.warning("Round %s 重新排序第 %d 次失败：%s", round_id, attempt, e)
                        if attempt >= 10:
                            break
                        continue
                
                if ranking_result is None:
                    LOGGER.error("Round %s 重新排序失败，使用默认排序", round_id)
                    for idx, item in enumerate(poems_with_reviews, 1):
                        item["rank"] = idx
                    ranking_result = {
                        "ranking": [{"id": item["id"], "rank": item["rank"]} for item in poems_with_reviews],
                        "overall": f"重新排序失败：{last_error}"
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
                # 处理新作品
                batch_prompt = build_batch_prompt(poems_base, reference_gold, random_samples, background_text, gold_samples_count)
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
                review_map = {
                    item.get("id"): item.get("review", "") for item in (ranking_result.get("reviews", []) if ranking_result else []) if isinstance(item, dict)
                }
                for base in poems_base:
                    pid = base["id"]
                    merged = dict(base)
                    merged["review"] = review_map.get(pid, f"未返回点评（{last_error}）" if last_error else "")
                    new_reviews.append(merged)

                # 合并已有评审和新评审
                poems_with_reviews = existing_reviews + new_reviews
                if new_reviews:
                    LOGGER.info("Round %s 新增 %d 首评审完成，合并后共 %d 首，开始统一排序", 
                              round_id, len(new_reviews), len(poems_with_reviews))
                
                # 重新排序所有作品（包括已有的和新增的）
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
                        LOGGER.warning("Round %s 重新排序第 %d 次失败：%s", round_id, attempt, e)
                        if attempt >= 10:
                            break
                        continue
                
                if ranking_result is None:
                    LOGGER.error("Round %s 重新排序失败，使用默认排序", round_id)
                    for idx, item in enumerate(poems_with_reviews, 1):
                        item["rank"] = idx
                    ranking_result = {
                        "ranking": [{"id": item["id"], "rank": item["rank"]} for item in poems_with_reviews],
                        "overall": f"重新排序失败：{last_error}"
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
        
        result = {
            "round": round_id,
            "source_id": poems[0].get("source_id", "") if poems else "",
            "total_poems": len(poems),
            "gold_title": gold_title,
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
        LOGGER.info("所有 round 的所有作品均已评审，直接退出")
        return

    # 并发或串行处理各个 round，边生成边写入
    if len(pending_rounds) == 1 or workers == 1:
        LOGGER.info("串行处理 %d 个待处理 round", len(pending_rounds))
        for round_id, poems in sorted(pending_rounds.items()):
            existing_result = results_map.get(round_id)
            result = process_one_round(round_id, poems, existing_result)
            with lock:
                results_map[round_id] = result
                write_output()
    else:
        LOGGER.info("使用并发线程数: %d 处理 %d 个待处理 round", workers, len(pending_rounds))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_round = {
                executor.submit(process_one_round, round_id, poems, results_map.get(round_id)): round_id
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


if __name__ == "__main__":
    main()
