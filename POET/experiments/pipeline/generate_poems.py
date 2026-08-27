#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Digital Poetry Party: models compose under the same Honglou-Poem prompt (§2.2 / Table 1).

Previous poems are hidden by default so each model writes independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

from common.llm_client import call_LLM, format_token_cost_summary, get_token_cost_summary

LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "glm-4.6",
    "kimi-k2",
    "gemini-2.5-pro",
    "o3",
    "gpt-5.2",
    "gemini-3",
    "claude-4.5",
    "hunyuan-turbos",
    "deepseek-reasoner",
    "v3.1-think",
    "gpt-4o",
    "grok-4",
]


def extract_json_payload(response: str) -> dict:
    """
    从模型响应中提取 JSON，增强兼容性处理控制字符。
    
    处理策略：
    1. 直接解析（最快）
    2. 修复字符串值中的控制字符（换行符、制表符等）
    3. 手动提取字段（最后手段）
    """
    # 首先尝试找到 JSON 块
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
    # 使用状态机方法：遍历 JSON 字符串，在字符串值中转义控制字符
    try:
        fixed_chars = []
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                # 当前字符是转义后的，直接添加
                fixed_chars.append(char)
                escape_next = False
            elif char == '\\':
                # 转义字符
                fixed_chars.append(char)
                escape_next = True
            elif char == '"':
                # 引号：切换字符串状态
                fixed_chars.append(char)
                in_string = not in_string
            elif in_string:
                # 在字符串值中
                if char == '\n':
                    fixed_chars.append('\\n')
                elif char == '\r':
                    fixed_chars.append('\\r')
                elif char == '\t':
                    fixed_chars.append('\\t')
                elif ord(char) < 32:  # 其他控制字符
                    # 跳过或转义
                    fixed_chars.append(f'\\u{ord(char):04x}')
                else:
                    fixed_chars.append(char)
            else:
                # 不在字符串中，直接添加
                fixed_chars.append(char)
            
            i += 1
        
        fixed_json = ''.join(fixed_chars)
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        LOGGER.debug("修复后解析仍失败：%s", str(e)[:100])
    
    # 策略3：手动提取字段（最后手段）
    try:
        result = {}
        
        # 提取 title
        title_match = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
        if title_match:
            title_text = title_match.group(1)
            title_text = title_text.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"')
            result["title"] = title_text
        
        # 提取 poem（可能包含未转义的换行符）
        # 先尝试标准格式
        poem_match = re.search(r'"poem"\s*:\s*"', json_str)
        if poem_match:
            # 找到 poem 值的开始位置
            start_pos = poem_match.end()
            # 手动查找结束引号（考虑转义）
            end_pos = start_pos
            while end_pos < len(json_str):
                if json_str[end_pos] == '"' and (end_pos == start_pos or json_str[end_pos - 1] != '\\'):
                    break
                end_pos += 1
            
            if end_pos < len(json_str):
                poem_text = json_str[start_pos:end_pos]
                # 转义控制字符
                poem_text = poem_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                # 然后解析转义序列
                poem_text = poem_text.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"')
                result["poem"] = poem_text
        
        # 提取 notes
        notes_match = re.search(r'"notes"\s*:\s*"', json_str)
        if notes_match:
            start_pos = notes_match.end()
            end_pos = start_pos
            while end_pos < len(json_str):
                if json_str[end_pos] == '"' and (end_pos == start_pos or json_str[end_pos - 1] != '\\'):
                    break
                end_pos += 1
            
            if end_pos < len(json_str):
                notes_text = json_str[start_pos:end_pos]
                notes_text = notes_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                notes_text = notes_text.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"')
                result["notes"] = notes_text
        
        if result:
            LOGGER.info("使用手动提取方式，提取到字段：%s", list(result.keys()))
            # 确保必要字段存在
            result.setdefault("poem", "")
            result.setdefault("title", "未知")
            result.setdefault("notes", "")
            return result
    except Exception as e:
        LOGGER.warning("手动提取也失败：%s", e)
    
    # 最后回退
    LOGGER.error("所有 JSON 解析策略都失败")
    return {
        "title": "解析失败",
        "poem": json_str[:500] if len(json_str) > 500 else json_str,
        "notes": "JSON 解析失败，请检查模型输出格式"
    }


def build_prompt(task: Dict[str, Any], previous_poems: List[Dict[str, str]], current_position: int, show_previous: bool = False) -> List[dict]:
    """
    构建诗会 prompt，包含前面选手的诗作（匿名）。
    
    Args:
        task: 诗会任务（题目、背景等）
        previous_poems: 前面选手的诗作列表，格式 [{"title": "...", "poem": "..."}, ...]
        current_position: 当前选手的位置（从1开始）
        show_previous: 是否显示前面选手的作品（默认 False，与论文独立作诗一致）
    """
    topic = task.get("topic") or task.get("title") or "即席吟"
    background = task.get("background") or task.get("motivation") or ""
    author_hint = task.get("author") or task.get("persona") or "自拟身份"
    extra = task.get("notes") or task.get("instruction") or ""
    
    # 构建前面选手作品文本
    previous_poems_text = ""
    if show_previous and previous_poems:
        previous_poems_text = f"\n【前面选手的作品】（共 {len(previous_poems)} 首，匿名展示，供参考交流）\n"
        for idx, prev_poem in enumerate(previous_poems, 1):
            player_name = f"选手{idx}"
            prev_title = prev_poem.get("title", "无题")
            prev_poem_text = prev_poem.get("poem", "")
            previous_poems_text += f"{player_name} 的作品：\n题目：{prev_title}\n{prev_poem_text}\n\n"
        previous_poems_text += f"你是第 {current_position} 位作诗的选手，请创作你的作品。\n"
    elif not show_previous and previous_poems:
        previous_poems_text = f"请创作你的作品。\n"
    else:
        previous_poems_text = "你是第一位作诗的选手，请创作你的作品。\n"
    
    extra_text = f"\n【额外要求】{extra}" if extra else ""
    
    reference_text = "，也可参考前面选手的作品风格，但不可抄袭" if show_previous and previous_poems else ""
    user_text = f'''你正在参与一场古风诗会，多位文人雅客围绕同一题目轮流作诗，互相交流切磋，尽可能做出优秀的诗，取得诗会胜利。
请依据给定背景写一首原创古体诗（五言/七言/词皆可），题目可自拟或改写。
须兼顾意境与格律，可适度创新{reference_text}。

**重要：必须返回 JSON 格式**，格式为：
{{"title": "题目", "poem": "诗歌正文", "notes": "创作思路与手法"}}
不要附加任何额外解释文字。

【诗会题旨】{topic}
【作诗背景 / 动机】
{background}

【作者 / 人设】{author_hint}
{previous_poems_text}{extra_text}'''
    
    return [
        {
            "role": "system",
            "content": "你是一位参与诗会的诗人，需要现场作答并输出 JSON。注意：前面选手的作品是匿名展示的，你不知道他们的真实身份。",
        },
        {"role": "user", "content": user_text},
    ]


def process_one_task(task: Dict[str, Any], task_idx: int, model_list: List[str], seed: int = None, show_previous: bool = False, existing_items: List[Dict[str, Any]] = None, start_position: int = 0) -> List[Dict[str, Any]]:
    """
    处理单个任务的诗会，返回该任务的所有结果。
    
    注意：同一任务内的多个模型必须按顺序作诗，不能并发。
    
    Args:
        task: 诗会任务
        task_idx: 任务索引
        model_list: 模型列表（只包含需要补充的新模型）
        seed: 随机种子
        show_previous: 是否显示前面选手的作品（默认 False，与论文独立作诗一致）
        existing_items: 已有的结果项（用于续作时获取前面的诗作）
        start_position: 起始位置（续作时从已有最大 position + 1 开始）
    """
    task_id = task.get("id") or f"task-{task_idx+1}"
    is_continuation = existing_items is not None and len(existing_items) > 0
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    if is_continuation:
        LOGGER.info("续作诗会：%s（题目：%s）", task_id, task.get("topic") or task.get("title") or "未知")
    else:
        LOGGER.info("开始诗会：%s（题目：%s）", task_id, task.get("topic") or task.get("title") or "未知")
    LOGGER.info("=" * 60)
    
    # 构建前面选手的诗作（包括已有的和新增的）
    previous_poems: List[Dict[str, str]] = []
    if existing_items:
        # 从已有结果中提取前面选手的诗作
        for item in sorted(existing_items, key=lambda x: x.get("position", 0)):
            previous_poems.append({
                "title": item.get("title", "无题"),
                "poem": item.get("poem", ""),
            })
        LOGGER.info("已加载 %d 首前面选手的作品", len(previous_poems))
    
    # 为这个任务随机打乱模型顺序（使用任务索引作为种子的一部分，确保每个任务打乱顺序不同）
    shuffled_models = model_list.copy()
    if seed is not None:
        random.seed(seed + task_idx)  # 每个任务使用不同的种子
    random.shuffle(shuffled_models)
    LOGGER.info("本场诗会的模型顺序（已随机打乱）：%s", " -> ".join(shuffled_models))
    
    task_results: List[Dict[str, Any]] = []
    
    # 按顺序依次作诗（不能并发，因为后面的人需要看到前面的诗）
    current_position = start_position
    for model_name in shuffled_models:
        current_position += 1
        model_idx = current_position
        LOGGER.info("")
        LOGGER.info("--- 选手 %d/%d 正在作诗（模型：%s，匿名：选手%d）---", 
                   model_idx, len(shuffled_models), model_name, model_idx)
        
        try:
            # 构建 prompt，包含前面选手的诗作（匿名）
            prompt = build_prompt(task, previous_poems, model_idx, show_previous)
            response, _ = call_LLM(prompt, model_name=model_name)
            payload = extract_json_payload(response)
            
            result = {
                "source_id": task_id,
                "id": f"{task_id}-{model_idx}",
                "round": task_id,
                "position": model_idx,  # 在本场诗会中的位置
                "model": model_name,
                "anonymous_name": f"选手{model_idx}",  # 匿名名称
                "title": payload.get("title", task.get("topic")),
                "poem": payload.get("poem", response),
                "notes": payload.get("notes", ""),
                "background": task,
            }
            
            # 将当前选手的诗作添加到 previous_poems（匿名，不包含模型信息）
            # 这样后续的新模型可以看到这个模型的作品
            previous_poems.append({
                "title": result["title"],
                "poem": result["poem"],
            })
            
            LOGGER.info("✓ 选手 %d（%s）完成：%s", model_idx, model_name, result["title"])
            task_results.append(result)
            
        except Exception as exc:
            LOGGER.error("✗ 选手 %d（%s）生成失败：%s", model_idx, model_name, exc) #要不要考虑取消这一部份？生成失败直接不写入？算了吧，至少还好看点
            result = {
                "round": task_id,
                "position": model_idx,
                "model": model_name,
                "anonymous_name": f"选手{model_idx}",
                "title": task.get("topic") or "生成失败",
                "poem": "",
                "notes": f"生成失败：{exc}",
                "background": task,
            }
            task_results.append(result)
            # 即使失败，也继续下一轮（但 previous_poems 不添加失败的作品）
    
    return task_results


def main():
    parser = argparse.ArgumentParser(description="Digital Poetry Party generation (§2.2)")
    parser.add_argument(
        "--tasks",
        default=str(REPO_ROOT / "data" / "honglou" / "honglou_poems.json"),
        help="Honglou-Poem JSON (default: data/honglou/honglou_poems.json)",
    )
    parser.add_argument("--output", default=str(Path(__file__).with_name("poetry_meet_results.json")), help="输出路径")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated API model ids (sent as-is; defaults to the 12 models in Table 1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于打乱模型顺序，默认随机）")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数（用于并发处理不同任务，同一任务内的多个模型仍按顺序作诗）")
    parser.add_argument("--show-previous", action="store_true", default=False, help="Show earlier poems in the same meeting (off by default; paper setting is independent writing)")
    parser.add_argument("--hide-previous", action="store_false", dest="show_previous", help="Hide earlier poems (paper default)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tasks_path = Path(args.tasks).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks 文件需要是数组")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        raise ValueError("模型列表为空")
    
    # 显示模型列表
    LOGGER.info("模型列表（共 %d 个）：%s", len(model_list), ", ".join(model_list))
    
    # 设置全局随机种子（如果指定）
    if args.seed is not None:
        random.seed(args.seed)
        LOGGER.info("使用随机种子：%d", args.seed)
    
    # 读取已存在的输出，用于断点续跑和诗会续作
    all_results: List[Dict[str, Any]] = []
    # 按 source_id 分组，记录每个任务已测试的模型和最大 position
    existing_by_source: Dict[str, Dict[str, Any]] = {}  # {source_id: {"models": set, "max_position": int, "items": list}}
    
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if not isinstance(item, dict):
                        continue
                    source_id = str(item.get("source_id", ""))
                    round_id = item.get("round", "")
                    model = item.get("model", "")
                    position = item.get("position", 0)
                    
                    if source_id:
                        if source_id not in existing_by_source:
                            existing_by_source[source_id] = {
                                "models": set(),
                                "max_position": 0,
                                "items": []
                            }
                        if model:
                            existing_by_source[source_id]["models"].add(model)
                        existing_by_source[source_id]["max_position"] = max(
                            existing_by_source[source_id]["max_position"], position
                        )
                        existing_by_source[source_id]["items"].append(item)
                        all_results.append(item)
                
                LOGGER.info("检测到已有结果文件，已加载 %d 个结果，涉及 %d 个任务", 
                          len(all_results), len(existing_by_source))
                for sid, info in existing_by_source.items():
                    LOGGER.info("  任务 %s: 已测试 %d 个模型，最大 position=%d", 
                              sid, len(info["models"]), info["max_position"])
        except Exception as exc:
            LOGGER.warning("读取已有输出失败，将重新生成：%s", exc)
            all_results = []
            existing_by_source = {}
    
    # 构建待处理任务列表（支持续作：检查每个任务需要补充哪些模型）
    pending_tasks: List[tuple[int, Dict[str, Any], List[str]]] = []  # (task_idx, task, new_models)
    for task_idx, task in enumerate(tasks):
        task_id = str(task.get("id") or f"task-{task_idx+1}")
        
        # 检查这个任务已测试的模型
        existing_models = existing_by_source.get(task_id, {}).get("models", set())
        new_models = [m for m in model_list if m not in existing_models]
        
        if not new_models:
            LOGGER.info("任务 %s: 所有模型已测试，跳过", task_id)
            continue
        
        if existing_models:
            LOGGER.info("任务 %s: 续作模式，已有 %d 个模型，将补充 %d 个新模型: %s", 
                      task_id, len(existing_models), len(new_models), ", ".join(new_models))
        else:
            LOGGER.info("任务 %s: 新任务，将测试 %d 个模型", task_id, len(new_models))
        
        pending_tasks.append((task_idx, task, new_models))
    
    if not pending_tasks:
        LOGGER.info("所有任务均已完成，直接退出")
        # 按任务顺序和位置排序结果（保持输出的一致性）
        all_results.sort(key=lambda x: (x.get("round", ""), x.get("position", 0)))
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("诗会结果已写入：%s（共 %d 个结果）", output_path, len(all_results))
        LOGGER.info("=" * 60)
        return
    
    workers = max(1, args.workers)
    lock = threading.Lock()
    
    def write_output():
        """将当前结果实时写盘，便于断点续跑。"""
        sorted_results = sorted(all_results, key=lambda x: (x.get("round", ""), x.get("position", 0)))
        output_path.write_text(json.dumps(sorted_results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 如果只有一个任务或只有一个worker，串行处理
    if len(pending_tasks) == 1 or workers == 1:
        LOGGER.info("串行处理 %d 个待处理任务", len(pending_tasks))
        for task_idx, task, new_models in pending_tasks:
            task_id = str(task.get("id") or f"task-{task_idx+1}")
            existing_info = existing_by_source.get(task_id, {})
            existing_items = existing_info.get("items", [])
            start_pos = existing_info.get("max_position", 0)
            
            task_results = process_one_task(
                task, task_idx, new_models, args.seed, args.show_previous,
                existing_items, start_pos
            )
            with lock:
                all_results.extend(task_results)
                write_output()
    else:
        # 并发处理多个任务
        LOGGER.info("使用并发线程数: %d 处理 %d 个待处理任务", workers, len(pending_tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {}
            for task_idx, task, new_models in pending_tasks:
                task_id = str(task.get("id") or f"task-{task_idx+1}")
                existing_info = existing_by_source.get(task_id, {})
                existing_items = existing_info.get("items", [])
                start_pos = existing_info.get("max_position", 0)
                
                future = executor.submit(
                    process_one_task, task, task_idx, new_models, args.seed, args.show_previous,
                    existing_items, start_pos
                )
                future_to_task[future] = (task_idx, task)
            for future in as_completed(future_to_task):
                task_idx, task = future_to_task[future]
                try:
                    task_results = future.result()
                    with lock:
                        all_results.extend(task_results)
                        write_output()
                    LOGGER.info("任务 %d 完成，新增 %d 个结果", task_idx + 1, len(task_results))
                except Exception as exc:
                    LOGGER.error("任务 %d 处理失败：%s", task_idx + 1, exc)
    
    # 按任务顺序和位置排序结果（保持输出的一致性）
    all_results.sort(key=lambda x: (x.get("round", ""), x.get("position", 0)))
    
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("诗会结果已写入：%s（共 %d 个结果）", output_path, len(all_results))
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

