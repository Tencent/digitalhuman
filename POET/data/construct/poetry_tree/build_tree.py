#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poetry-Tree construction (§3.1 / Algorithm 1).

Needs a Honglou-Poem JSON that includes expert critiques (``poet_analyse``).
The public ``data/honglou/honglou_poems.json`` does **not** contain those
fields (copyright). This script therefore cannot reproduce the paper trees
from the files in this repo. Download ``poetry_tree.json`` from
https://huggingface.co/datasets/Zihao1/Poet-4B_training_data for evaluation,
or email yizh6@mail2.sysu.edu.cn for the complete data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

from common.llm_client import call_LLM, format_token_cost_summary, get_token_cost_summary


# -------------------------------------------------
# 日志配置：控制台格式 + 同步写入文件，保持原有格式
# -------------------------------------------------
LOG_DIR = Path("output/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOG_PATH = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
DEFAULT_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"

if not getattr(logger, "_pipeline_log_configured", False):
    logger.add(
        PIPELINE_LOG_PATH,
        format=DEFAULT_LOG_FORMAT,
        encoding="utf-8",
        enqueue=True,
    )
    logger.info(f"日志写入文件: {PIPELINE_LOG_PATH}")
    logger._pipeline_log_configured = True


class PoetryExtractor:
    """从输入文件中提取诗词信息
    
    支持：
    - hongloumeng.json：结构化诗词数据
    - .txt / .pdf：旧版从文本/PDF中解析（保留兼容）
    """
    
    def __init__(self, pdf_path: str, model_name: str = "dsv3", output_file: Optional[str] = None):
        self.pdf_path = Path(pdf_path)
        self.model_name = model_name
        self.extracted_poems = []
        self.output_file = Path(output_file) if output_file else None
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self) -> str:
        """读取原始文本；若传入 .txt 文件则直接读取（兼容旧流程）"""
        logger.info(f"开始读取文本文件: {self.pdf_path}")
        full_text = ""
        
        try:
            if self.pdf_path.suffix.lower() == ".txt":
                with open(self.pdf_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            else:
                if pdfplumber is None:
                    raise RuntimeError("Reading PDF requires pdfplumber. Prefer --source data/honglou/honglou_poems.json")
                with pdfplumber.open(self.pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text:
                            full_text += f"\n--- 第{page_num}页 ---\n{text}"
                        if page_num % 50 == 0:
                            logger.info(f"已处理 {page_num} 页")
        except Exception as e:
            logger.error(f"读取文本失败: {e}")
            raise
            
        logger.info(f"文本提取完成，总长度: {len(full_text)} 字符")
        return full_text
    
    def _rule_based_identification(self, text: str) -> List[Dict]:
        """基于规则的场景和诗词识别"""
        items = []
        lines = text.split('\n')
        
        current_scene = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # 跳过页面标记
            if '--- 第' in line and '页 ---' in line:
                i += 1
                continue
            
            # 识别场景标题（通常是大标题，后面有【说明】但没有【注释】）
            # 场景标题特征：标题后几行内有【说明】，但200字符内没有【注释】
            if len(line) > 3 and len(line) < 50 and not line.startswith('【') and not line.startswith('《'):
                look_ahead = '\n'.join(lines[i:min(i+10, len(lines))])
                if '【说明】' in look_ahead:
                    # 检查是否有【注释】，如果有则不是场景
                    look_ahead_short = look_ahead[:500]
                    if '【注释】' not in look_ahead_short:
                        # 这可能是场景标题
                        items.append({
                            "type": "scene",
                            "title": line,
                            "start_marker": line[:min(10, len(line))]
                        })
                        current_scene = line
                        i += 1
                        continue
            
            # 识别单个诗词标题
            # 模式1：标题（第X回）
            if re.search(r'（第[^）]+回）', line):
                title = line.strip()
                items.append({
                    "type": "poem",
                    "title": title,
                    "start_marker": title[:min(10, len(title))],
                    "scene_title": current_scene
                })
                i += 1
                continue
            
            # 模式2：又副册判词之一、副册判词一首等
            if re.search(r'(又副册|副册|正册).*判词', line) or re.search(r'判词.*[之一二三四五六七八九十]', line):
                title = line.strip()
                items.append({
                    "type": "poem",
                    "title": title,
                    "start_marker": title[:min(10, len(title))],
                    "scene_title": current_scene
                })
                i += 1
                continue
            
            # 模式3：其他可能的诗词标题（后面有诗句或【注释】）
            if len(line) > 2 and len(line) < 40:
                look_ahead = '\n'.join(lines[i:min(i+15, len(lines))])
                # 检查是否有诗句特征（有标点）或【注释】
                if ('【注释】' in look_ahead or 
                    re.search(r'[，。；：！？]\s*$', look_ahead[:200]) or
                    re.search(r'^[^【《（]+[，。；：！？]', look_ahead[:100])):
                    # 避免重复识别场景
                    if not any(item.get("title") == line for item in items):
                        items.append({
                            "type": "poem",
                            "title": line,
                            "start_marker": line[:min(10, len(line))],
                            "scene_title": current_scene
                        })
            
            i += 1
        
        return items
    
    def extract_poems_with_context(self, text: str, structure: List[Dict]) -> List[Dict]:
        """基于识别的结构提取诗词，处理场景分组和跨诗引用"""
        logger.info(f"开始提取诗词，共识别 {len(structure)} 个结构项...")
        
        poems = []
        scene_info = {}  # 存储场景的说明信息
        
        # 先提取所有场景的说明
        for item in structure:
            if item.get("type") == "scene":
                scene_title = item.get("title", "")
                # 提取场景的【说明】
                scene_text = self._extract_section_by_marker(text, item.get("start_marker", ""))
                scene_explanation = self._extract_field(scene_text, "说明")
                if scene_explanation:
                    scene_info[scene_title] = scene_explanation
        
        # 提取每个诗词（使用大模型辅助提取）
        for idx, item in enumerate(structure):
            if item.get("type") != "poem":
                continue
            
            poem_title = item.get("title", "")
            scene_title = item.get("scene_title")
            
            # 提取该诗词的文本段（包含前后上下文以便处理引用）
            poem_text = self._extract_section_with_context(text, item, structure, idx)
            
            # 使用大模型提取诗词信息
            poem_info = self._extract_poem_with_llm(poem_text, poem_title, scene_title, scene_info)
            
            if poem_info and poem_info.get("内容"):
                poems.append(poem_info)
                logger.info(f"提取诗词: {poem_title}")
                time.sleep(1)  # 避免请求过快
        
        return poems
    
    def _extract_section_with_context(self, text: str, item: Dict, structure: List[Dict], idx: int) -> str:
        """提取诗词文本段，包含上下文以便处理引用"""
        marker = item.get("start_marker", "")
        if not marker:
            return ""
        
        start_idx = text.find(marker)
        if start_idx == -1:
            return ""
        
        # 提取当前诗词的文本（最多3000字符）
        current_text = text[start_idx:start_idx+3000]
        
        # 如果当前诗词的评解有引用，包含下一首诗词的评解
        if "见下" in current_text or "下题" in current_text:
            # 查找下一首诗词
            for next_idx in range(idx+1, len(structure)):
                if structure[next_idx].get("type") == "poem":
                    next_marker = structure[next_idx].get("start_marker", "")
                    next_start = text.find(next_marker)
                    if next_start != -1:
                        next_text = text[next_start:next_start+2000]
                        current_text += "\n\n--- 下一首诗词（用于解析评解引用）---\n" + next_text
                    break
        
        # 如果当前诗词属于场景，包含场景说明
        scene_title = item.get("scene_title")
        if scene_title:
            for scene_item in structure:
                if scene_item.get("type") == "scene" and scene_item.get("title") == scene_title:
                    scene_marker = scene_item.get("start_marker", "")
                    scene_start = text.find(scene_marker)
                    if scene_start != -1:
                        scene_text = text[scene_start:scene_start+1000]
                        current_text = f"--- 场景说明 ---\n{scene_text}\n\n--- 当前诗词 ---\n{current_text}"
                    break
        
        return current_text
    
    def _extract_poem_with_llm(self, text: str, poem_title: str, scene_title: str, scene_info: Dict) -> Optional[Dict]:
        """使用大模型提取诗词信息，处理场景分组和跨诗引用"""
        prompt = [
            {
                "role": "system",
                "content": """你是一位专业的古典诗词分析专家。请从给定的文本中提取诗词的完整信息。

重要说明：
1. 如果文本中包含"场景说明"，这是该场景下所有诗词的通用说明，应该包含在【说明】中
2. 如果【评解】中写有"见下题...评解"或类似引用，请查找文本中下一首诗词的【评解】内容，将其作为当前诗词的评解
3. 如果诗词属于某个场景（如"金陵十二钗图册判词"），场景的【说明】应该包含在诗词的【说明】中

请以JSON格式返回，格式如下：
{
    "标题": "诗词标题",
    "内容": "完整的诗词原文（诗句）",
    "说明": "背景说明（包含场景说明和诗词自身说明）",
    "注释": "对诗词中字词、典故的注释",
    "评解": "对诗词的赏析和评价（如果原文有引用，请解析引用）",
    "场景": "所属场景标题（如果有）"
}

如果文本中不包含完整的诗词信息，请返回null。"""
            },
            {
                "role": "user",
                "content": f"请从以下文本中提取诗词信息：\n\n{text[:5000]}"
            }
        ]
        
        # 解析失败时重试调用大模型，最多重试20次
        max_retries = 20
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                response, _ = call_LLM(prompt, self.model_name)
                
                # 解析JSON
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```(?:json)?\s*', '', response)
                    response = re.sub(r'\s*```$', '', response)
                
                try:
                    poem_info = json.loads(response)
                    if poem_info and isinstance(poem_info, dict) and poem_info.get("内容"):
                        # 确保标题和场景正确
                        if not poem_info.get("标题"):
                            poem_info["标题"] = poem_title
                        if scene_title:
                            poem_info["场景"] = scene_title
                            # 如果场景有说明但诗词说明中没有，添加场景说明
                            if scene_title in scene_info and scene_info[scene_title] not in poem_info.get("说明", ""):
                                if poem_info.get("说明"):
                                    poem_info["说明"] = f"{scene_info[scene_title]}\n\n{poem_info['说明']}"
                                else:
                                    poem_info["说明"] = scene_info[scene_title]
                        return poem_info
                except json.JSONDecodeError:
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{.*?"内容".*?\}', response, re.DOTALL)
                    if json_match:
                        try:
                            poem_info = json.loads(json_match.group())
                            if poem_info and isinstance(poem_info, dict) and poem_info.get("内容"):
                                return poem_info
                        except:
                            pass
                    
                    # 如果解析失败且不是最后一次尝试，记录警告并重试
                    if attempt < max_retries:
                        logger.warning(f"无法解析JSON响应（第{attempt}/{max_retries}次尝试）: {response[:200]}，将重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"无法解析JSON响应（已重试{max_retries}次）: {response[:200]}")
                        return None
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"提取诗词信息失败（第{attempt}/{max_retries}次尝试）: {e}，将重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"提取诗词信息失败（已重试{max_retries}次）: {e}")
                    return None
        
        return None
    
    def _extract_section_by_marker(self, text: str, marker: str) -> str:
        """根据标记提取文本段"""
        if not marker:
            return ""
        
        # 找到标记位置
        idx = text.find(marker)
        if idx == -1:
            return ""
        
        # 提取从标记开始到下一个标题或一定长度的文本
        section = text[idx:idx+3000]  # 限制长度
        
        # 尝试找到下一个诗词标题作为结束
        next_title_patterns = [
            r'\n[^【《（\n]+（第[^）]+回）',
            r'\n[^【《（\n]{3,30}\n',
        ]
        
        for pattern in next_title_patterns:
            match = re.search(pattern, section[500:])  # 跳过前500字符避免匹配到自己
            if match:
                section = section[:500+match.start()]
                break
        
        return section
    
    def _extract_poem_content(self, text: str) -> str:
        """提取诗词原文内容"""
        # 诗词内容通常在【说明】之前，或者在标题后直接出现
        # 尝试匹配诗句（通常有标点符号）
        lines = text.split('\n')
        poem_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 跳过标题行
            if i < 3 and ('（第' in line or '【说明】' in line):
                continue
            
            # 如果遇到【说明】、【注释】等标记，停止
            if re.match(r'^【(说明|注释|评解)】', line):
                break
            
            # 检查是否是诗句（包含常见标点或符合诗句特征）
            if re.search(r'[，。；：！？、]$', line) or len(line) <= 20:
                poem_lines.append(line)
        
        return '\n'.join(poem_lines).strip()
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """提取指定字段（说明、注释、评解）"""
        pattern = rf'【{field_name}】\s*(.*?)(?=【|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 清理内容
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content
        return ""
    
    def _resolve_review_reference(self, review: str, full_text: str, current_title: str) -> str:
        """解析评解引用（如"见下题...评解"）"""
        # 查找引用的目标
        if "下题" in review or "下" in review:
            # 查找当前标题后面的诗词
            current_idx = full_text.find(current_title)
            if current_idx != -1:
                # 在后面的文本中查找下一个诗词的评解
                next_text = full_text[current_idx + len(current_title):current_idx + 5000]
                next_review = self._extract_field(next_text, "评解")
                if next_review and "见" not in next_review:
                    return next_review
        
        if "上题" in review or "上" in review:
            # 查找当前标题前面的诗词
            current_idx = full_text.find(current_title)
            if current_idx != -1:
                prev_text = full_text[max(0, current_idx-5000):current_idx]
                prev_review = self._extract_field(prev_text, "评解")
                if prev_review and "见" not in prev_review:
                    return prev_review
        
        return review  # 如果无法解析，返回原文本
    
    def split_by_chapter_marker(self, text: str) -> List[Tuple[int, str, str]]:
        """通过（第x回）标记分割文本，并提取诗名"""
        lines = text.splitlines(True)  # 保留换行符，方便计算位置
        if not lines:
            return [(0, "", text)]

        # 预计算每行的起始下标
        line_starts = []
        cursor = 0
        for line in lines:
            line_starts.append(cursor)
            cursor += len(line)

        pattern = re.compile(r'（第[一二三四五六七八九十\d]+回）')

        def looks_like_title(raw_line: str) -> bool:
            line = raw_line.strip()
            if not line:
                return False
            if len(line) > 40:
                return False
            if any(line.startswith(prefix) for prefix in ("【", "（", "(", "第")):
                return False
            if any(keyword in line for keyword in ("说明", "注释", "评解")):
                return False
            # 避免纯序号
            if re.match(r'^[0-9一二三四五六七八九十]+\s*[、\.．)]', line):
                return False
            return True

        chunks: List[Tuple[int, str, str]] = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            match = pattern.fullmatch(stripped)
            if not match:
                continue

            chapter_marker = stripped

            # 向上查找最近的标题行
            title_line_idx = None
            for back in range(idx - 1, -1, -1):
                candidate = lines[back].strip()
                if not candidate:
                    continue
                if looks_like_title(candidate):
                    title_line_idx = back
                    break

            if title_line_idx is not None:
                poem_title = lines[title_line_idx].strip()
                chunk_start_idx = line_starts[title_line_idx]
            else:
                poem_title = chapter_marker
                chunk_start_idx = line_starts[idx]

            # 结束位置：找到下一个章节标记的起始位置
            next_start_idx = len(text)
            for forward in range(idx + 1, len(lines)):
                next_line = lines[forward].strip()
                if pattern.fullmatch(next_line):
                    next_start_idx = line_starts[forward]
                    break

            chunk_text = text[chunk_start_idx:next_start_idx].strip()
            if chunk_text:
                chunks.append((chunk_start_idx, poem_title, chunk_text))

        logger.info(f"通过（第x回）标记分割为 {len(chunks)} 个片段")
        if not chunks:
            logger.warning("未切分出任何片段，返回全文")
            return [(0, "", text)]
        return chunks
    
    def extract_poem_from_chunk(self, chunk: str, poem_title: str = "") -> Optional[Dict]:
        """从文本片段中提取诗词信息
        
        Args:
            chunk: 文本片段
            poem_title: 诗名（（第x回）的上一行）
        """
        # 构建包含诗名的提示
        title_hint = f"\n\n注意：这首诗词的标题应该是：{poem_title}\n" if poem_title else ""
        
        prompt = [
            {
                "role": "system",
                "content": """你是一位专业的古典诗词分析专家。请从给定的文本中提取诗词的完整信息。

重要说明：
1. 如果文本中包含"场景说明"或场景标题（如"金陵十二钗图册判词"），这是该场景下所有诗词的通用说明，应该包含在【说明】中
2. 如果【评解】中写有"见下题...评解"或类似引用，请查找文本中下一首诗词的【评解】内容，将其作为当前诗词的评解
3. 如果文本中包含多首诗词，请只提取第一首完整的诗词信息
4. 请特别注意用户提供的诗名，确保标题准确

请以JSON格式返回，格式如下：
{
    "标题": "诗词标题（包含（第X回），如果用户提供了诗名，请使用该诗名）",
    "内容": "完整的诗词原文（诗句）",
    "说明": "背景说明（包含场景说明和诗词自身说明）",
    "注释": "对诗词中字词、典故的注释",
    "评解": "对诗词的赏析和评价（如果原文有引用，请解析引用）",
    "场景": "所属场景标题（如果有，如'金陵十二钗图册判词'）"
}

如果文本中不包含完整的诗词信息，请返回null。只返回JSON，不要其他说明。"""
            },
            {
                "role": "user",
                "content": f"请从以下文本中提取诗词信息：{title_hint}\n\n{chunk[:6000]}"
            }
        ]
        
        # 解析失败时重试调用大模型，最多重试20次
        max_retries = 20
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                response, _ = call_LLM(prompt, self.model_name)
                
                # 解析JSON
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```(?:json)?\s*', '', response)
                    response = re.sub(r'\s*```$', '', response)
                
                try:
                    poem_info = json.loads(response)
                    if poem_info and isinstance(poem_info, dict) and poem_info.get("内容"):
                        return poem_info
                except json.JSONDecodeError:
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{.*?"内容".*?\}', response, re.DOTALL)
                    if json_match:
                        try:
                            poem_info = json.loads(json_match.group())
                            if poem_info and isinstance(poem_info, dict) and poem_info.get("内容"):
                                return poem_info
                        except:
                            pass
                    
                    # 如果解析失败且不是最后一次尝试，记录警告并重试
                    if attempt < max_retries:
                        logger.warning(f"无法解析JSON响应（第{attempt}/{max_retries}次尝试）: {response[:200]}，将重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"无法解析JSON响应（已重试{max_retries}次）: {response[:200]}")
                        return None
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"提取诗词信息失败（第{attempt}/{max_retries}次尝试）: {e}，将重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"提取诗词信息失败（已重试{max_retries}次）: {e}")
                    return None
        
        return None
    
    def save_poem_to_file(self, poem_info: Dict):
        """将提取的诗词保存到JSON文件（如果已存在则更新，否则添加）"""
        if not self.output_file:
            return
        
        try:
            # 读取现有数据
            if self.output_file.exists():
                with open(self.output_file, "r", encoding="utf-8") as f:
                    poems = json.load(f)
            else:
                poems = []
            
            # 检查是否已存在（优先使用 id，其次使用标题）
            poem_id = poem_info.get("id")
            title = poem_info.get("标题", "")
            found = False
            
            for i, existing_poem in enumerate(poems):
                existing_id = existing_poem.get("id")
                existing_title = existing_poem.get("标题", "")
                
                # 匹配逻辑：优先使用 id，其次使用标题
                if poem_id is not None and existing_id is not None and poem_id == existing_id:
                    poems[i] = poem_info  # 更新已存在的诗词
                    found = True
                    break
                elif title and existing_title and title == existing_title:
                    poems[i] = poem_info  # 更新已存在的诗词
                    found = True
                    break
            
            if not found:
                poems.append(poem_info)  # 添加新诗词
            
            # 写回文件
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(poems, f, ensure_ascii=False, indent=2)
            
            action = "更新" if found else "添加"
            logger.info(f"已{action}诗词到文件: {poem_info.get('标题', '未知标题')}")
        except Exception as e:
            logger.error(f"保存诗词到文件失败: {e}")
    
    def extract_all_poems(self, batch_size: int = 10, max_workers: int = 5) -> List[Dict]:
        """提取所有诗词信息（支持断点续跑）
        
        - 如果输入为 hongloumeng.json：直接读取结构化数据
        - 否则：走旧版从文本/PDF中解析（通过（第x回）标记分割，支持并发）
        """
        # 1) 新流程：直接从 hongloumeng.json 读取
        if self.pdf_path.suffix.lower() == ".json":
            logger.info(f"从结构化JSON文件读取诗词: {self.pdf_path}")
            try:
                # 先检查是否已有提取结果
                existing_poems = {}
                if self.output_file and self.output_file.exists():
                    try:
                        with open(self.output_file, "r", encoding="utf-8") as f:
                            existing_list = json.load(f)
                            # 建立索引：以 id 或标题为键
                            for poem in existing_list:
                                poem_id = poem.get("id")
                                title = poem.get("标题", "")
                                if poem_id is not None:
                                    existing_poems[poem_id] = poem
                                elif title:
                                    existing_poems[title] = poem
                        logger.info(f"检测到已存在 {len(existing_poems)} 首诗词，将跳过已提取的")
                    except Exception as e:
                        logger.warning(f"读取已有提取结果失败，将重新提取: {e}")
                
                with open(self.pdf_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results = []
                skipped_count = 0
                for idx, item in enumerate(tqdm(data, desc="加载结构化诗词", unit="首"), 1):
                    title = item.get("title", "").strip()
                    poet = item.get("poet", "").strip()
                    author = item.get("author", "").strip()
                    background = item.get("background", "").strip()
                    poet_analyse = item.get("poet_analyse", "").strip()
                    poem_id = item.get("id")
                    
                    if not poet:
                        continue
                    
                    # 检查是否已存在（优先使用 id，其次使用标题）
                    existing_poem = None
                    if poem_id is not None and poem_id in existing_poems:
                        existing_poem = existing_poems[poem_id]
                    elif title and title in existing_poems:
                        existing_poem = existing_poems[title]
                    
                    # 说明：使用作者赏析 + 背景 + 作者信息，供后续分析/生成树使用
                    explain_parts = []
                    if poet_analyse:
                        explain_parts.append(f"【作者赏析】\n{poet_analyse}")
                    if background:
                        explain_parts.append(f"【作诗背景】\n{background}")
                    if author:
                        explain_parts.append(f"【作者】\n{author}")
                    explain_text = "\n\n".join(explain_parts)
                    
                    poem_info = {
                        "标题": title or f"ID_{poem_id}",
                        "内容": poet,
                        "说明": explain_text,
                        "注释": "",
                        "评解": "",
                        "场景": "",
                        "id": poem_id,
                        "author": author,
                        "background": background,
                        "poet_analyse": poet_analyse,
                        # 初始 poet_id 为原始 id 的字符串，后续生成树时扩展
                        "poet_id": str(poem_id) if poem_id is not None else str(idx),
                    }
                    
                    # 如果已存在且内容完整，使用已有数据；否则使用新数据并保存
                    if existing_poem:
                        # 检查是否已有完整内容（至少要有"内容"字段）
                        if existing_poem.get("内容") and existing_poem.get("内容").strip():
                            results.append(existing_poem)
                            skipped_count += 1
                            continue
                        else:
                            # 已存在但内容不完整，使用新数据更新
                            results.append(poem_info)
                            # 立即保存到文件，更新已有记录
                            self.save_poem_to_file(poem_info)
                    else:
                        # 不存在，添加新数据
                        results.append(poem_info)
                        # 立即保存到文件，便于断点续跑
                        self.save_poem_to_file(poem_info)
                
                self.extracted_poems = results
                logger.info(f"从JSON加载完成，共 {len(self.extracted_poems)} 首诗词（跳过 {skipped_count} 首已提取的）")
                return self.extracted_poems
            except Exception as e:
                logger.error(f"从JSON读取诗词失败: {e}")
                raise
        
        # 2) 旧流程：从文本/PDF中解析
        logger.info("开始提取PDF/文本中的诗词...")
        full_text = self.extract_text_from_pdf()
        
        logger.info("开始通过（第x回）标记分割文本...")
        chunks = self.split_by_chapter_marker(full_text)
        
        logger.info(f"开始提取诗词信息，共 {len(chunks)} 个片段，并发数: {max_workers}")
        
        def process_chunk(args):
            """处理单个片段"""
            i, (start_idx, poem_title, chunk) = args
            try:
                poem_info = self.extract_poem_from_chunk(chunk, poem_title)
                if poem_info:
                    # 如果模型没有正确提取标题，使用我们提取的诗名
                    if not poem_info.get("标题") or poem_info.get("标题") == "（第x回）":
                        poem_info["标题"] = poem_title
                    
                    poem_info["片段索引"] = i
                    poem_info["提取的诗名"] = poem_title
                    return poem_info
            except Exception as e:
                logger.error(f"处理片段 {i} ({poem_title}) 失败: {e}")
            return None
        
        # 并发处理
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_chunk, (i, chunk_data)): (i, chunk_data)
                for i, chunk_data in enumerate(chunks, 1)
            }
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="提取诗词", unit="段"):
                poem_info = future.result()
                if poem_info:
                    results.append(poem_info)
                    # 立即保存到文件
                    self.save_poem_to_file(poem_info)
                    logger.info(f"成功提取诗词: {poem_info.get('标题', '未知标题')}")
        
        # 按片段索引排序
        results.sort(key=lambda x: x.get("片段索引", 0))
        self.extracted_poems = results
        
        logger.info(f"提取完成，共提取到 {len(self.extracted_poems)} 首诗词")
        return self.extracted_poems


class PoetryAnalyzer:
    """分析诗词的妙处"""
    
    def __init__(self, model_name: str = "dsv3"):
        self.model_name = model_name
    
    def analyze_merits(self, poem_info: Dict) -> Dict[str, str]:
        """分析诗词的所有妙处"""
        content = poem_info.get("内容", "")
        explanation = '【说明】'+str({poem_info.get("说明", "")}) if poem_info.get("说明", "") else ""
        annotation = '【注释】'+str({poem_info.get("注释", "")}) if poem_info.get("注释", "") else ""
        review = '【评解】'+str({poem_info.get("评解", "")}) if poem_info.get("评解", "") else ""
        
        prompt = [
            {
                "role": "system",
                "content": """你是一位造诣深厚的古典诗词鉴赏专家及文学批评家。你的任务是深入研读给定的诗词文本及其相关的【说明】，从中提炼并分析这首诗词的艺术特色（即“妙处”）。

### 核心任务
请识别诗词中5个左右的最突出的妙处。如果提供的专家解析中有明确指出的妙处，请优先采用；若专家解析未覆盖，请基于你的专业知识进行补充分析。

### 妙处分类与定义（参考标准）
以下是一些常见的妙处示例，但不仅限于此：
1. **音韵和谐**：分析平仄、押韵、双声叠韵或节奏感对诗歌音乐美的贡献。
2. **字词锤炼**：捕捉诗中极具表现力的动词、形容词或虚词（即“诗眼”），分析其如何化静为动、点石成金。
3. **修辞手法**：识别比喻、拟人、对仗、夸张、通感、互文等修辞，并解释其增强了何种艺术效果。
4. **意境营造**：分析诗歌通过景物描写构建了怎样的画面与氛围（如凄清、雄浑、恬淡），以及这种氛围如何服务于主题。
5. **情感表达**：分析诗人抒情的方式（直抒胸臆、借景抒情、托物言志、欲扬先抑等）及其情感的层次变化。
6. **结构安排**：分析起承转合、首尾呼应、铺垫照应、卒章显志等篇章布局的精妙之处。
7. **意象妙用**：分析诗人如何选取特定事物（如孤雁、残月）来寄托情思，或通过意象组合（蒙太奇）产生新的隐喻义。
8. **典故运用**：指出诗中引用的历史典故或前人诗句，解释其如何拓展诗歌的时空深度或含蓄地表达观点。
9. **巧妙留白**：分析诗歌结尾或句间留下的想象空间，即“言有尽而意无穷”之处。
……

### 输出要求
1. **格式**：必须返回一个标准的 JSON 字典格式。
2. **键名**：使用“妙处名称”作为Key。
3. **键值**：作为Value的解释必须具体、深入。
   - **必须引用**：解释中必须包含诗词原句或关键词。
   - **结合分析**：结合提供的内容，说明该妙处为何“妙”。
4. **真实性**：严禁虚构不存在的妙处。

### JSON 返回示例
{
    "音韵和谐": "...",
    "字词锤炼": "...",
    ...
}"""
            },
            {
                "role": "user",
                "content": f"""请分析以下诗词的所有妙处：

{content}

{explanation}

{annotation}

{review}

请以JSON格式返回所有妙处。"""
            }
        ]
        
        # 解析失败时重试调用大模型，最多重试20次
        max_retries = 20
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                response, _ = call_LLM(prompt, self.model_name)
                
                # 解析JSON
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```(?:json)?\s*', '', response)
                    response = re.sub(r'\s*```$', '', response)
                
                try:
                    merits = json.loads(response)
                    if isinstance(merits, dict) and len(merits) > 0:
                        return merits
                except json.JSONDecodeError:
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        try:
                            merits = json.loads(json_match.group())
                            if isinstance(merits, dict) and len(merits) > 0:
                                return merits
                        except:
                            pass
                
                # 如果解析失败且不是最后一次尝试，记录警告并重试
                if attempt < max_retries:
                    logger.warning(f"无法解析妙处JSON（第{attempt}/{max_retries}次尝试）: {response[:200]}，将重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"无法解析妙处JSON（已重试{max_retries}次）: {response[:200]}")
                    return {}
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"分析妙处失败（第{attempt}/{max_retries}次尝试）: {e}，将重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"分析妙处失败（已重试{max_retries}次）: {e}")
                    return {}
        
        return {}


class PoetryTreeGenerator:
    """生成诗词树（逐步削弱优势）"""
    
    def __init__(self, model_name: str = "dsv3", use_cache: bool = True, max_children_per_level: Optional[int] = None):
        self.model_name = model_name
        self.tree = {}  # 存储诗词树结构
        self.use_cache = use_cache  # 是否启用缓存，避免重复生成
        self.max_children_per_level = max_children_per_level  # 每层最多子节点数（随机采样）
    
    def _extract_chinese_chars(self, text: str) -> List[str]:
        """提取文本中的所有中文字符（去除标点符号和空白）"""
        if not text:
            return []
        # 去除标点符号、空白字符、换行符
        cleaned = re.sub(r'[，。！？；：、\s\n\r\t]', '', text)
        # 提取所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', cleaned)
        return chinese_chars
    
    def _count_different_chars(self, parent_poem: str, child_poem: str) -> int:
        """计算两个诗歌之间有多少个不同的字（按位置比较）"""
        parent_chars = self._extract_chinese_chars(parent_poem)
        child_chars = self._extract_chinese_chars(child_poem)
        
        # 按位置比较，统计不同的字
        min_len = min(len(parent_chars), len(child_chars))
        different_count = 0
        
        for i in range(min_len):
            if parent_chars[i] != child_chars[i]:
                different_count += 1
        
        return different_count
    def _count_different_chars_difflib(self, text1: str, text2: str) -> int:
        """
        使用difflib计算文本差异
        """
        import difflib
        
        # 提取中文字符
        import re
        def extract_chinese(text):
            return re.findall(r'[\u4e00-\u9fff]', text)
        
        ch1 = extract_chinese(text1)
        ch2 = extract_chinese(text2)
        
        # 创建差异匹配器
        matcher = difflib.SequenceMatcher(None, ch1, ch2)
        
        # 计算差异
        total_diff = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # 替换的字符数
                total_diff += max((i2 - i1), (j2 - j1))
            elif tag == 'delete':
                # 删除的字符数
                total_diff += (i2 - i1)
            elif tag == 'insert':
                # 插入的字符数
                total_diff += (j2 - j1)
            # 'equal' 不需要处理
        
        return total_diff
    
    def _calculate_max_diff_chars(self, poem: str) -> int:
        """
        计算诗歌允许的最大不同字符数（中文字数的15%，向上取整）
        
        Args:
            poem: 诗歌文本
            
        Returns:
            最大允许的不同字符数
        """
        chinese_chars = self._extract_chinese_chars(poem)
        chinese_char_count = len(chinese_chars)
        max_diff = int(math.ceil(chinese_char_count * 0.15))
        return max_diff
    
    def _check_poem_validity(self, parent_poem: str, child_poem: str, max_diff_chars: Optional[int] = None) -> Tuple[bool, str]:
        """
        检查子节点和父节点是否符合要求
        
        Args:
            parent_poem: 父节点诗歌
            child_poem: 子节点诗歌
            max_diff_chars: 最大允许的不同字符数（如果为None，则使用15%规则）
        
        Returns:
            (is_valid, error_message): 是否有效，错误信息
        """
        # 检查是否相同（去除空白后比较）
        if parent_poem.strip() == child_poem.strip():
            return False, "子节点和父节点不能相同"
        
        # 计算不同的字数
        # different_count = self._count_different_chars(parent_poem, child_poem)
        different_count = self._count_different_chars_difflib(parent_poem, child_poem)
        
        # 如果没有指定最大差异数，则使用15%规则
        if max_diff_chars is None:
            max_diff_chars = self._calculate_max_diff_chars(parent_poem)
        
        if different_count > max_diff_chars:
            return False, f"子节点和父节点有{different_count}个不同的字，超过了{max_diff_chars}个字的限制（原诗中文字数的15%）"
        
        return True, ""
    
    
    def generate_weakened_poem(
        self, 
        original_poem: str, 
        all_merits: Dict[str, str], 
        removed_merits: Dict[str, str],
        remaining_merits: Dict[str, str],
        background: str = "",
        additional_reminder: str = "",
        max_diff_chars: Optional[int] = None,
    ) -> Tuple[str, str]:
        """基于原诗和剩余优势生成削弱后的诗，并返回解释
        
        Args:
            original_poem: 基础诗词（通常是父节点的诗词）
            all_merits: 原诗的所有优势
            removed_merits: 需要移除的优势字典 {名称: 描述}
            remaining_merits: 需要保留的优势字典 {名称: 描述}
            background: 写作背景简介（可选，包含说明/评解等）
            additional_reminder: 额外的提醒文本
            max_diff_chars: 最大允许的不同字符数（如果为None，则使用15%规则）
        
        Returns:
            (poem, explanation) 元组，poem是新生成的诗词，explanation是解释说明
        """
        
        # 计算最大允许的不同字符数
        if max_diff_chars is None:
            max_diff_chars = self._calculate_max_diff_chars(original_poem)
        
        removed_merits_desc = "\n".join([
            f"- {name}: {desc}" for name, desc in removed_merits.items()
        ])
        
        remaining_merits_desc = "\n".join([
            f"- {name}: {desc}" for name, desc in remaining_merits.items()
        ])
        background_desc = background.strip() if isinstance(background, str) else ""
        
        # 准备额外提醒文本（避免在 f-string 中使用反斜杠）
        reminder_text = f"{additional_reminder}\n" if additional_reminder else ""
        
        prompt = [
            {
                "role": "system",
                "content": f"""你是一位优秀的古典诗词创作专家。请根据要求重新创作诗词。
要求：
1. 保持原诗的主题和基本意境
2. 保留指定的优势（妙处）
3. 去除指定的优势（妙处），使诗词在该方面有所削弱（只通过尽可能少的字眼修改实现削弱，最多不超过{max_diff_chars}个字，以确保诗歌保持其他优势）
4. 保持诗词的完整性和可读性
5. 保持相同的体裁和格律（如果原诗有格律要求）
6. 尽量保持与原诗相同的时代背景、人物关系和叙事视角

重要说明：
- 你会得到父节点诗词及其具备的所有优势
- 你需要去除一个指定的优势，保留其他所有优势
- 这是逐步削弱的过程，每次只去除一个优势

请以JSON格式返回，包含：
1. "poem": 新创作的，被削弱后的诗词内容
2. "explanation": 详细解释说明：
   - 如何去除/削弱了要求去除的优势（具体说明在哪些方面、通过什么方式削弱）
   - 如何保留/体现了需要保留的优势（具体说明在哪些方面、通过什么方式体现）

格式示例：
{{
    "poem": "新创作的，被削弱后的诗词内容",
    "explanation": "详细的str解释，不要输出字典等其他格式"
}}"""
            },
            {
                "role": "user",
                "content": f"""父节点诗词（作为基础）：
{original_poem}

父节点具备的所有优势：
{json.dumps(all_merits, ensure_ascii=False, indent=2)}

需要保留的优势（父节点优势减去本次要移除的）：
{remaining_merits_desc}

本次要去除的优势（仅此一个，需要削弱这个方面）：
{removed_merits_desc}

写作背景与人物关系简介（仅供你理解，请勿在新诗中直白复述背景文字）：
{background_desc}
{reminder_text}请重新创作一首诗词，保持保留的优势，但削弱或去除指定的优势。并以JSON格式返回被削弱后的新诗词和解释说明。"""
            }
        ]
        
        # 解析失败时重试调用大模型，最多重试20次
        max_retries = 20
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                response, _ = call_LLM(prompt, self.model_name)
                response = response.strip()
                
                # 尝试解析JSON
                if response.startswith("```"):
                    response = re.sub(r'^```(?:json)?\s*', '', response)
                    response = re.sub(r'\s*```$', '', response)
                
                try:
                    result = json.loads(response)
                    poem = result.get("poem", "").strip()
                    explanation = result.get("explanation", "").strip()
                    
                    # 清理诗词内容
                    poem = re.sub(r'^["\'「」『』]', '', poem)
                    poem = re.sub(r'["\'「」『』]$', '', poem)
                    
                    if not poem:
                        if attempt < max_retries:
                            logger.warning(f"模型返回的诗词内容为空（第{attempt}/{max_retries}次尝试），将重试...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.warning("模型返回的诗词内容为空（已重试{max_retries}次）")
                            return "", explanation
                    
                    return poem, explanation
                except json.JSONDecodeError:
                    # 如果无法解析JSON，尝试提取诗词和解释
                    json_match = re.search(r'\{.*?"poem".*?"explanation".*?\}', response, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            poem = result.get("poem", "").strip()
                            explanation = result.get("explanation", "").strip()
                            if poem:
                                poem = re.sub(r'^["\'「」『』]', '', poem)
                                poem = re.sub(r'["\'「」『』]$', '', poem)
                                return poem, explanation
                        except:
                            pass
                    
                    # 如果解析失败且不是最后一次尝试，记录警告并重试
                    if attempt < max_retries:
                        logger.warning(f"无法解析JSON响应（第{attempt}/{max_retries}次尝试）: {response[:200]}，将重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # 如果还是无法解析，返回原始响应作为诗词，解释为空
                        logger.warning(f"无法解析JSON（已重试{max_retries}次），使用原始响应: {response[:200]}")
                        poem = response.strip()
                        poem = re.sub(r'^["\'「」『』]', '', poem)
                        poem = re.sub(r'["\'「」『』]$', '', poem)
                        return poem, ""
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"生成削弱诗词失败（第{attempt}/{max_retries}次尝试）: {e}，将重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"生成削弱诗词失败（已重试{max_retries}次）: {e}")
                    return "", ""
        
        return "", ""
    
    def generate_tree(
        self, 
        original_poem: str, 
        all_merits: Dict[str, str],
        max_depth: Optional[int] = None,
        background: str = "",
        base_id: Optional[str] = None,
        existing_tree: Optional[Dict] = None,
    ) -> Dict:
        """生成完整的诗词树
        
        树结构：每个节点代表删除某些优势后的诗
        例如有5个优势[A,B,C,D,E]：
        - 根节点：原诗（保留所有优势）
        - 第1层：删除A、删除B、删除C、删除D、删除E（5个节点）
        - 第2层：删除AB、删除AC、...（每个第1层节点生成4个子节点）
        - 以此类推
        
        Args:
            original_poem: 原诗内容
            all_merits: 所有妙处的字典
            max_depth: 最大深度（None表示生成所有可能的组合）
            background: 背景信息
            base_id: 基础ID
            existing_tree: 已有的树结构（用于断点重跑）
        
        Returns:
            诗词树结构，格式：
            {
                "poem": "诗词内容",
                "merits": {"保留的妙处": "描述"},
                "removed": ["被移除的妙处"],
                "children": [子节点列表]
            }
        """
        merit_names = list(all_merits.keys())
        n = len(merit_names)
        
        if max_depth is None:
            max_depth = n
        
        # 计算最大允许的不同字符数（15%规则）
        max_diff_chars = self._calculate_max_diff_chars(original_poem)
        
        logger.info(f"开始生成诗词树，共有 {n} 个优势，最大深度 {max_depth}，是否使用缓存: {self.use_cache}")
        logger.info(f"优势列表: {merit_names}")
        logger.info(f"最大允许不同字符数: {max_diff_chars}（原诗中文字数的15%，向上取整）")
        
        # 缓存已生成的诗词，避免重复生成（可选）
        # 缓存格式: {cache_key: {"poem": str, "explanation": str}}
        poem_cache = {} if self.use_cache else None
        
        # 从已有树中构建节点缓存（用于断点重跑）
        # 只缓存 poet_id，用于快速判断是否已存在，不递归读取子树
        existing_poet_ids: set = set()
        if existing_tree:
            def extract_existing_poet_ids(node: Dict):
                """递归提取已有节点的 poet_id（不读取子树内容）"""
                poet_id = node.get("poet_id")
                if poet_id:
                    existing_poet_ids.add(poet_id)
                # 递归提取所有子节点的 poet_id
                for child in node.get("children", []):
                    extract_existing_poet_ids(child)
            
            extract_existing_poet_ids(existing_tree)
            logger.info(f"从已有树中检测到 {len(existing_poet_ids)} 个已存在的 poet_id，将跳过这些节点")
        
        def get_cache_key(removed: List[str]) -> str:
            """生成缓存键"""
            return ",".join(sorted(removed))
        
        def build_node(remaining_merits: Dict[str, str], removed_merits: List[str], depth: int, parent_poem: str, current_poet_id: str) -> Dict:
            """递归构建节点
            
            Args:
                remaining_merits: 当前节点保留的优势
                removed_merits: 当前节点已移除的优势列表
                depth: 当前深度
                parent_poem: 父节点的诗词内容（用于生成当前节点）
            """
            cache_key = get_cache_key(removed_merits)
            
            # 检查 poet_id 是否已存在，如果存在则直接跳过（不生成子树）
            if current_poet_id in existing_poet_ids:
                logger.info(f"跳过已存在的节点 poet_id={current_poet_id}，移除 {removed_merits}")
                # 返回一个占位节点，不生成子树
                return {
                    "poem": "",  # 占位，实际不会使用
                    "explanation": "",
                    "removed": removed_merits.copy(),
                    "id": base_id,
                    "poet_id": current_poet_id,
                    "children": []
                }
            
            # 如果这是根节点，使用原诗，并在根节点保留完整的 merits
            if len(removed_merits) == 0:
                poem_content = original_poem
                explanation = "这是原诗，保留所有优势。"
                node = {
                    "poem": poem_content,
                    "explanation": explanation,
                    "merits": all_merits.copy(),  # 仅根节点需要保留全部妙处信息
                    "removed": [],
                    "id": base_id,
                    "poet_id": current_poet_id,
                    "children": []
                }
            else:
                # 是否启用缓存
                if self.use_cache and poem_cache is not None and cache_key in poem_cache:
                    cached_result = poem_cache[cache_key]
                    poem_content = cached_result["poem"]
                    explanation = cached_result.get("explanation", "")
                    # logger.info(f"使用缓存: 移除 {removed_merits}")
                else:
                    # 生成削弱后的诗（基于父节点的诗）
                    # 计算父节点具备的优势 = 剩余优势 + 本次要移除的优势
                    # 注意：removed_merits 包含所有已移除的优势，我们需要找到本次要移除的那个
                    # 父节点具备的优势 = remaining_merits + (removed_merits 中最后一个，即本次要移除的)
                    if removed_merits:
                        # 本次要移除的优势是 removed_merits 中的最后一个
                        current_removed_merit = removed_merits[-1]
                        # 父节点具备的优势 = 剩余优势 + 本次要移除的优势
                        parent_merits = remaining_merits.copy()
                        parent_merits[current_removed_merit] = all_merits[current_removed_merit]
                        # 本次要移除的优势（只有一个）
                        current_removed_dict = {current_removed_merit: all_merits[current_removed_merit]}
                    else:
                        # 这种情况不应该发生（非根节点一定有 removed_merits）
                        parent_merits = remaining_merits.copy()
                        current_removed_dict = {}
                    
                    # 生成诗歌，如果不符合要求则重试（最多重试5次）
                    max_validation_retries = 5
                    additional_reminder = ""
                    for validation_attempt in range(max_validation_retries):
                        poem_content, explanation = self.generate_weakened_poem(
                            parent_poem,  # 使用父节点的诗作为基础
                            parent_merits,  # 父节点具备的优势（不是根节点的所有优势）
                            current_removed_dict,  # 本次要移除的优势（只有一个）
                            remaining_merits,  # 需要保留的优势（父节点优势 - 本次要移除的）
                            background='【作诗背景】'+background.split('【作诗背景】')[-1],
                            additional_reminder=additional_reminder,
                            max_diff_chars=max_diff_chars,
                        )
                        
                        # 检查生成的诗歌是否符合要求
                        is_valid, error_msg = self._check_poem_validity(parent_poem, poem_content, max_diff_chars)
                        
                        if is_valid:
                            # 符合要求，退出重试循环
                            break
                        else:
                            # 不符合要求，添加提醒并重试
                            logger.warning(f"生成的诗歌不符合要求（第{validation_attempt + 1}/{max_validation_retries}次尝试）: {error_msg}")
                            logger.warning(f"父节点诗词：{parent_poem}")
                            logger.warning(f"生成的诗词：{poem_content}")
                            additional_reminder = f"\n\n**重要提醒**：{error_msg}。请确保新生成的诗词与父节点诗词不同，且最多只有{max_diff_chars}个不同的字（原诗中文字数的15%，向上取整）。\n父节点诗词：{parent_poem}\n上次生成的不合格诗词：{poem_content}\n"
                            
                            if validation_attempt < max_validation_retries - 1:
                                logger.info(f"将重新生成诗歌...")
                                time.sleep(1)  # 短暂延迟后重试
                            else:
                                logger.error(f"已重试{max_validation_retries}次，仍不符合要求，使用当前结果")
                    
                    if self.use_cache and poem_cache is not None:
                        poem_cache[cache_key] = {
                            "poem": poem_content,
                            "explanation": explanation
                        }
                    # logger.info(f"生成新诗: 移除 {removed_merits}, 保留 {list(remaining_merits.keys())}")
                    time.sleep(2)  # 避免请求过快

                # 子节点不再重复存储 merits / removed_names，仅保留 removed 即可
                node = {
                    "poem": poem_content,
                    "explanation": explanation,
                    "removed": removed_merits.copy(),
                    "id": base_id,
                    "poet_id": current_poet_id,
                    "children": []
                }
            
            # 如果达到最大深度或没有剩余优势，停止递归
            if depth >= max_depth or len(remaining_merits) == 0:
                return node
            
            # 为每个剩余的优势生成子节点（移除该优势）——为避免线程过多，这里使用顺序生成
            children = []
            merit_names_list = list(remaining_merits.keys())
            if self.max_children_per_level is not None and len(merit_names_list) > self.max_children_per_level:
                merit_names_list = random.sample(merit_names_list, self.max_children_per_level)

            for idx, merit_name in enumerate(merit_names_list, 1):
                new_remaining = remaining_merits.copy()
                new_removed = removed_merits + [merit_name]
                del new_remaining[merit_name]
                # 子节点 poet_id 在父节点基础上增加序号，如 2-1, 2-2, 2-1-1 等
                child_poet_id = f"{current_poet_id}-{idx}" if current_poet_id else str(idx)
                
                # 生成新的子节点（build_node 内部会检查 poet_id 是否已存在并跳过）
                child = build_node(new_remaining, new_removed, depth + 1, poem_content, child_poet_id)
                children.append(child)
            node["children"] = children
            
            return node
        
        root_base_id = base_id or "unknown"
        root_poet_id = str(root_base_id)
        root = build_node(all_merits.copy(), [], 0, original_poem, root_poet_id)
        
        # 统计树节点数量
        def count_nodes(node: Dict) -> int:
            count = 1
            for child in node.get("children", []):
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(root)
        logger.info(f"诗词树生成完成，共 {total_nodes} 个节点")
        
        return root


class PoetryAnalysisPipeline:
    """完整的诗词分析流程（支持增量写入，统一并行度控制）"""
    
    def __init__(
        self, 
        pdf_path: str,
        extract_model: str = "dsv3",
        analyze_model: str = "dsv3",
        generate_model: str = "dsv3",
        output_dir: str = "output",
        use_cache_for_tree: bool = True, #是否缓存
        max_workers: int = 5,            # 统一并行度
        max_children_per_level: Optional[int] = None,  # 每层子节点数量上限（随机采样）
    ):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置输出文件路径
        extracted_poems_file = self.output_dir / "extracted_poems.json"
        final_results_file = self.output_dir / f"final_results_{generate_model}.json"
        token_cost_file = self.output_dir / f"token_cost_summary_{generate_model}.json"
        
        # 初始化各个组件，传入输出文件
        self.extractor = PoetryExtractor(pdf_path, extract_model, str(extracted_poems_file))
        self.analyzer = PoetryAnalyzer(analyze_model)
        self.generator = PoetryTreeGenerator(
            generate_model,
            use_cache=use_cache_for_tree,
            max_children_per_level=max_children_per_level,
        )
        
        # 保存文件路径
        self.extracted_poems_file = extracted_poems_file
        self.final_results_file = final_results_file
        self.token_cost_file = token_cost_file
        self.max_workers = max_workers
        self.max_children_per_level = max_children_per_level
    
    def update_poem_in_extracted_file(self, poem_data: Dict):
        """更新extracted_poems.json中的诗词数据（优先使用ID匹配，其次使用标题）
        
        只更新"妙处"字段，保留原有数据，避免覆盖其他字段
        """
        try:
            if not self.extracted_poems_file.exists():
                logger.warning(f"extracted_poems.json 不存在，无法更新")
                return
            
            # 读取现有数据
            with open(self.extracted_poems_file, "r", encoding="utf-8") as f:
                poems = json.load(f)
            
            # 更新或添加诗词（优先使用ID匹配，其次使用标题）
            poem_id = poem_data.get("id")
            poem_title = poem_data.get("标题", "")
            merits = poem_data.get("妙处")
            found = False
            
            if not merits:
                logger.warning(f"要更新的诗词数据中没有'妙处'字段，跳过更新")
                return
            
            for i, item in enumerate(poems):
                # 优先使用ID匹配
                if poem_id is not None and item.get("id") == poem_id:
                    # 只更新"妙处"字段，保留原有数据
                    poems[i]["妙处"] = merits
                    # 同时更新其他可能变化的字段（如果存在）
                    if "标题" in poem_data:
                        poems[i]["标题"] = poem_data["标题"]
                    found = True
                    logger.debug(f"已更新ID {poem_id} 的妙处字段，共 {len(merits)} 个妙处")
                    break
                # 如果没有ID或ID不匹配，使用标题匹配
                elif poem_title and item.get("标题") == poem_title:
                    # 只更新"妙处"字段，保留原有数据
                    poems[i]["妙处"] = merits
                    # 如果原诗词没有ID但新数据有，也更新ID
                    if poem_id is not None and item.get("id") is None:
                        poems[i]["id"] = poem_id
                    found = True
                    logger.debug(f"已更新标题 '{poem_title}' 的妙处字段，共 {len(merits)} 个妙处")
                    break
            
            poem_identifier = f"ID_{poem_id}" if poem_id is not None else poem_title
            
            if not found:
                # 如果没有找到匹配的，添加新诗词
                poems.append(poem_data)
                logger.info(f"未找到匹配的诗词，已添加新诗词: {poem_identifier}")
            
            # 写回文件
            with open(self.extracted_poems_file, "w", encoding="utf-8") as f:
                json.dump(poems, f, ensure_ascii=False, indent=2)
            
            if found:
                logger.info(f"成功更新 {poem_identifier} 的妙处到 extracted_poems.json")
        except Exception as e:
            poem_identifier = f"ID_{poem_data.get('id')}" if poem_data.get('id') is not None else poem_data.get('标题', '未知')
            logger.error(f"更新extracted_poems.json失败 ({poem_identifier}): {e}", exc_info=True)
    
    def save_final_result(self, poem_data: Dict):
        """将最终结果追加到JSON文件"""
        try:
            # 读取现有数据
            if self.final_results_file.exists():
                with open(self.final_results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            else:
                results = []
            
            # 更新或添加诗词
            poem_title = poem_data.get("标题", "")
            found = False
            for i, item in enumerate(results):
                if item.get("标题") == poem_title:
                    results[i] = poem_data
                    found = True
                    break
            
            if not found:
                results.append(poem_data)
            
            # 写回文件
            with open(self.final_results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存最终结果到文件失败: {e}")
    
    def run_full_pipeline(
        self, 
        poem_limit: Optional[int] = None,
        max_tree_depth: Optional[int] = None
    ):
        """运行完整流程（增量写入）"""
        logger.info("=" * 50)
        logger.info("开始诗词分析流程")
        logger.info("=" * 50)
        
        # 步骤1: 提取诗词（支持增量与跳过）
        logger.info("\n步骤1: 提取诗词信息")
        poems_file = self.extractor.output_file
        poems: List[Dict] = []

        # 检查是否已有提取结果
        if poems_file and poems_file.exists():
            logger.info(f"检测到已存在提取结果文件，尝试读取: {poems_file}")
            try:
                with open(poems_file, "r", encoding="utf-8") as f:
                    poems = json.load(f)
                # 验证数据有效性：至少要有内容
                valid_poems = [p for p in poems if p.get("内容") and p.get("内容").strip()]
                if len(valid_poems) > 0:
                    logger.info(f"成功读取 {len(valid_poems)} 首已提取的诗词，将在此基础上继续")
                    poems = valid_poems
                else:
                    logger.warning("现有提取结果文件为空或无效，将重新提取所有诗词")
                    poems = self.extractor.extract_all_poems(max_workers=self.max_workers)
            except Exception as e:
                logger.warning(f"读取已有提取结果失败，将重新提取: {e}")
                poems = self.extractor.extract_all_poems(max_workers=self.max_workers)
        else:
            logger.info("未找到已存在的提取结果文件，开始提取所有诗词")
            poems = self.extractor.extract_all_poems(max_workers=self.max_workers)
        
        if poem_limit:
            poems = poems[:poem_limit]
            logger.info(f"限制处理前 {poem_limit} 首诗词")
        
        logger.info(f"提取完成，诗词已保存到: {poems_file}")
        # 输出当前 API 成本统计（提取阶段后，通常为空或仅包含旧流程消耗）
        logger.info("当前 TOKEN 成本统计（提取阶段后）:\n" + format_token_cost_summary())
        
        # 步骤2: 分析妙处（直接更新extracted_poems.json）
        logger.info("\n步骤2: 分析诗词妙处")
        logger.info("说明：如果extracted_poems.json中某ID的诗词已有妙处，则直接使用；如果没有，则分析并更新")
        
        # 重新读取extracted_poems.json（可能已被更新）
        if poems_file and poems_file.exists():
            try:
                with open(poems_file, "r", encoding="utf-8") as f:
                    poems = json.load(f)
                if poem_limit:
                    poems = poems[:poem_limit]
                logger.info(f"重新读取extracted_poems.json，共 {len(poems)} 首诗词")
            except Exception as e:
                logger.warning(f"重新读取extracted_poems.json失败: {e}")
        
        # 筛选需要分析的诗词（没有"妙处"字段或"妙处"为空或无效）
        def is_merits_valid(poem: Dict) -> bool:
            """检查妙处是否有效"""
            merits = poem.get("妙处")
            if not merits:
                return False
            if not isinstance(merits, dict):
                return False
            if len(merits) == 0:
                return False
            # 检查是否有有效的键值对（键和值都不为空）
            for key, value in merits.items():
                if key and value and str(key).strip() and str(value).strip():
                    return True
            return False
        
        poems_to_analyze = [
            (i, poem) for i, poem in enumerate(poems, 1)
            if not is_merits_valid(poem)
        ]
        
        # 统计已分析的诗词数量
        analyzed_count = len(poems) - len(poems_to_analyze)
        
        if not poems_to_analyze:
            logger.info(f"所有诗词已分析完成（共 {analyzed_count} 首），跳过分析步骤")
        else:
            logger.info(f"需要分析 {len(poems_to_analyze)} 首诗词（已分析 {analyzed_count} 首），并发数: {self.max_workers}")
            
            def analyze_poem(args):
                """分析单首诗词（基于ID匹配更新）"""
                i, poem = args
                poem_id = poem.get("id")
                title = poem.get("标题", "")
                poem_identifier = f"ID_{poem_id}" if poem_id is not None else title
                
                try:
                    # 再次检查是否已分析（防止并发情况下的重复分析）
                    if is_merits_valid(poem):
                        logger.info(f"跳过已分析的诗词 {i}/{len(poems)} ({poem_identifier}): {title}")
                        return poem
                    
                    logger.info(f"处理第 {i}/{len(poems)} 首诗词 ({poem_identifier}): {title}")
                    merits = self.analyzer.analyze_merits(poem)
                    if merits and isinstance(merits, dict) and len(merits) > 0:
                        poem["妙处"] = merits
                        logger.info(f"发现 {len(merits)} 个妙处: {list(merits.keys())}")
                    else:
                        logger.warning(f"诗词 {poem_identifier} ({title}) 分析结果为空，将跳过")
                        return None
                    return poem
                except Exception as e:
                    logger.error(f"分析诗词 {poem_identifier} ({title}) 失败: {e}")
                    return None
            
            # 并发分析
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(analyze_poem, args): args
                    for args in poems_to_analyze
                }
                
                for future in tqdm(as_completed(futures), total=len(poems_to_analyze), desc="分析妙处", unit="首"):
                    poem = future.result()
                    if poem:
                        # 立即更新extracted_poems.json（基于ID匹配）
                        self.update_poem_in_extracted_file(poem)
        
        logger.info(f"分析完成，结果已更新到: {poems_file}")
        # 输出 API 成本统计（分析阶段后）
        logger.info("当前 TOKEN 成本统计（分析阶段后）:\n" + format_token_cost_summary())
        
        # 读取最新结果（包含妙处，包括已有妙处的和刚分析的）
        if poems_file and poems_file.exists():
            try:
                with open(poems_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                if poem_limit:
                    results = results[:poem_limit]
                logger.info(f"读取最新结果，共 {len(results)} 首诗词（包含已有妙处和刚分析的）")
            except Exception as e:
                logger.warning(f"读取最新结果失败: {e}")
                results = poems
        else:
            results = poems
        
        # 步骤3: 生成诗词树（集中写入单个文件）
        logger.info("\n步骤3: 生成诗词树")

        # 已有的最终结果（可能包含部分诗词树）
        existing_final: List[Dict] = []
        if self.final_results_file.exists():
            try:
                with open(self.final_results_file, "r", encoding="utf-8") as f:
                    existing_final = json.load(f)
                logger.info(f"检测到已存在最终结果文件，将在其基础上继续生成树: {self.final_results_file}")
            except Exception as e:
                logger.warning(f"读取已有最终结果失败，将重新生成树: {e}")
                existing_final = []

        final_by_title: Dict[str, Dict] = {
            item.get("标题"): item for item in existing_final if isinstance(item, dict)
        }

        aggregated_results: List[Dict] = []
        
        # 筛选需要生成树的诗词
        poems_to_generate = []
        for i, poem_data in enumerate(results, 1):
            poem_content = poem_data.get("内容", "")
            merits = poem_data.get("妙处", {})
            title = poem_data.get("标题", "")
            
            if not poem_content or not merits:
                logger.warning(f"第 {i} 首诗词缺少内容或妙处，跳过")
                continue

            # 如果已有该诗词的完整树，则跳过
            existing_item = final_by_title.get(title)
            if existing_item and existing_item.get("诗词树"):
                # 检查树是否完整（简单检查：是否有children）
                existing_tree = existing_item.get("诗词树")
                if existing_tree and isinstance(existing_tree, dict):
                    # 检查树是否看起来完整（有根节点和至少一些子节点）
                    # 这里可以更复杂地检查，但为了简单，我们假设如果存在树就认为可能完整
                    # 如果需要断点重跑，可以传递 existing_tree 给 generate_tree
                    logger.info(f"检测到已有树结构，将尝试断点重跑: {title}")
                    poems_to_generate.append((i, poem_data, existing_tree))
                else:
                    logger.info(f"跳过已生成树的诗词 {i}/{len(results)}: {title}")
                    aggregated_results.append(existing_item)
                    continue
            else:
                poems_to_generate.append((i, poem_data, None))
        
        if not poems_to_generate:
            logger.info("所有诗词树已生成完成，跳过生成步骤")
        else:
            logger.info(f"需要生成 {len(poems_to_generate)} 首诗词的树，并发数: {self.max_workers}")
            
            def generate_tree_for_poem(args):
                """为单首诗词生成树（支持断点重跑）"""
                i, poem_data, existing_tree = args
                poem_content = poem_data.get("内容", "")
                merits = poem_data.get("妙处", {})
                title = poem_data.get("标题", "")
                
                try:
                    if existing_tree:
                        logger.info(f"\n为第 {i} 首诗词继续生成诗词树（断点重跑，{len(merits)} 个优势）: {title}")
                    else:
                        logger.info(f"\n为第 {i} 首诗词生成诗词树（{len(merits)} 个优势）: {title}")

                    # 拼接背景信息（说明 + 评解），作为写作背景简介
                    background_parts = []
                    if poem_data.get("说明"):
                        background_parts.append(f"【说明】\n{poem_data['说明']}")
                    if poem_data.get("评解"):
                        background_parts.append(f"【评解】\n{poem_data['评解']}")
                    background_text = "\n\n".join(background_parts)

                    base_id = poem_data.get("id")
                    tree = self.generator.generate_tree(
                        poem_content, 
                        merits, 
                        max_depth=max_tree_depth,
                        background=background_text,
                        base_id=str(base_id) if base_id is not None else None,
                        existing_tree=existing_tree,  # 传递已有树结构，支持断点重跑
                    )
                    
                    poem_data["诗词树"] = tree
                    return poem_data
                except Exception as e:
                    logger.error(f"生成诗词树失败 ({title}): {e}")
                    return None
            
            # 并发生成树（注意：每首诗的树内部已经并发，这里只并发不同诗的树）
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(generate_tree_for_poem, args): args
                    for args in poems_to_generate
                }
                
                for future in tqdm(as_completed(futures), total=len(poems_to_generate), desc="生成诗词树", unit="首"):
                    poem_data = future.result()
                    if poem_data:
                        aggregated_results.append(poem_data)
                        # 追加写入同一个结果文件
                        self.save_final_result(poem_data)
        
        # 统一保存最终结果，确保只有一个文件
        with open(self.final_results_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_results, f, ensure_ascii=False, indent=2)
        logger.info(f"诗词树生成完成，结果已集中保存在: {self.final_results_file}")
        # 输出最终 API 成本统计（包含提取+分析+生成树阶段的总消耗）
        token_summary_str = format_token_cost_summary()
        logger.info("最终 TOKEN 成本统计（全部阶段）:\n" + token_summary_str)

        # 将 TOKEN_COST 统计写入 JSON 文件，便于后续分析
        try:
            token_summary = get_token_cost_summary()
            with open(self.token_cost_file, "w", encoding="utf-8") as f:
                json.dump(token_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"TOKEN_COST 统计已写入: {self.token_cost_file}")
        except Exception as e:
            logger.error(f"写入 TOKEN_COST JSON 失败: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("流程完成！")
        logger.info("=" * 50)
        
        # 读取最终结果返回
        with open(self.final_results_file, "r", encoding="utf-8") as f:
            final_results = json.load(f)
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description="Build Poetry-Tree JSON (§3.1 / Algorithm 1)")
    parser.add_argument(
        "--source",
        default=str(REPO_ROOT / "data" / "honglou" / "honglou_poems.json"),
        help="Honglou-Poem JSON with expert critiques (public honglou_poems.json is not enough; see README)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("output")),
        help="Directory for extracted_poems.json and final_results_*.json",
    )
    parser.add_argument("--model", default="glm-4.6", help="API model id (sent as-is)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-children", type=int, default=3, help="Children sampled per node (paper setting: 3)")
    parser.add_argument("--poem-limit", type=int, default=None, help="Debug: only the first N poems")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    pipeline = PoetryAnalysisPipeline(
        pdf_path=args.source,
        extract_model=args.model,
        analyze_model=args.model,
        generate_model=args.model,
        output_dir=args.output_dir,
        use_cache_for_tree=args.use_cache,
        max_workers=args.workers,
        max_children_per_level=args.max_children,
    )
    results = pipeline.run_full_pipeline(
        poem_limit=args.poem_limit,
        max_tree_depth=args.max_depth,
    )
    print(f"Done. {len(results)} trees -> {pipeline.final_results_file}")


if __name__ == "__main__":
    main()

