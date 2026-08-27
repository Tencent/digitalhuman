"""Extract poems and narrative context from a user-provided novel text (§2.1)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.llm_client import call_LLM
from models import Chapter, Poetry
from text_processor import TextProcessor


class PoetryExtractor:
    """诗歌提取器，使用大模型提取章节中的诗歌"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        global_result_file: str = "output/extracted_poetries.json",
        chapter_summary_file: str = "output/chapter_summaries.json",
        max_workers: Optional[int] = None,
    ):
        """
        初始化提取器
        
        Args:
            model_name: 用于调用的模型 id，会原样写入 API 请求
            global_result_file: 全局提取结果文件路径
            max_workers: 并发线程数，None 时读取环境变量 EXTRACTOR_MAX_WORKERS（默认 3）
        """
        env_workers = os.getenv("EXTRACTOR_MAX_WORKERS")
        try:
            default_workers = int(env_workers) if env_workers is not None else 3
        except ValueError:
            default_workers = 3
        self.model_name = model_name
        self.global_result_file = Path(global_result_file)
        self.chapter_summary_file = Path(chapter_summary_file)
        self.chapter_summaries: Dict[str, str] = self._load_chapter_summaries()
        self.max_workers = max(1, max_workers if max_workers is not None else default_workers)
        self._summary_lock = Lock()
    
    def load_all_poetries_from_global_result(self) -> Tuple[List[Poetry], bool]:
        """
        从全局提取结果文件加载所有诗歌
        
        Returns:
            (诗歌列表, 是否成功加载)
        """
        if not self.global_result_file.exists():
            return [], False
        
        try:
            with open(self.global_result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                poetries = [Poetry(**item) for item in data]
            else:
                poetries = []
            
            return poetries, True
        except Exception as e:
            print(f"读取全局提取结果文件失败: {e}")
            return [], False
    
    def extract_poetries_from_chapter(
        self,
        chapter: Chapter,
        prev_summary: Optional[str] = None,
        next_summary: Optional[str] = None,
    ) -> Tuple[List[Poetry], bool]:
        """
        从章节中提取诗歌
        
        Args:
            chapter: 章节对象
            
        Returns:
            (诗歌列表, 是否来自缓存)
        """
        # 如果全局结果文件存在，直接跳过提取
        if self.global_result_file.exists():
            return [], True

        # Step 1: 提取诗歌与结构化信息
        raw_poems = self._extract_poems_with_details(chapter)
        poetries: List[Poetry] = []
        for item in raw_poems:
            content = item.get("content", "").strip()
            if not content:
                continue
            poetry = Poetry(
                content=content,
                chapter=chapter.chapter_number,
                author=item.get("author_name"),
                author_profile=item.get("author_profile"),
                title=item.get("title"),
                background=item.get("background_detail", ""),
                exam_topic=item.get("exam_topic"),
            )
            poetries.append(poetry)

        if not poetries:
            return [], False

        # Step 2: 摘要章节剧情并保存
        chapter_summary = self._summarize_chapter(chapter)
        if chapter_summary:
            self._persist_chapter_summary(chapter.chapter_number, chapter_summary)
        else:
            chapter_summary = ""

        prev_context = prev_summary or self.chapter_summaries.get(str(chapter.chapter_number - 1), "")
        next_context = next_summary or self.chapter_summaries.get(str(chapter.chapter_number + 1), "")

        # Step 3: 推断作者写作意图
        for poetry in poetries:
            intent = self._infer_author_intent(
                poem=poetry,
                chapter=chapter,
                chapter_summary=chapter_summary,
                prev_summary=prev_context,
                next_summary=next_context,
            )
            poetry.author_intent = intent or ""

        return poetries, False
    
    def _extract_poems_with_details(self, chapter: Chapter) -> List[Dict[str, str]]:
        """调用模型提取诗歌并返回结构化信息"""
        prompt = f"""你是《红楼梦》文献整理师，请从下面的章节中提取所有正式诗作。

章节：第{chapter.chapter_number}回《{chapter.title}》
正文片段：
{chapter.content}

输出 JSON，格式如下：
{{
  "poetries": [
    {{
      "title": "标题，若无可填null",
      "content": "诗歌原文（逐字逐句）",
      "author_name": "作者姓名或身份(如果是叙述者，则是曹雪芹)",
      "author_profile": "【作者分析】这是一个字符串，请深入分析作者在作此诗时的状态。请综合以下几点进行描述，形成一段连贯的文字（约100-150字）：\\n1. **身份与处境**：作者在故事中的核心身份、社会地位以及当时的生存状态。\\n2. **性格特质**：作者最突出的性格特点，尤其是与诗歌内容相关联的性格。\\n3. **当时的情感关系**：与关键人物（尤其是贾宝玉）的情感互动状态，是否存在误会、爱恋、嫉妒等情绪。\\n4. **与诗歌相关的个人背景**：作者的身世、经历或特定遭遇是如何影响这首诗的创作的。",
      "background_detail": "【作诗背景】这是一个字符串，请详细描述诗歌创作的直接背景。请综合以下几点进行描述，形成一段连贯的文字（约100-150字）：\\n1. **时间**：明确的季节、节气、日期或具体时刻（如清晨、黄昏）。\\n2. **地点**：具体的创作地点，如大观园的某个角落、某个房间内。\\n3. **前情提要**：在作诗前不久，作者刚刚经历了什么关键事件或对话。\\n4. **核心触发事件**：是什么景象、声音或思绪直接点燃了作者的创作灵感。",
      "exam_topic": "面向诗赛的命题式题目"
    }}
  ]
}}

规则：
1. 仅保留格律诗、词、曲等完整篇章（内容要求两联以上），排除随口打趣、对白或题目。
2. 内容需与原文一致，不得改写。
3. 若某字段缺失可填null，但不可省略键。
4. 即使没有诗，也需返回{{"poetries":[]}}。"""

        response = self._call_model(prompt)
        data = self._safe_load_json(response)
        if not isinstance(data, dict):
            return []
        return data.get("poetries", []) or []
    
    def _summarize_chapter(self, chapter: Chapter) -> str:
        """对章节进行剧情摘要，保留关键情节"""
        prompt = f"""请精炼总结《红楼梦》第{chapter.chapter_number}回《{chapter.title}》的剧情焦点。

要求：
1. 交代主要事件、角色互动、情感变化以及对后续剧情的影响。
2. 不要照搬原文，请用现代中文复述。

章节内容：
{chapter.content[:5000]}
"""
        response = self._call_model(prompt, system="你是严谨的古典小说编年体整理者。")
        return response.strip()

    def _infer_author_intent(
        self,
        poem: Poetry,
        chapter: Chapter,
        chapter_summary: str,
        prev_summary: str,
        next_summary: str,
    ) -> str:
        """结合前后章摘要与原文上下文，推断作者写作目的"""
        context = {
            "previous_chapter_summary": prev_summary or "暂无可用摘要",
            "current_chapter_summary": chapter_summary or "暂无可用摘要",
            "next_chapter_summary": next_summary or "暂无可用摘要",
            "current_chapter_excerpt": chapter.content,
            "poem_content": poem.content,
            "poem_background": poem.background,
            "author_profile": poem.author_profile or poem.author,
        }
        prompt = f"""结合以下材料，说明这首诗被写出的目的或功能：

{json.dumps(context, ensure_ascii=False, indent=2)}

请输出JSON：
{{
  "author_intent": "一句话概括写作目的与作用",
  "rationale": "简述依据（可引用剧情或人物状态）"
}}

注意区分作者是曹雪芹（叙事用途、伏笔、讽喻等）还是书中角色（情绪抒发、社交应酬、应景即兴等）。"""

        response = self._call_model(prompt, system="你是《红楼梦》诗词研究者，擅长解读写作动机。")
        data = self._safe_load_json(response)
        if isinstance(data, dict):
            intent = data.get("author_intent") or data.get("目的") or data.get("intent")
            if isinstance(intent, str):
                return intent.strip()
        return response.strip()

    def _call_model(self, prompt: str, system: str = "你是一个专业的文学分析专家。") -> str:
        """统一的模型调用入口"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response, _ = call_LLM(messages, self.model_name)
        return response
    
    def _safe_load_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """容错解析JSON响应"""
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    def _load_chapter_summaries(self) -> Dict[str, str]:
        if not self.chapter_summary_file.exists():
            return {}
        try:
            with open(self.chapter_summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            print(f"读取章节摘要失败: {exc}")
        return {}

    def _persist_chapter_summary(self, chapter_number: int, summary: str) -> None:
        if not summary:
            return
        with self._summary_lock:
            self.chapter_summaries[str(chapter_number)] = summary
            self.chapter_summary_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.chapter_summary_file, "w", encoding="utf-8") as f:
                    json.dump(self.chapter_summaries, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"保存章节摘要失败: {exc}")


def _load_chapters(data_dir: Path) -> List[Chapter]:
    """加载原始文本并切分章节"""
    processor = TextProcessor()
    text_files = list(data_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"未在 {data_dir} 下找到文本文件")
    text_path = text_files[0]
    text = processor.load_text_from_file(str(text_path))
    chapters = processor.split_by_chapters(text)
    if not chapters:
        raise ValueError("章节切分结果为空")
    return chapters


def _summarize_and_persist(extractor: PoetryExtractor, chapter: Chapter) -> None:
    summary = extractor._summarize_chapter(chapter)
    extractor._persist_chapter_summary(chapter.chapter_number, summary)


def _process_chapter_poetries(extractor: PoetryExtractor, chapter: Chapter) -> List[Dict[str, Any]]:
    prev_summary = extractor.chapter_summaries.get(str(chapter.chapter_number - 1), "")
    next_summary = extractor.chapter_summaries.get(str(chapter.chapter_number + 1), "")
    poetries, _ = extractor.extract_poetries_from_chapter(
        chapter,
        prev_summary=prev_summary,
        next_summary=next_summary,
    )
    return [poetry.model_dump() for poetry in poetries]


def _ensure_summaries(extractor: PoetryExtractor, chapters: List[Chapter], force: bool = False):
    """确保每个章节都已有摘要，可选择强制重新生成"""
    if force and extractor.chapter_summary_file.exists():
        extractor.chapter_summary_file.unlink()
        extractor.chapter_summaries = {}
    pending: List[Chapter] = []
    for chapter in chapters:
        key = str(chapter.chapter_number)
        if not force and key in extractor.chapter_summaries:
            continue
        pending.append(chapter)
    if not pending:
        return
    if extractor.max_workers <= 1 or len(pending) == 1:
        for chapter in tqdm(pending, desc="生成章节摘要", unit="回"):
            _summarize_and_persist(extractor, chapter)
        return
    with ThreadPoolExecutor(max_workers=extractor.max_workers) as executor:
        futures = [executor.submit(_summarize_and_persist, extractor, chapter) for chapter in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成章节摘要", unit="回"):
            future.result()


def _to_public_record(idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map extractor fields onto the public Honglou-Poem schema."""
    return {
        "id": idx,
        "title": rec.get("title") or "",
        "poet": rec.get("content") or rec.get("poet") or "",
        "author": rec.get("author") or "",
        "background": rec.get("background") or "",
    }


def run_extraction(
    data_dir: str = "novel",
    model_name: str = "gpt-4o",
    force: bool = False,
    max_workers: Optional[int] = None,
):
    data_path = Path(data_dir)
    chapters = _load_chapters(data_path)
    print(f"共加载 {len(chapters)} 个章节。")

    extractor = PoetryExtractor(model_name=model_name, max_workers=max_workers)

    if extractor.global_result_file.exists():
        if not force:
            print(f"检测到 {extractor.global_result_file} 已存在，跳过提取。（使用 --force 可重新生成）")
            return
        extractor.global_result_file.unlink()
        print("已删除旧的 extracted_poetries.json，将重新提取。")

    _ensure_summaries(extractor, chapters, force=force)

    all_poems: List[Dict[str, Any]] = []
    if extractor.max_workers <= 1:
        for chapter in tqdm(chapters, desc="提取诗歌", unit="回"):
            all_poems.extend(_process_chapter_poetries(extractor, chapter))
    else:
        with ThreadPoolExecutor(max_workers=extractor.max_workers) as executor:
            futures = [executor.submit(_process_chapter_poetries, extractor, chapter) for chapter in chapters]
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取诗歌", unit="回"):
                all_poems.extend(future.result())
    extractor.global_result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(extractor.global_result_file, "w", encoding="utf-8") as f:
        json.dump([_to_public_record(i, rec) for i, rec in enumerate(all_poems, 1)], f, ensure_ascii=False, indent=2)
    print(f"完成！共提取 {len(all_poems)} 首诗歌，已保存至 {extractor.global_result_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract Honglou-Poem entries from a user-provided novel (.txt)")
    parser.add_argument("--data-dir", type=str, default="novel", help="自备原文目录（默认 novel/，仅读取 .txt）")
    parser.add_argument("--model", type=str, default=os.getenv("DEFAULT_MODEL", "gpt-4o"), help="API model id (sent as-is)")
    parser.add_argument("--force", action="store_true", help="重新生成摘要与提取结果（覆盖缓存）")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="模型并发线程数，默认读取 EXTRACTOR_MAX_WORKERS（未设置则为 3）",
    )
    args = parser.parse_args()

    run_extraction(
        data_dir=args.data_dir,
        model_name=args.model,
        force=args.force,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()


