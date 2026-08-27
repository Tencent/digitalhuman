"""Minimal models for Honglou-Poem extraction (§2.1)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class Poetry(BaseModel):
    content: str = Field(description="诗歌正文")
    background: str = Field(description="作诗背景")
    chapter: int = Field(description="所属回数")
    author: Optional[str] = Field(default=None, description="作者")
    author_profile: Optional[str] = Field(default=None, description="作者身份与当时状态")
    title: Optional[str] = Field(default=None, description="标题")
    exam_topic: Optional[str] = Field(default=None, description="命题式题目")
    author_intent: Optional[str] = Field(default=None, description="写作目的")


class Chapter(BaseModel):
    chapter_number: int
    title: str
    content: str
    poetries: List[Poetry] = Field(default_factory=list)
