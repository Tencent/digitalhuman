"""
文本处理模块：按回数分割《红楼梦》等文本
"""
import re
from typing import List
from pathlib import Path

from .models import Chapter


class TextProcessor:
    """文本处理器，用于分割章节"""
    
    def __init__(self):
        # 匹配回数的正则表达式，支持多种格式
        self.chapter_pattern = re.compile(
            r'第[一二三四五六七八九十百千万\d]+回\s*[^\n]*'
        )
    
    def split_by_chapters(self, text: str) -> List[Chapter]:
        """
        将文本按回数分割成章节
        
        Args:
            text: 完整文本内容
            
        Returns:
            章节列表
        """
        chapters = []
        
        # 找到所有回数的位置
        matches = list(self.chapter_pattern.finditer(text))
        
        if not matches:
            # 如果没有找到回数，将整个文本作为一章
            chapters.append(Chapter(
                chapter_number=1,
                title="未命名章节",
                content=text
            ))
            return chapters
        
        # 处理每一章
        for i, match in enumerate(matches):
            chapter_header = match.group(0)
            start_pos = match.start()
            
            # 确定章节结束位置
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # 提取章节内容
            chapter_content = text[start_pos:end_pos]
            
            # 解析回数和标题
            chapter_number, title = self._parse_chapter_header(chapter_header)
            
            chapters.append(Chapter(
                chapter_number=chapter_number,
                title=title,
                content=chapter_content
            ))
        
        return chapters
    
    def _parse_chapter_header(self, header: str) -> tuple[int, str]:
        """
        解析章节标题，提取回数和标题
        
        Args:
            header: 章节标题行，如 "第一回  甄士隐梦幻识通灵　贾雨村风尘怀闺秀"
            
        Returns:
            (回数, 标题)
        """
        # 提取回数
        number_match = re.search(r'第([一二三四五六七八九十百千万\d]+)回', header)
        if number_match:
            number_str = number_match.group(1)
            chapter_number = self._chinese_to_number(number_str)
        else:
            chapter_number = 1
        
        # 提取标题（回数后面的内容）
        title_match = re.search(r'第[一二三四五六七八九十百千万\d]+回\s*(.+)', header)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = "未命名"
        
        return chapter_number, title
    
    def _chinese_to_number(self, chinese_num: str) -> int:
        """
        将中文数字转换为阿拉伯数字
        
        Args:
            chinese_num: 中文数字，如 "一"、"十二"、"一百二十"
            
        Returns:
            阿拉伯数字
        """
        # 如果已经是数字，直接返回
        if chinese_num.isdigit():
            return int(chinese_num)
        
        # 中文数字映射
        chinese_digits = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000, '万': 10000
        }
        
        # 简单处理：支持一到九十九
        if len(chinese_num) == 1:
            return chinese_digits.get(chinese_num, 1)
        elif len(chinese_num) == 2:
            if chinese_num[0] == '十':
                return 10 + chinese_digits.get(chinese_num[1], 0)
            elif chinese_num[1] == '十':
                return chinese_digits.get(chinese_num[0], 1) * 10
        elif len(chinese_num) == 3:
            # 如 "十二"
            if chinese_num[0] == '十':
                return 10 + chinese_digits.get(chinese_num[2], 0)
            elif chinese_num[1] == '十':
                tens = chinese_digits.get(chinese_num[0], 1)
                ones = chinese_digits.get(chinese_num[2], 0)
                return tens * 10 + ones
        
        # 更复杂的数字处理（简化版，可根据需要扩展）
        # 这里先返回一个默认值，实际使用时可以完善
        try:
            # 尝试直接转换
            result = 0
            if '百' in chinese_num:
                parts = chinese_num.split('百')
                if parts[0]:
                    result += chinese_digits.get(parts[0], 1) * 100
                if len(parts) > 1 and parts[1]:
                    if '十' in parts[1]:
                        tens_parts = parts[1].split('十')
                        if tens_parts[0]:
                            result += chinese_digits.get(tens_parts[0], 1) * 10
                        if len(tens_parts) > 1 and tens_parts[1]:
                            result += chinese_digits.get(tens_parts[1], 0)
                    else:
                        result += chinese_digits.get(parts[1], 0)
                return result
            elif '十' in chinese_num:
                parts = chinese_num.split('十')
                if parts[0]:
                    result += chinese_digits.get(parts[0], 1) * 10
                if len(parts) > 1 and parts[1]:
                    result += chinese_digits.get(parts[1], 0)
                return result
        except:
            pass
        
        # 默认返回1
        return 1
    
    def load_text_from_file(self, file_path: str) -> str:
        """
        从文件加载文本
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本内容
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法读取文件: {file_path}，尝试的编码都不支持")


