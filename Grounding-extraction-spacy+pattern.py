#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
船舶搁浅事故信息抽取程序 (spaCy + Pattern)
抽取: Event Trigger Words, Event Arguments, Event Types
"""

import spacy
import re
from typing import List, Dict, Tuple, Set

class GroundingEventExtractor:
    def __init__(self):
        """初始化抽取器"""
        # 加载spaCy模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("请先安装spacy模型: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # 定义Pattern关键词列表
        
        # Event Trigger Words (触发词)
        self.trigger_patterns = [
            "grounded", "grounding", "ran aground", "struck",
            "hit", "collided", "beached", "stranded",
            "stuck", "lodged", "foundered", "aground"
        ]
        
        # Event Types (事件类型关键词)
        self.event_type_patterns = {
            "grounding": ["grounding", "grounded", "ran aground", "aground"],
            "collision": ["struck", "hit", "collided", "collision"],
            "stranding": ["stranded", "beached", "stuck", "lodged"],
            "accident": ["accident", "incident", "casualty", "mishap"]
        }
        
        # Event Arguments Patterns (事件论元模式)
        self.argument_patterns = {
            "vessel": [
                "ship", "vessel", "tanker", "cargo ship", "container ship",
                "bulk carrier", "cruise ship", "ferry", "boat", "MV", "MT", "MS"
            ],
            "location": [
                "reef", "rock", "shoal", "sandbar", "beach", "coast",
                "harbor", "port", "channel", "strait", "bay", "island",
                "waters", "sea", "ocean"
            ],
            "cause": [
                "weather", "storm", "fog", "wind", "wave", "current",
                "navigation error", "mechanical failure", "engine failure",
                "steering failure", "human error", "poor visibility"
            ],
            "time": [
                "morning", "afternoon", "evening", "night", "dawn", "dusk",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            "damage": [
                "damage", "breach", "hole", "crack", "oil spill", "pollution",
                "leak", "flooding", "listing", "capsized"
            ],
            "response": [
                "rescue", "salvage", "tow", "refloat", "evacuate",
                "Coast Guard", "emergency", "response", "assist"
            ]
        }
    
    def extract_trigger_words(self, text: str) -> List[Tuple[str, int, int]]:
        """
        抽取触发词及其位置
        返回: [(触发词, 开始位置, 结束位置), ...]
        """
        triggers = []
        text_lower = text.lower()
        
        for pattern in self.trigger_patterns:
            # 使用正则表达式查找所有匹配
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                triggers.append((pattern, match.start(), match.end()))
        
        return triggers
    
    def extract_event_type(self, text: str, triggers: List[Tuple[str, int, int]]) -> str:
        """
        基于触发词确定事件类型
        """
        if not triggers:
            return "unknown"
        
        # 统计各类型的触发词数量
        type_scores = {event_type: 0 for event_type in self.event_type_patterns}
        
        for trigger, _, _ in triggers:
            for event_type, patterns in self.event_type_patterns.items():
                if trigger in patterns:
                    type_scores[event_type] += 1
        
        # 返回得分最高的事件类型
        max_type = max(type_scores, key=type_scores.get)
        return max_type if type_scores[max_type] > 0 else "grounding"
    
    def extract_arguments(self, text: str) -> Dict[str, List[str]]:
        """
        抽取事件论元
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        arguments = {arg_type: [] for arg_type in self.argument_patterns}
        
        # 1. 基于关键词匹配抽取
        text_lower = text.lower()
        for arg_type, patterns in self.argument_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    # 查找完整的短语
                    for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower, re.IGNORECASE):
                        start = match.start()
                        end = match.end()
                        arguments[arg_type].append(text[start:end])
        
        # 2. 使用spaCy的NER增强抽取
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # 地理位置
                arguments["location"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:  # 时间
                arguments["time"].append(ent.text)
            elif ent.label_ in ["ORG"]:  # 组织（可能是船公司或救援组织）
                if any(keyword in ent.text.lower() for keyword in ["guard", "rescue", "maritime"]):
                    arguments["response"].append(ent.text)
                else:
                    arguments["vessel"].append(ent.text)
        
        # 3. 基于依存句法抽取船名
        for token in doc:
            # 查找船舶名称（通常是专有名词）
            if token.pos_ == "PROPN" and token.dep_ in ["nsubj", "nsubjpass"]:
                # 检查是否与触发词相关
                if any(trigger in token.head.text.lower() for trigger in self.trigger_patterns):
                    arguments["vessel"].append(token.text)
        
        # 去重
        for arg_type in arguments:
            arguments[arg_type] = list(set(arguments[arg_type]))
        
        return arguments
    
    def extract(self, text: str) -> Dict:
        """
        完整的事件抽取
        """
        # 1. 抽取触发词
        triggers = self.extract_trigger_words(text)
        
        # 2. 确定事件类型
        event_type = self.extract_event_type(text, triggers)
        
        # 3. 抽取事件论元
        arguments = self.extract_arguments(text)
        
        return {
            "text": text,
            "trigger_words": [t[0] for t in triggers],
            "event_type": event_type,
            "arguments": arguments,
            "trigger_positions": triggers
        }
    
    def display_results(self, result: Dict):
        """
        格式化显示抽取结果
        """
        print("\n" + "="*60)
        print("船舶搁浅事故信息抽取结果")
        print("="*60)
        
        print(f"\n原文: {result['text'][:200]}...")
        print(f"\n事件类型: {result['event_type']}")
        
        print(f"\n触发词: {', '.join(result['trigger_words']) if result['trigger_words'] else '未找到'}")
        
        print("\n事件论元:")
        for arg_type, values in result['arguments'].items():
            if values:
                print(f"  - {arg_type:12s}: {', '.join(values)}")
        
        if result['trigger_positions']:
            print(f"\n触发词位置: {result['trigger_positions']}")


def main():
    """主程序"""
    # 创建抽取器
    extractor = GroundingEventExtractor()
    
    # 示例文本列表
    test_texts = [
        """The cargo ship MV Ever Given ran aground in the Suez Canal on March 23, 2021, 
        blocking the waterway for six days. The grounding was caused by strong winds and 
        poor visibility during a sandstorm. The vessel was successfully refloated by 
        tugboats and the Egyptian authorities.""",
        
        """A bulk carrier grounded on a reef near the Great Barrier Reef yesterday morning. 
        The vessel suffered hull damage and minor oil leak was reported. Coast Guard 
        dispatched emergency response teams to assess the situation.""",
        
        """The ferry struck rocks and beached itself near the harbor entrance during 
        heavy fog conditions. All 150 passengers were safely evacuated. Salvage operations 
        are planned for high tide tomorrow.""",
        
        """Container ship Blue Star collided with a sandbar in Singapore Strait on Monday 
        night due to navigation error. The ship remained stuck for 12 hours before being 
        freed by tugboats. No injuries or pollution reported."""
    ]
    
    # 处理每个示例文本
    for i, text in enumerate(test_texts, 1):
        print(f"\n\n{'#'*60}")
        print(f"示例 {i}:")
        result = extractor.extract(text)
        extractor.display_results(result)
    
    print("\n\n" + "="*60)
    print("自定义文本测试")
    print("="*60)
    
    # 交互式测试
    while True:
        print("\n输入船舶搁浅事故描述文本 (输入 'quit' 退出):")
        user_text = input("> ")
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_text.strip():
            result = extractor.extract(user_text)
            extractor.display_results(result)


if __name__ == "__main__":
    print("船舶搁浅事故信息抽取系统 (spaCy + Pattern)")
    print("使用基于模式的方法抽取事件触发词、论元和类型")
    main()