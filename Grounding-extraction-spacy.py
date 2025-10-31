#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
船舶搁浅事故信息抽取程序 (纯spaCy)
使用spaCy的NLP功能抽取事件信息
"""

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from typing import List, Dict, Tuple
import json

class SpacyGroundingExtractor:
    def __init__(self):
        """初始化纯spaCy抽取器"""
        # 加载spaCy模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("请先安装spacy模型: python -m spacy download en_core_web_sm")
            self.nlp = None
            return
        
        # 创建Matcher对象
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # 初始化模式
        self._init_patterns()
    
    def _init_patterns(self):
        """初始化spaCy匹配模式"""
        
        # 1. 触发词模式 (使用Matcher)
        trigger_patterns = [
            # 基本触发词模式
            [{"LOWER": "grounded"}],
            [{"LOWER": "grounding"}],
            [{"LOWER": "ran"}, {"LOWER": "aground"}],
            [{"LOWER": "struck"}],
            [{"LOWER": "hit"}],
            [{"LOWER": "collided"}],
            [{"LOWER": "beached"}],
            [{"LOWER": "stranded"}],
            [{"LOWER": "stuck"}],
            
            # 更复杂的模式
            [{"LOWER": "vessel"}, {"LOWER": {"IN": ["grounded", "struck", "hit"]}}],
            [{"LOWER": "ship"}, {"LOWER": {"IN": ["grounded", "struck", "hit"]}}],
        ]
        
        self.matcher.add("TRIGGER", trigger_patterns)
        
        # 2. 船舶名称模式
        vessel_patterns = [
            # 船舶前缀 + 名称
            [{"TEXT": {"IN": ["MV", "MT", "MS", "SS"]}}, {"POS": "PROPN"}],
            [{"TEXT": {"IN": ["MV", "MT", "MS", "SS"]}}, {"POS": "PROPN"}, {"POS": "PROPN"}],
            
            # 船舶类型 + 名称
            [{"LOWER": {"IN": ["cargo", "container", "bulk", "cruise"]}}, 
             {"LOWER": "ship"}, {"POS": "PROPN"}],
            [{"LOWER": "ferry"}, {"POS": "PROPN"}],
            [{"LOWER": "tanker"}, {"POS": "PROPN"}],
        ]
        
        self.matcher.add("VESSEL", vessel_patterns)
        
        # 3. 位置模式
        location_patterns = [
            [{"LOWER": {"IN": ["reef", "rock", "shoal", "sandbar", "beach", "coast"]}}],
            [{"LOWER": {"IN": ["harbor", "port", "channel", "strait", "bay"]}}],
            [{"LOWER": "near"}, {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}],
            [{"LOWER": "in"}, {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}],
            [{"LOWER": "off"}, {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}],
        ]
        
        self.matcher.add("LOCATION", location_patterns)
        
        # 4. 原因模式
        cause_patterns = [
            [{"LOWER": {"IN": ["weather", "storm", "fog", "wind", "wave"]}}],
            [{"LOWER": "strong"}, {"LOWER": {"IN": ["wind", "current", "wave"]}}],
            [{"LOWER": "poor"}, {"LOWER": "visibility"}],
            [{"LOWER": {"IN": ["navigation", "mechanical", "engine", "steering"]}}, 
             {"LOWER": {"IN": ["error", "failure"]}}],
            [{"LOWER": "human"}, {"LOWER": "error"}],
        ]
        
        self.matcher.add("CAUSE", cause_patterns)
        
        # 5. 损害模式
        damage_patterns = [
            [{"LOWER": {"IN": ["damage", "breach", "hole", "crack", "leak"]}}],
            [{"LOWER": "oil"}, {"LOWER": "spill"}],
            [{"LOWER": "hull"}, {"LOWER": "damage"}],
            [{"LOWER": {"IN": ["minor", "major", "severe"]}}, {"LOWER": "damage"}],
        ]
        
        self.matcher.add("DAMAGE", damage_patterns)
        
        # 6. 响应行动模式
        response_patterns = [
            [{"LOWER": {"IN": ["rescue", "salvage", "tow", "refloat", "evacuate"]}}],
            [{"LOWER": "coast"}, {"LOWER": "guard"}],
            [{"LOWER": "emergency"}, {"LOWER": "response"}],
            [{"LOWER": {"IN": ["dispatched", "deployed", "sent"]}}, 
             {"LOWER": {"IN": ["team", "crews", "vessels"]}}],
        ]
        
        self.matcher.add("RESPONSE", response_patterns)
    
    def extract_with_dependency(self, doc) -> Dict[str, List[str]]:
        """使用依存句法分析提取信息"""
        extracted = {
            "vessels": [],
            "locations": [],
            "causes": [],
            "times": [],
            "damages": [],
            "responses": []
        }
        
        # 查找触发词及其相关成分
        trigger_verbs = ["grounded", "grounding", "struck", "hit", "collided", 
                        "beached", "stranded", "stuck", "ran"]
        
        for token in doc:
            # 1. 查找主语（通常是船舶）
            if token.lemma_ in trigger_verbs or token.text.lower() in trigger_verbs:
                # 获取主语
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        # 获取完整的名词短语
                        vessel_phrase = self._get_noun_phrase(child)
                        if vessel_phrase:
                            extracted["vessels"].append(vessel_phrase)
                
                # 获取介词宾语（通常是位置）
                for child in token.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                location_phrase = self._get_noun_phrase(grandchild)
                                if location_phrase:
                                    extracted["locations"].append(location_phrase)
            
            # 2. 查找原因（通常由 "due to", "caused by", "because of" 引导）
            if token.text.lower() in ["due", "caused", "because"]:
                for child in token.children:
                    if child.dep_ in ["pobj", "agent"]:
                        cause_phrase = self._get_noun_phrase(child)
                        if cause_phrase:
                            extracted["causes"].append(cause_phrase)
        
        return extracted
    
    def _get_noun_phrase(self, token) -> str:
        """获取完整的名词短语"""
        # 获取token所在的名词短语
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        return token.text
    
    def extract_with_ner(self, doc) -> Dict[str, List[str]]:
        """使用命名实体识别提取信息"""
        extracted = {
            "organizations": [],
            "locations": [],
            "times": [],
            "persons": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                extracted["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                extracted["locations"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                extracted["times"].append(ent.text)
            elif ent.label_ == "PERSON":
                extracted["persons"].append(ent.text)
        
        return extracted
    
    def extract_with_matcher(self, doc) -> Dict[str, List[str]]:
        """使用Matcher提取模式匹配的信息"""
        matches = self.matcher(doc)
        
        extracted = {
            "triggers": [],
            "vessels": [],
            "locations": [],
            "causes": [],
            "damages": [],
            "responses": []
        }
        
        for match_id, start, end in matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]
            
            if match_label == "TRIGGER":
                extracted["triggers"].append(span.text)
            elif match_label == "VESSEL":
                extracted["vessels"].append(span.text)
            elif match_label == "LOCATION":
                extracted["locations"].append(span.text)
            elif match_label == "CAUSE":
                extracted["causes"].append(span.text)
            elif match_label == "DAMAGE":
                extracted["damages"].append(span.text)
            elif match_label == "RESPONSE":
                extracted["responses"].append(span.text)
        
        return extracted
    
    def extract(self, text: str) -> Dict:
        """综合提取方法"""
        if not self.nlp:
            return {}
        
        # 处理文本
        doc = self.nlp(text)
        
        # 1. 使用Matcher提取
        matcher_results = self.extract_with_matcher(doc)
        
        # 2. 使用NER提取
        ner_results = self.extract_with_ner(doc)
        
        # 3. 使用依存句法分析提取
        dep_results = self.extract_with_dependency(doc)
        
        # 合并结果
        final_results = {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "trigger_words": list(set(matcher_results["triggers"])),
            "event_type": self._determine_event_type(matcher_results["triggers"]),
            "arguments": {
                "vessels": list(set(
                    matcher_results["vessels"] + 
                    dep_results["vessels"] + 
                    [org for org in ner_results["organizations"] 
                     if any(ship_word in org.lower() 
                           for ship_word in ["ship", "vessel", "tanker", "ferry", "MV", "MT", "MS"])]
                )),
                "locations": list(set(
                    matcher_results["locations"] + 
                    ner_results["locations"] + 
                    dep_results["locations"]
                )),
                "times": list(set(ner_results["times"] + dep_results["times"])),
                "causes": list(set(matcher_results["causes"] + dep_results["causes"])),
                "damages": list(set(matcher_results["damages"] + dep_results["damages"])),
                "responses": list(set(matcher_results["responses"] + dep_results["responses"])),
                "persons": ner_results["persons"]
            },
            "linguistic_features": {
                "pos_tags": [(token.text, token.pos_) for token in doc][:10],  # 前10个词性标注
                "dependencies": [(token.text, token.dep_, token.head.text) 
                               for token in doc 
                               if token.dep_ != "punct"][:10],  # 前10个依存关系
                "entities": [(ent.text, ent.label_) for ent in doc.ents]
            }
        }
        
        return final_results
    
    def _determine_event_type(self, triggers: List[str]) -> str:
        """基于触发词确定事件类型"""
        if not triggers:
            return "unknown"
        
        triggers_lower = [t.lower() for t in triggers]
        
        if any(t in triggers_lower for t in ["grounding", "grounded", "ran aground"]):
            return "grounding"
        elif any(t in triggers_lower for t in ["struck", "hit", "collided"]):
            return "collision_grounding"
        elif any(t in triggers_lower for t in ["stranded", "beached", "stuck"]):
            return "stranding"
        else:
            return "marine_accident"
    
    def display_results(self, result: Dict):
        """格式化显示结果"""
        print("\n" + "="*70)
        print("船舶搁浅事故信息抽取结果 (纯spaCy)")
        print("="*70)
        
        print(f"\n原文: {result['text']}")
        
        print(f"\n事件类型: {result['event_type']}")
        print(f"触发词: {', '.join(result['trigger_words']) if result['trigger_words'] else '未找到'}")
        
        print("\n事件论元:")
        for arg_type, values in result['arguments'].items():
            if values:
                print(f"  {arg_type:12s}: {', '.join(values)}")
        
        print("\n语言学特征:")
        print("  词性标注 (前10个):")
        for word, pos in result['linguistic_features']['pos_tags']:
            print(f"    {word:15s} -> {pos}")
        
        print("  依存关系 (前10个):")
        for word, dep, head in result['linguistic_features']['dependencies'][:5]:
            print(f"    {word:12s} --{dep:10s}--> {head}")
        
        if result['linguistic_features']['entities']:
            print("  命名实体:")
            for ent_text, ent_label in result['linguistic_features']['entities']:
                print(f"    {ent_text:20s} ({ent_label})")


def main():
    """主程序"""
    # 创建抽取器
    extractor = SpacyGroundingExtractor()
    
    if not extractor.nlp:
        print("无法加载spaCy模型，请先安装：")
        print("pip install spacy")
        print("python -m spacy download en_core_web_sm")
        return
    
    # 示例文本
    test_texts = [
        """The cargo ship MV Ever Given ran aground in the Suez Canal on March 23, 2021, 
        blocking the waterway for six days. The grounding was caused by strong winds and 
        poor visibility during a sandstorm. The vessel was successfully refloated by 
        tugboats and the Egyptian authorities.""",
        
        """A bulk carrier grounded on a reef near the Great Barrier Reef yesterday morning. 
        The vessel suffered hull damage and minor oil leak was reported. Coast Guard 
        dispatched emergency response teams to assess the situation.""",
        
        """The ferry Blue Star struck rocks and beached itself near Sydney harbor entrance 
        during heavy fog conditions at 3:00 AM. All 150 passengers were safely evacuated 
        by rescue teams. Salvage operations are planned for high tide tomorrow morning.""",
        
        """Container ship Ever Fortune collided with a sandbar in Singapore Strait on Monday 
        night due to navigation error. The 200-meter vessel remained stuck for 12 hours 
        before being freed by six tugboats. No injuries or pollution reported, but the 
        ship sustained minor hull damage."""
    ]
    
    # 处理每个示例
    for i, text in enumerate(test_texts, 1):
        print(f"\n\n{'#'*70}")
        print(f"示例 {i}:")
        result = extractor.extract(text)
        extractor.display_results(result)
    
    # 交互式测试
    print("\n\n" + "="*70)
    print("自定义文本测试")
    print("="*70)
    
    while True:
        print("\n输入船舶搁浅事故描述文本 (输入 'quit' 退出):")
        user_text = input("> ")
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_text.strip():
            result = extractor.extract(user_text)
            extractor.display_results(result)
            
            # 可选：输出JSON格式
            print("\n是否查看JSON格式输出? (y/n)")
            if input("> ").lower() == 'y':
                # 移除无法序列化的部分
                json_result = {
                    "text": result["text"],
                    "trigger_words": result["trigger_words"],
                    "event_type": result["event_type"],
                    "arguments": result["arguments"]
                }
                print(json.dumps(json_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("船舶搁浅事故信息抽取系统 (纯spaCy)")
    print("使用spaCy的Matcher、NER和依存句法分析")
    main()