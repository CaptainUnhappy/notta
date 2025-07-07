# -*- coding: utf-8 -*-
"""
多Agent协作RAG系统
实现支持多跳推理的Agentic RAG架构
"""

from typing import List, Dict, Any, Optional
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEFAULT_MODEL
from services.vector_store import VectorStoreService
import logging
import json

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    规划Agent：负责分析用户查询，制定多跳推理计划
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Planner",
            model=DeepSeek(
                id=DEFAULT_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            ),
            instructions="""
            你是一个查询规划专家，负责分析用户问题并制定检索策略。
            
            【核心职责】
            1. 分析用户查询的复杂度和信息需求
            2. 判断是否需要多跳推理
            3. 制定检索计划和推理步骤
            4. 识别关键实体和关系
            
            【输出格式】
            请以JSON格式输出规划结果：
            {
                "query_type": "simple/multi_hop",
                "key_entities": ["实体1", "实体2"],
                "reasoning_steps": [
                    {
                        "step": 1,
                        "action": "search",
                        "target": "搜索目标",
                        "purpose": "搜索目的"
                    }
                ],
                "expected_hops": 2
            }
            """,
            tools=[ReasoningTools()],
            markdown=True
        )
    
    def plan_query(self, user_query: str) -> Dict[str, Any]:
        """
        分析用户查询并制定检索计划
        
        Args:
            user_query: 用户查询
            
        Returns:
            Dict: 包含规划信息的字典
        """
        prompt = f"""
        请分析以下用户查询并制定检索计划：
        
        用户查询：{user_query}
        
        请特别注意：
        1. 如果查询涉及多个实体之间的关系，标记为multi_hop
        2. 识别所有关键实体和可能的中间实体
        3. 制定逐步的检索和推理策略
        """
        
        response = self.agent.run(prompt)
        
        try:
            # 尝试从响应中提取JSON
            content = response.content
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # 寻找JSON对象
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
            
            plan = json.loads(json_str)
            logger.info(f"查询规划完成: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"解析规划结果失败: {e}")
            # 返回默认规划
            return {
                "query_type": "simple",
                "key_entities": [user_query],
                "reasoning_steps": [
                    {
                        "step": 1,
                        "action": "search",
                        "target": user_query,
                        "purpose": "直接搜索相关信息"
                    }
                ],
                "expected_hops": 1
            }

class RetrieverAgent:
    """
    检索Agent：负责根据规划执行文档检索
    """
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.agent = Agent(
            name="Retriever",
            model=DeepSeek(
                id=DEFAULT_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            ),
            instructions="""
            你是一个信息检索专家，负责根据检索计划执行精确的文档搜索。
            
            【核心职责】
            1. 根据检索计划执行多轮搜索
            2. 优化搜索关键词
            3. 评估检索结果的相关性
            4. 识别信息缺口
            
            【搜索策略】
            1. 使用不同的关键词组合
            2. 关注实体名称和关系词
            3. 考虑同义词和相关概念
            """,
            tools=[ReasoningTools()],
            markdown=True
        )
    
    def retrieve_documents(self, plan: Dict[str, Any], similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        根据规划执行文档检索
        
        Args:
            plan: 查询规划
            similarity_threshold: 相似度阈值
            
        Returns:
            List: 检索到的文档列表
        """
        all_documents = []
        
        for step in plan.get('reasoning_steps', []):
            if step['action'] == 'search':
                search_target = step['target']
                logger.info(f"执行检索步骤 {step['step']}: {search_target}")
                
                # 执行搜索
                docs = self.vector_store.search_documents(search_target, similarity_threshold)
                
                # 为每个文档添加检索步骤信息
                for doc in docs:
                    doc_info = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'retrieval_step': step['step'],
                        'search_target': search_target,
                        'purpose': step['purpose']
                    }
                    all_documents.append(doc_info)
        
        logger.info(f"总共检索到 {len(all_documents)} 个文档片段")
        return all_documents
    
    def expand_search(self, entities: List[str], similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        基于实体进行扩展搜索
        
        Args:
            entities: 实体列表
            similarity_threshold: 相似度阈值
            
        Returns:
            List: 检索到的文档列表
        """
        expanded_docs = []
        
        for entity in entities:
            logger.info(f"扩展搜索实体: {entity}")
            docs = self.vector_store.search_documents(entity, similarity_threshold)
            
            for doc in docs:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'retrieval_step': 'expansion',
                    'search_target': entity,
                    'purpose': f'扩展搜索实体: {entity}'
                }
                expanded_docs.append(doc_info)
        
        return expanded_docs

class AnalyzerAgent:
    """
    分析Agent：负责分析检索结果，执行多跳推理
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Analyzer",
            model=DeepSeek(
                id=DEFAULT_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            ),
            instructions="""
            你是一个信息分析专家，负责分析检索到的文档并执行多跳推理。
            
            【核心职责】
            1. 分析文档内容，提取关键信息
            2. 识别实体之间的关系
            3. 执行多跳推理，连接信息片段
            4. 判断信息是否充分回答用户问题
            
            【推理原则】
            1. 基于事实进行推理，不要臆测
            2. 明确标识推理链条
            3. 如果信息不足，明确指出缺失的信息
            4. 提供置信度评估
            
            【输出格式】
            请以JSON格式输出分析结果：
            {
                "reasoning_chain": [
                    {
                        "step": 1,
                        "fact": "从文档中提取的事实",
                        "source": "文档来源"
                    }
                ],
                "conclusion": "最终结论",
                "confidence": 0.9,
                "missing_info": ["缺失的信息"],
                "need_more_search": false
            }
            """,
            tools=[ReasoningTools()],
            markdown=True
        )
    
    def analyze_documents(self, documents: List[Dict[str, Any]], user_query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析检索到的文档并执行推理
        
        Args:
            documents: 检索到的文档列表
            user_query: 用户查询
            plan: 查询规划
            
        Returns:
            Dict: 分析结果
        """
        # 构建文档上下文
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"""文档 {i+1} (检索步骤: {doc['retrieval_step']}, 目标: {doc['search_target']}):
{doc['content']}
""")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        请分析以下检索到的文档，回答用户问题并执行多跳推理。
        
        用户问题：{user_query}
        
        查询规划：{json.dumps(plan, ensure_ascii=False, indent=2)}
        
        检索到的文档：
        {context}
        
        请特别注意：
        1. 如果这是一个多跳问题，请明确展示推理链条
        2. 连接不同文档中的信息片段
        3. 评估信息的完整性和可靠性
        4. 如果需要更多信息，请明确指出
        """
        
        response = self.agent.run(prompt)
        
        try:
            # 尝试从响应中提取JSON
            content = response.content
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # 寻找JSON对象
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
            
            analysis = json.loads(json_str)
            logger.info(f"文档分析完成: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"解析分析结果失败: {e}")
            # 返回原始响应
            return {
                "reasoning_chain": [],
                "conclusion": response.content,
                "confidence": 0.5,
                "missing_info": [],
                "need_more_search": False
            }

class MultiAgentRAGSystem:
    """
    多Agent协作RAG系统主类
    """
    
    def __init__(self, vector_store: VectorStoreService):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(vector_store)
        self.analyzer = AnalyzerAgent()
        self.vector_store = vector_store
        
    def process_query(self, user_query: str, similarity_threshold: float = 0.5, max_iterations: int = 3) -> Dict[str, Any]:
        """
        处理用户查询，执行多Agent协作
        
        Args:
            user_query: 用户查询
            similarity_threshold: 相似度阈值
            max_iterations: 最大迭代次数
            
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始处理查询: {user_query}")
        
        # 第一步：规划
        print("🤔 Planner Agent 正在分析查询...")
        plan = self.planner.plan_query(user_query)
        print(f"📋 查询规划完成: {plan['query_type']} 类型，预期 {plan['expected_hops']} 跳推理")
        
        iteration = 0
        all_documents = []
        final_analysis = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔍 第 {iteration} 轮检索和分析...")
            
            # 第二步：检索
            print("📚 Retriever Agent 正在检索文档...")
            if iteration == 1:
                # 首次检索：按照规划执行
                documents = self.retriever.retrieve_documents(plan, similarity_threshold)
            else:
                # 后续检索：基于缺失信息扩展搜索
                if final_analysis and final_analysis.get('missing_info'):
                    documents = self.retriever.expand_search(final_analysis['missing_info'], similarity_threshold)
                else:
                    break
            
            if not documents:
                print("❌ 未检索到相关文档")
                break
                
            all_documents.extend(documents)
            print(f"✅ 检索到 {len(documents)} 个文档片段")
            
            # 第三步：分析
            print("🧠 Analyzer Agent 正在分析文档...")
            analysis = self.analyzer.analyze_documents(all_documents, user_query, plan)
            final_analysis = analysis
            
            print(f"📊 分析完成，置信度: {analysis.get('confidence', 0)}")
            
            # 判断是否需要继续搜索
            if not analysis.get('need_more_search', False) or analysis.get('confidence', 0) > 0.8:
                print("✅ 分析完成，信息充分")
                break
            
            print("🔄 需要更多信息，准备下一轮检索...")
        
        # 构建最终结果
        result = {
            'user_query': user_query,
            'plan': plan,
            'total_documents': len(all_documents),
            'iterations': iteration,
            'analysis': final_analysis,
            'documents': all_documents
        }
        
        logger.info(f"查询处理完成，共 {iteration} 轮迭代")
        return result