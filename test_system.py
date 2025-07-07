#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统功能测试脚本
用于验证多Agent RAG系统的基本功能
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from models.multi_agent_rag import PlannerAgent, RetrieverAgent, AnalyzerAgent
from config.settings import DEEPSEEK_API_KEY

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_planner_agent():
    """
    测试Planner Agent
    """
    print("🧪 测试 Planner Agent...")
    
    if DEEPSEEK_API_KEY == "sk-your-deepseek-api-key":
        print("⚠️ 跳过Planner Agent测试 - 需要配置DeepSeek API密钥")
        return
    
    try:
        planner = PlannerAgent()
        
        test_queries = [
            "张三参与了哪个项目？",
            "飞天项目是什么？",
            "李四和王五的关系是什么？"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            try:
                plan = planner.plan_query(query)
                print(f"规划结果: {plan}")
            except Exception as e:
                print(f"❌ 规划失败: {e}")
        
        print("✅ Planner Agent 测试完成")
        
    except Exception as e:
        print(f"❌ Planner Agent 测试失败: {e}")

def test_mock_retriever():
    """
    测试模拟的Retriever功能
    """
    print("\n🧪 测试 Mock Retriever...")
    
    # 模拟文档数据
    mock_docs = [
        {
            'content': '张三与李四在一个项目中合作，他们负责开发公司的AI产品。',
            'metadata': {'source': '项目记录', 'type': '团队信息'},
            'retrieval_step': 1,
            'search_target': '张三',
            'purpose': '查找张三的相关信息'
        },
        {
            'content': '李四是项目经理，负责管理飞天项目的整体进度和团队协调。',
            'metadata': {'source': '项目记录', 'type': '职责信息'},
            'retrieval_step': 2,
            'search_target': '李四',
            'purpose': '查找李四的项目信息'
        }
    ]
    
    print(f"模拟检索到 {len(mock_docs)} 个文档:")
    for i, doc in enumerate(mock_docs, 1):
        print(f"  {i}. {doc['content'][:50]}...")
    
    print("✅ Mock Retriever 测试完成")
    return mock_docs

def test_analyzer_agent(mock_docs):
    """
    测试Analyzer Agent
    """
    print("\n🧪 测试 Analyzer Agent...")
    
    if DEEPSEEK_API_KEY == "sk-your-deepseek-api-key":
        print("⚠️ 跳过Analyzer Agent测试 - 需要配置DeepSeek API密钥")
        return
    
    try:
        analyzer = AnalyzerAgent()
        
        test_query = "张三参与了哪个项目？"
        test_plan = {
            "query_type": "multi_hop",
            "key_entities": ["张三"],
            "reasoning_steps": [
                {
                    "step": 1,
                    "action": "search",
                    "target": "张三",
                    "purpose": "查找张三的相关信息"
                }
            ],
            "expected_hops": 2
        }
        
        print(f"查询: {test_query}")
        print(f"使用 {len(mock_docs)} 个文档进行分析")
        
        try:
            analysis = analyzer.analyze_documents(mock_docs, test_query, test_plan)
            print(f"分析结果: {analysis}")
        except Exception as e:
            print(f"❌ 分析失败: {e}")
        
        print("✅ Analyzer Agent 测试完成")
        
    except Exception as e:
        print(f"❌ Analyzer Agent 测试失败: {e}")

def test_system_integration():
    """
    测试系统集成
    """
    print("\n🧪 测试系统集成...")
    
    # 检查配置
    print(f"DeepSeek API Key: {'已配置' if DEEPSEEK_API_KEY != 'sk-your-deepseek-api-key' else '未配置'}")
    
    # 检查模块导入
    try:
        from models.multi_agent_rag import MultiAgentRAGSystem
        print("✅ 多Agent系统模块导入成功")
    except Exception as e:
        print(f"❌ 多Agent系统模块导入失败: {e}")
    
    # 检查CLI应用
    try:
        from cli_app import AgenticRAGCLI
        print("✅ CLI应用模块导入成功")
    except Exception as e:
        print(f"❌ CLI应用模块导入失败: {e}")
    
    print("✅ 系统集成测试完成")

def main():
    """
    主测试函数
    """
    print("🚀 开始系统功能测试")
    print("=" * 50)
    
    # 测试各个组件
    test_system_integration()
    test_planner_agent()
    mock_docs = test_mock_retriever()
    test_analyzer_agent(mock_docs)
    
    print("\n" + "=" * 50)
    print("🎉 系统功能测试完成")
    
    # 提供使用建议
    print("\n💡 使用建议:")
    if DEEPSEEK_API_KEY == "sk-your-deepseek-api-key":
        print("1. 请在 config/settings.py 中配置正确的 DeepSeek API 密钥")
    print("2. 运行 'python cli_app.py' 启动交互模式")
    print("3. 运行 'python cli_app.py -q \"张三参与了哪个项目？\"' 进行单次查询")
    print("4. 运行 'python cli_app.py -b test_queries.txt' 进行批处理测试")

if __name__ == "__main__":
    main()