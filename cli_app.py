#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic RAG CLI应用
支持多跳推理的命令行界面
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from models.multi_agent_rag import MultiAgentRAGSystem
from services.vector_store import VectorStoreService
from utils.document_processor import DocumentProcessor
from config.settings import (
    DEEPSEEK_API_KEY,
    DEFAULT_SIMILARITY_THRESHOLD,
    VECTOR_STORE_PATH
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgenticRAGCLI:
    """
    Agentic RAG命令行应用
    """
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.document_processor = DocumentProcessor()
        self.rag_system = MultiAgentRAGSystem(self.vector_store)
        
    def setup_knowledge_base(self, documents_path: str = None):
        """
        设置知识库
        
        Args:
            documents_path: 文档路径（可选）
        """
        print("🔧 正在设置知识库...")
        
        if documents_path and Path(documents_path).exists():
            print(f"📁 从 {documents_path} 加载文档...")
            # 这里可以添加文档加载逻辑
            # 由于原项目使用Streamlit上传，这里我们使用Mock数据
        
        # 创建Mock知识库数据
        self._create_mock_knowledge_base()
        print("✅ 知识库设置完成")
    
    def _create_mock_knowledge_base(self):
        """
        创建Mock知识库数据，用于演示多跳推理
        """
        mock_documents = [
            {
                "content": "张三是一名资深的软件工程师，目前在北京科技公司工作。他擅长Python和机器学习技术。",
                "metadata": {"source": "员工档案", "type": "个人信息"}
            },
            {
                "content": "张三与李四在同一个项目组中合作，他们负责开发公司的AI产品。",
                "metadata": {"source": "项目记录", "type": "团队信息"}
            },
            {
                "content": "李四是项目经理，负责管理飞天项目的整体进度和团队协调。",
                "metadata": {"source": "项目记录", "type": "职责信息"}
            },
            {
                "content": "飞天项目是公司的重点AI项目，旨在开发下一代智能对话系统。",
                "metadata": {"source": "项目文档", "type": "项目描述"}
            },
            {
                "content": "飞天项目团队包括张三、李四、王五等多名工程师，预计2024年完成。",
                "metadata": {"source": "项目文档", "type": "团队组成"}
            },
            {
                "content": "王五负责飞天项目的前端开发工作，与张三的后端开发形成配合。",
                "metadata": {"source": "项目记录", "type": "分工信息"}
            },
            {
                "content": "公司还有另一个项目叫做星辰项目，由赵六负责，专注于数据分析平台。",
                "metadata": {"source": "项目文档", "type": "其他项目"}
            },
            {
                "content": "赵六是数据科学家，专门负责星辰项目的算法设计和数据处理。",
                "metadata": {"source": "员工档案", "type": "个人信息"}
            },
            {
                "content": "李四之前还参与过云端项目，该项目已于2023年成功上线。",
                "metadata": {"source": "项目历史", "type": "历史记录"}
            },
            {
                "content": "张三在加入飞天项目之前，曾经在智能助手项目中担任核心开发者。",
                "metadata": {"source": "员工档案", "type": "工作经历"}
            }
        ]
        
        # 将Mock数据添加到向量存储
        for i, doc in enumerate(mock_documents):
            # 调用向量存储的添加方法
            success = self.rag_system.vector_store.add_document(
                content=doc["content"],
                metadata=doc["metadata"]
            )
            if success:
                logger.info(f"添加文档 {i+1}: {doc['content'][:50]}...")
            else:
                logger.error(f"添加文档 {i+1} 失败")
        
        print(f"📚 已添加 {len(mock_documents)} 个文档到知识库")
    
    def interactive_mode(self):
        """
        交互模式
        """
        print("\n🚀 欢迎使用 Agentic RAG 多跳推理系统！")
        print("💡 支持的查询示例：")
        print("   - 张三参与了哪个项目？")
        print("   - 飞天项目的团队成员有哪些？")
        print("   - 李四负责什么工作？")
        print("\n输入 'quit' 或 'exit' 退出程序\n")
        
        while True:
            try:
                user_input = input("🤖 请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🔍 正在处理查询: {user_input}")
                print("=" * 60)
                
                # 处理查询
                result = self.process_query(user_input)
                
                # 显示结果
                self._display_result(result)
                
                print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理查询时出错: {e}")
                print(f"❌ 处理查询时出错: {e}")
    
    def process_query(self, query: str, similarity_threshold: float = None) -> Dict[str, Any]:
        """
        处理单个查询
        
        Args:
            query: 用户查询
            similarity_threshold: 相似度阈值
            
        Returns:
            Dict: 处理结果
        """
        if similarity_threshold is None:
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        
        try:
            result = self.rag_system.process_query(query, similarity_threshold)
            return result
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return {
                'error': str(e),
                'user_query': query
            }
    
    def _display_result(self, result: Dict[str, Any]):
        """
        显示查询结果
        
        Args:
            result: 查询结果
        """
        if 'error' in result:
            print(f"❌ 错误: {result['error']}")
            return
        
        analysis = result.get('analysis', {})
        plan = result.get('plan', {})
        
        # 显示查询规划
        print(f"📋 查询类型: {plan.get('query_type', 'unknown')}")
        print(f"🎯 关键实体: {', '.join(plan.get('key_entities', []))}")
        print(f"🔄 迭代次数: {result.get('iterations', 0)}")
        print(f"📚 检索文档数: {result.get('total_documents', 0)}")
        
        # 显示推理链条
        reasoning_chain = analysis.get('reasoning_chain', [])
        if reasoning_chain:
            print("\n🧠 推理链条:")
            for i, step in enumerate(reasoning_chain, 1):
                print(f"   {i}. {step.get('fact', '')}")
                if step.get('source'):
                    print(f"      📖 来源: {step['source']}")
        
        # 显示最终结论
        conclusion = analysis.get('conclusion', '')
        if conclusion:
            print(f"\n✅ 最终答案: {conclusion}")
        
        # 显示置信度
        confidence = analysis.get('confidence', 0)
        print(f"📊 置信度: {confidence:.2f}")
        
        # 显示缺失信息
        missing_info = analysis.get('missing_info', [])
        if missing_info:
            print(f"⚠️ 缺失信息: {', '.join(missing_info)}")
    
    def batch_mode(self, queries_file: str, output_file: str = None):
        """
        批处理模式
        
        Args:
            queries_file: 查询文件路径
            output_file: 输出文件路径
        """
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = []
            
            for i, query in enumerate(queries, 1):
                print(f"\n处理查询 {i}/{len(queries)}: {query}")
                result = self.process_query(query)
                results.append(result)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {output_file}")
            else:
                for result in results:
                    self._display_result(result)
                    print("\n" + "-" * 40 + "\n")
                    
        except FileNotFoundError:
            print(f"❌ 文件未找到: {queries_file}")
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            print(f"❌ 批处理失败: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="Agentic RAG 多跳推理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cli_app.py                          # 交互模式
  python cli_app.py -q "张三参与了哪个项目？"    # 单次查询
  python cli_app.py -b queries.txt           # 批处理模式
  python cli_app.py --setup-kb               # 设置知识库
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='单次查询模式'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=str,
        help='批处理模式，指定查询文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出文件路径（批处理模式）'
    )
    
    parser.add_argument(
        '--setup-kb',
        action='store_true',
        help='设置知识库'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f'相似度阈值 (默认: {DEFAULT_SIMILARITY_THRESHOLD})'
    )
    
    parser.add_argument(
        '--docs-path',
        type=str,
        help='文档路径（设置知识库时使用）'
    )
    
    args = parser.parse_args()
    
    # 检查API密钥
    if DEEPSEEK_API_KEY == "sk-your-deepseek-api-key":
        print("❌ 请在 config/settings.py 中设置正确的 DEEPSEEK_API_KEY")
        sys.exit(1)
    
    # 创建CLI应用
    app = AgenticRAGCLI()
    
    try:
        if args.setup_kb:
            # 设置知识库模式
            app.setup_knowledge_base(args.docs_path)
        elif args.query:
            # 单次查询模式
            app.setup_knowledge_base()  # 确保知识库已设置
            print(f"🔍 处理查询: {args.query}")
            result = app.process_query(args.query, args.threshold)
            app._display_result(result)
        elif args.batch:
            # 批处理模式
            app.setup_knowledge_base()  # 确保知识库已设置
            app.batch_mode(args.batch, args.output)
        else:
            # 交互模式（默认）
            app.setup_knowledge_base()  # 确保知识库已设置
            app.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()