"""
配置文件，包含所有常量和配置项
"""

# 1. 文件路径
VECTOR_STORE_PATH = "faiss_index"
HISTORY_FILE = "chat_history.json"

# 2. 模型配置
DEFAULT_MODEL = "deepseek-chat"
AVAILABLE_MODELS = ["deepseek-chat", "deepseek-reasoner"]

# DeepSeek API配置
DEEPSEEK_API_KEY = "your_deepseek_api_key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EMBEDDING_MODEL = "bge-m3:latest"
AVAILABLE_EMBEDDING_MODELS = ["bge-m3:latest", "nomic-embed-text:latest", "mxbai-embed-large:latest", "bge-large-en-v1.5:latest", "bge-large-zh-v1.5:latest"]
EMBEDDING_BASE_URL = "http://localhost:11434"

# 3. RAG配置
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 30
MAX_RETRIEVED_DOCS = 3

# 4. LangChain配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 5. 对话历史配置
MAX_HISTORY_TURNS = 5