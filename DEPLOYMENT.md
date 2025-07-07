# Notta 部署指南

本文档提供了Notta多Agent RAG系统的详细部署指南，适用于开发、测试和生产环境。

## 🚀 快速部署

### 1. 环境准备

#### 系统要求
- Python 3.11+
- 8GB+ RAM（用于嵌入模型）
- 2GB+ 磁盘空间
- 稳定的网络连接（用于API调用）

#### 获取DeepSeek API Key
1. 访问 [DeepSeek官网](https://deepseek.com)
2. 注册账号并获取API Key
3. 确保账户有足够的API调用额度

### 2. 项目部署

#### 方式一：Git克隆（推荐）
```bash
# 克隆项目
git clone <your-github-repo-url>
cd notta

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：ZIP包部署
```bash
# 解压ZIP包
unzip notta-main.zip
cd notta-main

# 后续步骤同上
```

### 3. 环境配置

#### 设置API Key
```bash
# Windows
set DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Linux/Mac
export DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 或创建.env文件
echo "DEEPSEEK_API_KEY=your_deepseek_api_key_here" > .env
```

#### 验证配置
```bash
# 检查Python版本
python --version

# 检查依赖安装
pip list | grep -E "agno|langchain|sentence-transformers"

# 检查API Key
echo $DEEPSEEK_API_KEY
```

### 4. 初始化系统

```bash
# 初始化知识库
python cli_app.py --setup-kb

# 验证系统
python cli_app.py -q "测试查询" --threshold 0.3
```

## 🔧 高级配置

### 自定义配置

编辑 `config/settings.py` 文件：

```python
# DeepSeek模型配置
DEEPSEEK_MODEL = "deepseek-chat"  # 或 "deepseek-reasoner"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# RAG系统参数
MAX_ITERATIONS = 3              # 最大推理迭代次数
SIMILARITY_THRESHOLD = 0.7       # 文档相似度阈值
MAX_DOCUMENTS = 5                # 最大检索文档数
CHUNK_SIZE = 1000               # 文档分块大小
CHUNK_OVERLAP = 200             # 分块重叠大小

# 嵌入模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "./faiss_index"
```

### 性能优化

#### 1. 嵌入模型优化
```python
# 使用GPU加速（如果可用）
model_kwargs = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 使用更大的模型（更好的效果，更慢的速度）
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

#### 2. 向量存储优化
```python
# 调整FAISS索引参数
index_params = {
    'nlist': 100,  # 聚类数量
    'nprobe': 10   # 搜索聚类数
}
```

## 🐳 Docker部署

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV DEEPSEEK_API_KEY=""

# 暴露端口（如果使用Web界面）
EXPOSE 8501

# 启动命令
CMD ["python", "cli_app.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  notta:
    build: .
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./faiss_index:/app/faiss_index
      - ./logs:/app/logs
    ports:
      - "8501:8501"
    restart: unless-stopped
```

### 部署命令
```bash
# 构建镜像
docker build -t notta:latest .

# 运行容器
docker run -d \
  --name notta \
  -e DEEPSEEK_API_KEY=your_api_key \
  -v $(pwd)/faiss_index:/app/faiss_index \
  -p 8501:8501 \
  notta:latest
```

## ☁️ 云端部署

### AWS部署

#### 1. EC2实例
```bash
# 选择实例类型
# t3.large (2 vCPU, 8GB RAM) - 推荐用于生产
# t3.medium (2 vCPU, 4GB RAM) - 适用于测试

# 安全组配置
# 入站规则：SSH (22), HTTP (80), HTTPS (443), Custom (8501)
```

#### 2. 部署脚本
```bash
#!/bin/bash
# deploy.sh

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和Git
sudo apt install -y python3.11 python3.11-venv git

# 克隆项目
git clone <your-repo-url>
cd notta

# 设置虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
echo "export DEEPSEEK_API_KEY=your_api_key" >> ~/.bashrc
source ~/.bashrc

# 初始化系统
python cli_app.py --setup-kb

# 设置systemd服务
sudo cp notta.service /etc/systemd/system/
sudo systemctl enable notta
sudo systemctl start notta
```

### Azure部署

#### 使用Azure Container Instances
```bash
# 创建资源组
az group create --name notta-rg --location eastus

# 部署容器
az container create \
  --resource-group notta-rg \
  --name notta-app \
  --image your-registry/notta:latest \
  --environment-variables DEEPSEEK_API_KEY=your_api_key \
  --ports 8501 \
  --cpu 2 \
  --memory 8
```

## 🔍 监控和维护

### 日志配置

```python
# config/logging.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                'logs/notta.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

### 健康检查

```python
# health_check.py
import requests
import sys

def health_check():
    try:
        # 检查API连接
        response = requests.get(
            "https://api.deepseek.com/health",
            timeout=10
        )
        
        # 检查向量存储
        from services.vector_store import VectorStoreService
        vector_store = VectorStoreService()
        vector_store.load_vector_store()
        
        print("✅ 系统健康检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

if __name__ == "__main__":
    if not health_check():
        sys.exit(1)
```

### 性能监控

```python
# monitoring.py
import psutil
import time
from datetime import datetime

def monitor_system():
    while True:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        print(f"[{datetime.now()}] CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
        
        # 告警阈值
        if cpu_percent > 80 or memory_percent > 80:
            print("⚠️ 系统资源使用率过高")
        
        time.sleep(60)  # 每分钟检查一次
```

## 🔧 故障排除

### 常见问题

#### 1. API连接失败
```bash
# 检查网络连接
curl -I https://api.deepseek.com

# 验证API Key
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/models
```

#### 2. 嵌入模型加载失败
```bash
# 清理缓存
rm -rf ~/.cache/huggingface/

# 手动下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

#### 3. 向量存储问题
```bash
# 重新初始化
rm -rf faiss_index/
python cli_app.py --setup-kb
```

#### 4. 内存不足
```bash
# 检查内存使用
free -h

# 优化配置
# 减少CHUNK_SIZE和MAX_DOCUMENTS
# 使用更小的嵌入模型
```

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python cli_app.py -q "测试" --debug

# 性能分析
python -m cProfile -o profile.stats cli_app.py -q "测试"
```

## 📊 性能基准

### 测试环境
- CPU: Intel i7-10700K
- RAM: 16GB DDR4
- 存储: NVMe SSD
- 网络: 100Mbps

### 性能指标

| 查询类型 | 响应时间 | 内存使用 | CPU使用 |
|----------|----------|----------|----------|
| 简单查询 | 3-5秒 | 2-3GB | 20-30% |
| 复杂推理 | 8-12秒 | 3-4GB | 40-60% |
| 批处理 | 15-30秒 | 4-6GB | 60-80% |

### 优化建议

1. **硬件优化**
   - 使用SSD存储提高I/O性能
   - 增加RAM减少模型加载时间
   - 使用GPU加速嵌入计算

2. **软件优化**
   - 启用模型缓存
   - 使用连接池管理API调用
   - 实现异步处理

3. **网络优化**
   - 使用CDN加速模型下载
   - 配置API调用重试机制
   - 实现请求限流

## 🔐 安全配置

### API Key管理
```bash
# 使用环境变量
export DEEPSEEK_API_KEY="sk-xxx"

# 使用密钥管理服务
# AWS Secrets Manager
# Azure Key Vault
# HashiCorp Vault
```

### 网络安全
```bash
# 防火墙配置
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8501/tcp

# SSL/TLS配置
# 使用nginx反向代理
# 配置Let's Encrypt证书
```

### 数据安全
```python
# 敏感数据加密
from cryptography.fernet import Fernet

def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()
```

## 📈 扩展部署

### 负载均衡
```nginx
# nginx.conf
upstream notta_backend {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://notta_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 数据库集成
```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseManager:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def store_query_result(self, query, result, metadata):
        # 存储查询结果用于分析和优化
        pass
```

### 微服务架构
```yaml
# docker-compose.yml
version: '3.8'

services:
  notta-api:
    build: ./api
    ports:
      - "8000:8000"
  
  notta-worker:
    build: ./worker
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: notta
      POSTGRES_USER: notta
      POSTGRES_PASSWORD: password
```

---

## 📞 技术支持

如遇到部署问题，请：

1. 查看日志文件：`logs/notta.log`
2. 运行健康检查：`python health_check.py`
3. 提交GitHub Issue
4. 联系技术支持团队

**祝您部署顺利！** 🚀