# AgentRag
多智能体法律咨询系统使用说明
# 1. 系统简介
本系统是一个基于 LangGraph 构建的多智能体法律咨询平台，能够自动根据用户问题分流到不同的专业智能体，并结合本地法律文档进行检索增强生成（RAG）。系统包含四个智能体：

 •	分流智能体（Triage）：分析用户输入，判断应交给哪个专家处理。

 •	法律研究智能体（Legal Research）：回答法律条文、司法解释、案例、法律程序等问题。

 •	合规审查智能体（Compliance Review）：对用户提供的合同条款、制度文件等给出合规性评价。

 •	通用回答智能体（General）：处理问候、闲聊等非法律问题。

所有智能体共用同一个 **checkpointer**（SQLite），支持多轮对话上下文的持久化与恢复，同时每个会话可通过**thread_id**进行隔离和清除。

# 2. 系统架构
```
用户输入
  ↓
[FastAPI 接口或命令行]
  ↓
MultiAgentState（图状态）
  ↓
分流节点（triage） → 条件路由
  ↓        ↓        ↓
法律研究  合规审查  通用回答
（RAG）    （RAG）   （直接回答）
  ↓        ↓        ↓
      END（结束）
```
 •	状态图：**MultiAgentState** 包含 messages（对话历史）、next（下一节点）、query（用户原始问题）。

 •	检索器：**EnsembleRetriever** 混合了 FAISS（向量检索）和 BM25（关键词检索），结果经 NVIDIA Rerank 重排。

 •	持久化：所有智能体和主图共用同一个**AsyncSqliteSaver**，通过不同前缀的 thread_id 实现状态隔离。

# 3. 环境要求
 •	Python 3.10+

 •	NVIDIA API Key（用于 embedding、rerank、chat 模型）

 •	依赖库见 requirements.txt

## 推荐安装步骤
```
git clone <your-repo-url>
cd multi-law-agent
pip install -r requirements.txt
```
**requirements.txt**中应包含以下核心包（示例）：

```
langgraph
langchain
langchain-community
langchain-nvidia-ai-endpoints
langchain-classic
langchain-text-splitters
aiosqlite
fastapi
uvicorn
faiss-cpu  (或 faiss-gpu)
python-dotenv
pydantic
```
# 4. 环境变量配置
在项目根目录创建 .env 文件，填写以下内容：

env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-qa-4
NVIDIA_RERANK_MODEL=nvidia/nv-rerank-qa-mistral-4b-v3
NVIDIA_CHAT_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
CHECKPOINTS_DB_PATH=./agent_db/agent_checkpoints.db   # 可选，默认会自动创建
 •	NVIDIA_API_KEY 为必填项，可从 NVIDIA NGC 获取。

 •	模型名称可根据实际可用模型调整。

 •	在NVIDIA[官网](https://build.nvidia.com/) 注册账号并添加api-key使用

# 5. 准备法律文档
系统依赖您提供的法律 PDF 文件来构建检索库。

1. 在项目根目录下创建 data 文件夹。

2. 将所有法律相关的 PDF 文件放入 data 文件夹中（如法律法规、司法解释、案例汇编等）。

首次运行时，系统会自动处理 PDF、生成向量索引和 BM25 索引（耗时取决于文件数量）。后续启动会直接加载已有索引。

# 6. 使用方式
## 6.1 命令行交互模式
直接运行主模块即可进入交互式对话：

```
python your_module_name.py
```
示例交互：

```
多 Agent 法律系统就绪。输入问题，输入 exit 退出。
输入你的问题: 什么是表见代理？
助手：[系统] 已分配至: legal_research
【问题答案】表见代理是指行为人虽无代理权，但相对人有理由相信其有代理权，该代理行为有效的制度。...

输入你的问题: 它和狭义无权代理有什么区别？
助手：...
```
输入 exit 会自动清除当前会话并关闭数据库连接。

## 6.2 FastAPI 服务模式
创建 app.py（见下方接口说明），启动服务：

```
python app.py
服务默认运行在 http://localhost:8000，Swagger 文档在 /docs。
```

# 7. 会话与上下文管理
 •	每轮对话必须使用相同的 thread_id 才能保持上下文（记忆）。

 •	系统内部为子智能体自动分配带前缀的 thread_id（如 triage_user-001、legal_research_user-001 等），避免状态类型冲突。

 •	可通过 /clear_chat 接口手动删除指定会话的所有记录。程序退出时不会自动清除，如需清除请主动调用。

# 8. 检索器索引管理
若法律文档有更新，可通过以下方法重建索引：

 •	方法一：删除 faiss_index 和 bm_25_index 文件夹，重新运行程序，系统会自动重新创建。

 •	方法二：在代码中调用 await system.file_to_ensemble() 强制清除旧索引并重新生成。

```
system = MultiLawAgent()
await system.file_to_ensemble(data_dir="data")
```

# 9. 效果比对
使用deepseek获取的问题与答案
<img width="975" height="704" alt="image" src="https://github.com/user-attachments/assets/2191539d-a90c-4067-91ca-abf72727b820" />

使用本地的RAG提取信息后
<img width="1754" height="805" alt="image" src="https://github.com/user-attachments/assets/620416ae-cc4f-46bd-a97c-8837f7e4f2ff" />


# 10. 关键代码细节
```
1. BM25sRetriever 自定义实现
代码中从 bm25s_retriever import BM25sRetriever, BM25，推测该模块实现了：

from_texts()：基于文本列表构建索引。

save() / load()：持久化（实际可能依赖 bm25s 的 save/load 方法）。

返回的 BM25sRetriever 实例继承了 BaseRetriever，可无缝用于 EnsembleRetriever。

2. 检索参数调优
FAISS 使用 search_type="mmr" 旨在增加多样性，避免重复内容；但 score_threshold=0.8 在 MMR 中不生效，应仅用于 similarity_score_threshold。

BM25 的 k=10 表示召回 10 篇。

混合权重 [0.6, 0.4] 强调语义检索稍强。

3. 重排器的输入限制
compress_documents() 的第一个参数是 query，第二个是文档列表。代码中将混合检索结果（retriever_doc）传入，但 retriever_doc 是 List[Document]，且已包含元数据。正确。

4. Agent 的使用
目前 Agent 未绑定任何工具，仅作为 LLM 调用器，作用等同于 model.invoke()，但保留了扩展性（未来可添加法律条文检索、计算等工具）。

5. 路径与持久化
FAISS 索引保存为 faiss_index 文件夹（内含 index.pkl 和 faiss_store 等）。

BM25 索引保存为 bm_25_index 文件夹（由 bm25s 生成）。

两个索引目录均支持再次加载，避免重复构建。
```
