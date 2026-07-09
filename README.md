# AgentRag
将本地文件数据转化成faiss和MB25数据并保存。避免启动时重复生成。

## 模型使用
在NVIDIA[官网](https://build.nvidia.com/) 注册账号并添加api-key使用

## 启动方法
直接运行 python law_rag.py

## 效果比对
使用deepseek获取的问题与答案
<img width="975" height="704" alt="image" src="https://github.com/user-attachments/assets/2191539d-a90c-4067-91ca-abf72727b820" />

使用本地的RAG提取信息后
<img width="1754" height="805" alt="image" src="https://github.com/user-attachments/assets/620416ae-cc4f-46bd-a97c-8837f7e4f2ff" />

## 工作流程
```
启动程序
   │
   ├─> 初始化 LawRagAgent（加载 Embedding/Rerank/Chat 模型）
   │
   ├─> 调用 get_ensemble_retriever()
   │     ├─> 检查 FAISS 和 BM25 索引是否存在
   │     │     ├─> 若存在 → 直接加载
   │     │     └─> 若不存在 → 扫描 data/ 目录下的 PDF
   │     │          ├─> 读取并切分文本
   │     │          ├─> 创建 FAISS 索引并保存
   │     │          └─> 创建 BM25 索引并保存
   │     └─> 返回 EnsembleRetriever
   │
   └─> 进入交互循环（while True）
         ├─> 接收用户问题 query
         ├─> 调用 chat(query, ensemble_retriever)
         │     ├─> ensemble_retriever.invoke(query) → 混合检索候选（10+10 融合）
         │     ├─> cross_encoder.compress_documents() → 重排，取 top_n=5
         │     ├─> 格式化上下文（[片段1] ... [片段5]）
         │     ├─> 构造最终 Prompt：问题 + 上下文
         │     ├─> agent.invoke() → LLM 生成回答
         │     └─> 打印回答
         └─> 继续下一轮提问
```

## 关键代码细节
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