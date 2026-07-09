# AgentRag
将本地文件数据转化成faiss和MB25数据并保存。避免启动时重复生成。

## 多数据库加载
将faiss按照文件生成并保存，启动时直接读取每一个faiss文件库并合并。
将bm25数据按照文成生成并保存，启动直接读取每一个bm25文件库使用EnsembleRetriever进行融合。

## 模型使用
在NVIDIA[官网](https://build.nvidia.com/) 注册账号并添加api-key使用

## bm25s集成
编写bm25s_retriever.py文件，实现保存和加载，并于适配langchain_core的retrievers。

## 文件形式
所有原始文档放在data目录下，文件形式为pdf

## 启动方法
直接运行 python law_rag.py

## 效果比对
使用deepseek获取的问题与答案
<img width="975" height="704" alt="image" src="https://github.com/user-attachments/assets/2191539d-a90c-4067-91ca-abf72727b820" />

使用本地的RAG提取信息后
<img width="1754" height="805" alt="image" src="https://github.com/user-attachments/assets/620416ae-cc4f-46bd-a97c-8837f7e4f2ff" />

## 工作流程图
<img width="3522" height="5174" alt="mermaid_20260709_0112f6" src="https://github.com/user-attachments/assets/dabf8f11-7c19-4f50-b5f8-0c84b29ecf34" />

## 核心组件交互时序图
<img width="4656" height="2979" alt="mermaid_20260709_fb4568" src="https://github.com/user-attachments/assets/337b27c8-ea5b-463e-9aff-3236af63458c" />

## 关键代码细节
```
1. 离线索引构建（每个 PDF 独立存储）
① 读取并切分 PDF（split_pdf_file 生成器）
python
def split_pdf_file(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            text = read_data(file_path, file)          # 提取全文
            split_cur_text = get_split_test(text, file) # 切分为块
            yield split_cur_text, file                 # 逐文件返回
细节：
使用 PyPDFLoader 读取全文。
RecursiveCharacterTextSplitter 切块：chunk_size=200, overlap=20，分隔符优先中文标点。

② 构建 FAISS 索引（text_to_faiss）
python
def text_to_faiss(split_text, embedding_model, faiss_db_path="faiss_index"):
    if os.path.exists(faiss_db_path):
        return None  # 已存在则跳过
    faiss_db = FAISS.from_texts(split_text, embedding_model)
    faiss_db.save_local(faiss_db_path)
细节：
embedding_model 是 NVIDIAEmbeddings，向量维度 1024。
索引保存为本地文件夹（包含 index.pkl 和 FAISS 文件）。

③ 构建 BM25 索引（text_to_bm25）
python
def text_to_bm25(split_text=None, bm25_db="bm_25_index"):
    if os.path.exists(bm25_db):
        return None
    bm_25_db = BM25sRetriever.from_texts(split_text)
    bm_25_db.k = NUM_K
    bm_25_db.save(bm25_db)
细节：
BM25sRetriever.from_texts() 内部使用 jieba 分词，并调用 bm25s 索引。
保存为文件夹，后续可 load 加载。

2. 索引加载与合并
④ 合并所有 FAISS 索引（get_faiss_data）
python
def get_faiss_data(faiss_save_dir: str):
    # 加载第一个索引
    faiss_db_path = os.path.join(faiss_save_dir, os.listdir(faiss_save_dir)[0])
    faiss_db = FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
    
    # 合并后续索引
    for sub_dir in os.listdir(faiss_save_dir)[1:]:
        path = os.path.join(faiss_save_dir, sub_dir)
        sub_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        faiss_db.merge_from(sub_db)   # 关键：合并
    
    retriever = faiss_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": NUM_K, "score_threshold": 0.8}
    )
    return retriever
注意：search_type="mmr" 与 score_threshold 不兼容，建议改为 search_type="similarity" 或调整参数。

⑤ 加载所有 BM25 并组合（get_bm25_data）
python
def get_bm25_data(bm25_save_dir: str):
    bm25_retrievers = []
    for sub_dir in os.listdir(bm25_save_dir):
        path = os.path.join(bm25_save_dir, sub_dir)
        retriever = BM25sRetriever.load(path)
        bm25_retrievers.append(retriever)
    # 将所有 BM25 检索器组合为一个 EnsembleRetriever
    ensemble_bm25 = EnsembleRetriever(retrievers=bm25_retrievers)
    return ensemble_bm25
细节：这里使用默认等权重，后续可以与 FAISS 再次组合。

3. 在线问答流程（LawRagAgent.chat）
⑥ 混合检索 + 重排 + 生成
python
def chat(self, query: str, ensemble_retriever):
    # 1. 混合检索
    retriever_doc = ensemble_retriever.invoke(query)[:NUM_K*2]  # 取前 20 个候选
    
    # 2. 重排序（取 top_n=5）
    advanced_retriever = self.cross_encoder.compress_documents(
        query=query,
        documents=[Document(page_content=p.page_content) for p in retriever_doc],
    )
    
    # 3. 格式化上下文
    context_blocks = []
    for i, hit in enumerate(advanced_retriever, 1):
        context_blocks.append(f"[片段{i}]\n{hit.page_content}")
    context = "\n\n".join(context_blocks)
    
    # 4. 构造 Prompt
    user_prompt = f"问题：{query}\n\n上下文：{context}"
    
    # 5. Agent 生成
    result = self.agent.invoke({"messages": HumanMessage(user_prompt)})
    final_msg = result["messages"][-1]
    final_msg.pretty_print()
细节：
重排模型 NVIDIARerank 的 top_n=5 在初始化时指定。
Agent 的 system_prompt 强制要求按格式输出（【问题答案】【法律依据】）。
```
