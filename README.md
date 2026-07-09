# AgentRag
将本地文件数据转化成faiss和MB25数据并保存。避免启动时重复生成。

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


