# -*- coding: utf-8 -*-
import json
import os

import jieba
from bm25s import BM25, tokenize
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class BM25sRetriever(BaseRetriever):
    """基于 bm25s 库的自定义检索器，实现 BaseRetriever 接口。"""

    k: int = 4
    bm25: Optional[BM25] = None
    corpus: List[str] = []

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, k: int = 4, bm25: Optional[BM25] = None, corpus: List[str] = None):
        # 由于 Pydantic v2 不推荐直接覆盖 __init__，我们使用 super() 并通过父类初始化
        super().__init__(k=k, bm25=bm25, corpus=corpus or [])

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索相关文档"""
        if self.bm25 is None:
            return []

        query_ = " ".join(jieba.cut(query))
        query_tokens = tokenize(query_, stopwords="chinese")

        results, scores = self.bm25.retrieve(query_tokens, k=self.k)
        documents = []
        for i in range(results.shape[1]):
            doc_text, score = results[0, i], scores[0, i]
            documents.append(Document(page_content=self.corpus[doc_text]))
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """异步检索（直接复用同步方法）"""
        return self._get_relevant_documents(query, run_manager=run_manager)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        k: int = 4,
        # 可添加其他 bm25s 参数，如 tokenizer 等
    ) -> "BM25sRetriever":
        """从文本列表创建 BM25sRetriever 实例并索引。"""
        corpus_ = [" ".join(jieba.cut(text)) for text in texts]
        corpus_tokens = tokenize(corpus_, stopwords="chinese")
        bm25 = BM25()
        bm25.index(corpus_tokens)
        return cls(k=k, bm25=bm25, corpus=texts)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        k: int = 4,
    ) -> "BM25sRetriever":
        """从 Document 列表创建。"""
        texts = [doc.page_content for doc in documents]
        return cls.from_texts(texts, k=k)

    def save(self, save_dir: str) -> None:
        """
        将 BM25 索引和元数据保存到指定目录。
        目录下会保存两个文件：
          - bm25_index/  (bm25s 自带的索引文件夹)
          - meta.json   (包含 k 和 corpus)
        """
        os.makedirs(save_dir, exist_ok=True)
        # 保存 bm25s 索引
        self.bm25.save(save_dir)
        # 保存元数据
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "k": self.k,
                "corpus": self.corpus,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, save_dir: str) -> "BM25sRetriever":
        """
        从目录加载检索器。
        """
        # 加载元数据
        with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 加载 bm25s 索引
        bm25 = BM25.load(save_dir)
        return cls(k=meta["k"], bm25=bm25, corpus=meta["corpus"])
