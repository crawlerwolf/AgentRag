# -*- coding: utf-8 -*-
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from multi_law_agent import MultiLawAgent

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("law_rag_api")

# ---------- 全局 agent 实例 ----------
law_agent: Optional[MultiLawAgent] = None


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="用户输入的法律问题")
    thread_id: Optional[str] = Field(description="会话ID，不传则自动生成")


class ChatResponse(BaseModel):
    answer: str = Field(description="法律助手的回答")
    thread_id: str = Field(description="使用的会话ID")
    cached: bool = Field(description="是否命中历史缓存")


class ClearRequest(BaseModel):
    thread_id: str = Field(description="要清除的会话ID")


class ClearResponse(BaseModel):
    message: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    retriever_ready: bool


# ---------- 应用生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Agent 和检索器"""
    global law_agent
    logger.info("正在启动 LawRagAgent...")
    law_agent = MultiLawAgent()
    await law_agent.init_agent()  # 加载功能agent
    # 初始化检索器（如果已有索引则加载，否则会尝试从 data/ 目录创建）
    await law_agent.init_retriever()
    logger.info("检索器初始化完成，API 服务就绪")
    yield


# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="法律 RAG 助手 API",
    description="基于 RAG 架构的企业级法律问答服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置（允许所有来源，生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_agent():
    if law_agent is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查"""
    agent_ready = law_agent is not None
    retriever_ready = law_agent.retriever is not None if law_agent else False
    return HealthResponse(
        status="ok" if agent_ready and retriever_ready else "degraded",
        agent_ready=agent_ready,
        retriever_ready=retriever_ready,
    )


@app.post("/clear_chat", response_model=ClearResponse)
async def clear_session(req: ClearRequest):
    """清除指定会话的所有检查点（包括子Agent的状态）"""
    check_agent()

    try:
        await law_agent.clear_session(req.thread_id)
        return ClearResponse(
            message=f"会话 {req.thread_id} 已清除",
            thread_id=req.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除会话失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    法律问答接口
    - **query**: 用户问题
    - **thread_id**: 可选，会话ID；若未提供则自动生成新会话
    """
    check_agent()
    if law_agent.retriever is None:
        raise HTTPException(status_code=500, detail="检索器未就绪，请稍后重试")

    # 生成或使用传入的会话ID
    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(f"收到请求 | thread_id={thread_id} | query={request.query[:80]}...")

    try:
        final_msg = await law_agent.get_message(request.query, thread_id)
        logger.info(f"回答生成完成 | thread_id={thread_id}")
        return ChatResponse(answer=final_msg, thread_id=thread_id, cached=False)

    except Exception as e:
        logger.exception("处理请求时发生错误")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "law_rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议关闭 reload
        log_level="info",
    )
