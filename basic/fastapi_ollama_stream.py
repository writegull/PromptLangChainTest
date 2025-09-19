# 功能说明：FastAPI实现Ollama流式响应

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

app = FastAPI(title="Ollama-OpenAI兼容API", version="1.0.0")

# Ollama服务器地址
OLLAMA_BASE_URL = "http://localhost:11434"


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


async def convert_to_ollama_messages(openai_messages: List[Message]) -> List[Dict]:
    """将OpenAI格式的消息转换为Ollama格式"""
    ollama_messages = []
    for msg in openai_messages:
        ollama_messages.append({"role": msg.role, "content": msg.content})
    return ollama_messages


async def stream_ollama_response(model: str, messages: List[Dict], temperature: float):
    """从Ollama获取流式响应"""
    async with httpx.AsyncClient() as client:
        # 准备Ollama请求数据
        ollama_data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        # 发送请求到Ollama
        async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_data,
                timeout=30.0
        ) as response:
            if response.status_code != 200:
                error = await response.aread()
                raise HTTPException(status_code=response.status_code, detail=error.decode())

            # 流式处理Ollama响应
            async for chunk in response.aiter_lines():
                if chunk.strip():
                    try:
                        data = json.loads(chunk)
                        if data.get("done", False):
                            break
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成端点"""

    if not request.stream:
        # 非流式响应（简单实现）
        async with httpx.AsyncClient() as client:
            ollama_messages = await convert_to_ollama_messages(request.messages)
            ollama_data = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                }
            }

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_data,
                timeout=30.0
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            ollama_response = response.json()

            # 转换为OpenAI格式
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ollama_response.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Ollama不返回token计数
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    else:
        # 流式响应
        ollama_messages = await convert_to_ollama_messages(request.messages)

        async def generate():
            # 生成OpenAI格式的流式响应
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_time = int(datetime.now().timestamp())
            model_name = request.model

            # 发送初始响应
            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created_time,
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            })}\n\n"

            # 处理Ollama流式响应
            async for content_chunk in stream_ollama_response(
                    request.model, ollama_messages, request.temperature
            ):
                yield f"data: {json.dumps({
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model_name,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': content_chunk},
                        'finish_reason': None
                    }]
                })}\n\n"

            # 发送结束标记
            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created_time,
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            })}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": model["name"],
                            "object": "model",
                            "created": int(datetime.now().timestamp()),
                            "owned_by": "ollama"
                        } for model in models
                    ]
                }
            else:
                return {
                    "object": "list",
                    "data": []
                }
    except Exception:
        return {
            "object": "list",
            "data": []
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8012)