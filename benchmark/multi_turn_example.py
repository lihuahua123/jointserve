import time
from openai import OpenAI
import aiohttp
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union
import json
import asyncio
@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)



user_messages = [
    "Tell me your ten favourite films",
    "Who directed each of these films?",
    "Which director has the most experience?",
    "What other films has this director directed?",
    "Do these films have anything in common?",
    "Which of those films is the oldest?",
    "How old was the director when this was released?",
]

api_url = "http://localhost:30000/v1/chat/completions"
assert api_url.endswith(
    "chat/completions"
), "OpenAI Chat Completions API URL must end with 'chat/completions'."
request_func_input = RequestFuncOutput()
request_func_input.use_beam_search = False
request_func_input.model="/hy-tmp/"
request_func_input.prompt=[{
        "role": "user",
        "content": user_messages[0],
    }]
request_func_input.output_len = 200
request_func_input.prompt_len = len(request_func_input.prompt)
async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                                "data: ")
                        
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                        most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            print(e)
            output.success = False
    return output
begin = time.time()       
asyncio.run(async_request_openai_chat_completions(request_func_input=request_func_input))
end = time.time()
print(end - begin)