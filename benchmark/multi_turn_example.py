import time
from openai import OpenAI
import aiohttp
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union
import json
import asyncio
import jsonlines
@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    need_cache: bool = False
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
    total_latency_in_engine: float = 0.0
    waiting_latency: float = 0.0
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

api_url = "http://localhost:9999/v1/chat/completions"
assert api_url.endswith(
    "chat/completions"
), "OpenAI Chat Completions API URL must end with 'chat/completions'."
async def send_cache_flush_request():
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            flush_url = "http://localhost:30000/flush_cache"
            async with session.get(url=flush_url) as response:
                assert response.status == 200
        except Exception as e:
            print(e)
            
async def send_cache_metric_request():
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            flush_url = "http://localhost:30000/cache_metrics"
            async with session.get(url=flush_url) as response:
                assert response.status == 200
        except Exception as e:
            print(e)    
                  
async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": request_func_input.prompt,
            "temperature": 0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "need_cache": request_func_input.need_cache,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        output = RequestFuncOutput()
        output.prompt_len = len(request_func_input.prompt)

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
                    output.total_latency_in_engine = data["arrival_time"]
                    output.waiting_latency = data["arrival_time"] - data["begin_to_run_time"]
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


async def client(idx, message_hostory_data,sleep_time=0):
    request_func_input = RequestFuncOutput()
    request_func_input.use_beam_search = False
    request_func_input.model="/hy-tmp/"
    request_func_input.output_len = 100
    request_func_input.prompt = []
    ttfts = []
    for index, message in enumerate(message_hostory_data):
        if sum([len(p["content"]) for p in request_func_input.prompt]) + len(message) + request_func_input.output_len > 4000:
            break
        request_func_input.need_cache = True
        # if index < 5:
        #     request_func_input.need_cache = True
        # elif index == 5:
        #     request_func_input.need_cache = False
        # else:
        #     request_func_input.need_cache = True
        request_func_input.prompt.append({
            "role": "user",
            "content": message,
        })
        result = await async_request_openai_chat_completions(request_func_input=request_func_input)
        await asyncio.sleep(sleep_time)
        request_func_input.prompt.append({
            "role": "assistant",
            "content": result.generated_text
        })
        # print(f"itl:{result.itl},latency:{result.latency},prompt_len:{result.prompt_len},ttft:{result.ttft},total_latency_in_engine:{result.total_latency_in_engine},waiting_latency:{result.waiting_latency}")
        # print(f"waiting_latency:{result.waiting_latency}")
        ttfts.append(result.ttft)
    # print("conversation turns:", index,"request_func_input.prompt",request_func_input.prompt)
    print(idx, "conversation turns:", index)
async def test():
    dataset_path = "/hy-tmp/sharegpt.json"
    dataset = []
    message_hostory_dataset = []
    # with open(dataset_path, "r+", encoding="utf8") as f:
    #     for item in jsonlines.Reader(f):
    #         dataset.append(item)
    with open(dataset_path) as f:
        dataset = json.load(f)
    for data in dataset:
        conversation_len = len(data["conversations"])
        message_history = []
        if  conversation_len==0 or conversation_len % 2 !=0 :
            continue
        for i in range(0,conversation_len,2):
            message_history.append(data["conversations"][i]["value"])
        message_hostory_dataset.append(message_history)
    client_tasks = []
    for idx, message_hostory_data in enumerate(message_hostory_dataset[:50]):
        # 阅读速度（300～500）字／分钟
        sleep = 0
        client_tasks.append(asyncio.create_task(client(idx,message_hostory_data,sleep)))
    for task in client_tasks:
        await task
    
    
begin = time.time()
asyncio.run(send_cache_flush_request())       
asyncio.run(test())
asyncio.run(send_cache_metric_request())   
end = time.time()
print(end - begin)