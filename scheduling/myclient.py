import time
from openai import OpenAI
import asyncio
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, Set
import argparse
import sys
import json
import fastapi
import uvicorn
import aiohttp
import os
from fastapi import APIRouter, Request
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from joint.preble_scheduler import GlobalSchedulerWithTime
from joint.my_scheduler import GlobalScheduler
from joint.utils import RequestFuncOutput,ChatCompletionRequest,parse_chat_message_content,ConversationMessage,ModelConfig,remove_prefix
from transformers import AutoTokenizer
from typing_extensions import List
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional, Union
import copy
user_messages = [
    "Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality?", # noqa: E501
    "Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport"
]
user_messages2 = [
    "Tell me five famous films."
    "Who directed each of these films?",
    "Which director has the most experience?",
    "What other films has this director directed?",
    "Do these films have anything in common?",
    "Which of those films is the oldest?",
    "How old was the director when this was released?",
    ]

clients = [OpenAI(api_key="api_key", base_url="http://localhost:8000/v1"),OpenAI(api_key="api_key", base_url="http://localhost:8000/v1")]
# models = client.models.list() # 第一个是llm，后面是lora模型
model = "lora1" #models.data[0].id
chat_template = None #"{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content']+' [/user][assistant]'}}{% else %}{{   message['content'] +' [/assistant]'  }}{% endif %}{% endfor %}"
times=[]
messages_record = []
clinet_index = 0
scheduler = GlobalScheduler(2) #GlobalSchedulerWithTime(num_nodes=2)
gtokenizer = AutoTokenizer.from_pretrained("/hy-tmp/")
modelConfig = ModelConfig("/hy-tmp/")
app = FastAPI()

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
@app.post("/v1/chat/completions")
async def forward_request(request: ChatCompletionRequest,raw_request: Request):
    request_dict = await raw_request.json()
    prompt = request.messages[0]["content"] #request_dict["messages"][0]["content"]
    conversation: List[ConversationMessage] = []
    for msg in request.messages:
        chat_parsed_result = parse_chat_message_content(
            msg)
        conversation.extend(chat_parsed_result.messages)
    prompt = gtokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=None,
                tools=None,
                documents=None,
                chat_template=chat_template,
                **(request.chat_template_kwargs or {}),
    )
    prompt_inputs = modelConfig.tokenize_prompt_input(
                request,
                gtokenizer,
                prompt
    )
    sampling_param = {"temperature":request.temperature,"max_tokens":request.max_tokens}
    request_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))

    clinet_index,new_prefix_len = scheduler.runtime_selector(request_id,prompt_inputs['prompt_token_ids'],request.model)
    print(clinet_index,request.model)
    # 下面这句话会把treecache的key值也改了,所以需要deepcopy
    # prompt_inputs['prompt_token_ids'][0] = prompt_inputs['prompt_token_ids'][0][1]
    # for sglang no need for request_dict["prompt"] if request_dict["input_ids"] is provided
    # request_dict["prompt"] = prompt
    request_dict["prompt"] = None
    prompt_len = len(prompt_inputs['prompt_token_ids'])
    # for sglang
    request_dict["input_ids"] = copy.deepcopy(prompt_inputs['prompt_token_ids'])
    request_dict["input_ids"][0] = request_dict["input_ids"][0][1]
    # for vllm
    # request_dict["prompt_token_ids"] = prompt_inputs['prompt_token_ids']
    now_per_gpu_load_len = scheduler.per_gpu_load_len[clinet_index]

    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    api_url = "http://localhost:30000/v1/chat/completions"
    
    async def get_requests():
        chunks = []
        generated_text = ""
        ttft = 0.0
        output = RequestFuncOutput()
        output.prompt_len = prompt_len - new_prefix_len
        send_out_time = time.perf_counter()
        most_recent_timestamp = send_out_time
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=request_dict,
                                            headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        #chunk_bytes = chunk_bytes.strip() 一定不能加这句，需要带上两个/n/n
                        if not chunk_bytes.strip():
                            continue
                        text = chunk_bytes.decode("utf-8")
                        chunks.append(text)
                        yield text
                        chunk_bytes = chunk_bytes.strip()
                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                                "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - send_out_time
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - send_out_time
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
        

        scheduler.finish_request(request_id,clinet_index,output,now_per_gpu_load_len) # 到时候可以改成异步
    
    return StreamingResponse(get_requests(),media_type="text/event-stream")


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""
    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)

def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val
if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="JointServe RESTful API server.")
    parser.add_argument("--host",
                        type=nullable_str,
                        default=None,
                        help="host name")
    parser.add_argument("--port", type=int, default=9999, help="port number")
    args = parser.parse_args()
    uvicorn.run(app,
                    host=args.host,
                    port=args.port,
                    )


# @app.post("/v1/chat/completions")
# async def forward_request(request: ChatCompletionRequest,
#                                  raw_request: Request):
#     # request_dict = await request.json()
#     model = request.model
#     text = request.messages[-1]["content"]
#     print(text)
#     # input_ids = tokenizer.encode(text)
#     # sampling_param = {"temperature":0,"max_tokens":100}
#     # request_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
#     # clinet_index,metrics_dict = scheduler.runtime_selector(text=text,
#     #     request_id = request_id,input_ids=input_ids,sampling_params=sampling_param)
#     # send_out_time = time.time()
#     # print(metrics_dict)
#     output =  clients[clinet_index].chat.completions.create(
#             messages=request.messages,
#             model=model,
#             temperature=0,
#             max_tokens=request.max_tokens,
#             stream=True, 
#             extra_body={"chat_template":chat_template,}
#         )
#     async def get_requests():
#         chunks = ''
#         for chunk in output:
#             # out_text = chunk.choices[0].delta.content
#             # if out_text is not None:
#             #     chunks += out_text
#             data = chunk.model_dump_json(exclude_unset=True)
#             yield f"data: {data}\n\n"
#         yield "data: [DONE]\n\n"
        
#         # generated_text = chunks
#         # latency = time.time()-send_out_time
#         # req_out = RequestFuncOutput(
#         #         rid=request_id,
#         #         prompt_text=text,
#         #         prompt_len=len(input_ids), 
#         #         # generated_text=generated_text,
#         #         send_out_time=send_out_time,
#         #         route_dest=clinet_index,
#         #         scheduling_overhead=metrics_dict["overhead"],
#         #         runtime_selected=clinet_index,
#         #         # max_new_tokens=len(generated_text),
#         #         ttft=latency/len(generated_text),
#         #         request_latency=latency,
#         #         output_len=len(generated_text),
#         #         tpot=latency/len(generated_text) *(len(generated_text)-1),
#         #         success=True,
#         #         global_time=time.time()
#         #     )
#         # scheduler.finish_request(text, request_id, input_ids, req_out) # 到时候可以改成异步
    
#     return StreamingResponse(get_requests(),media_type="text/event-stream")
