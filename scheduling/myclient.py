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
from preble.preble_scheduler import GlobalSchedulerWithTime
from preble.my_scheduler import GlobalScheduler
from preble.utils import RequestFuncOutput,ChatCompletionRequest,parse_chat_message_content,ConversationMessage,ModelConfig,remove_prefix
from transformers import AutoTokenizer
from typing_extensions import List
import random
import string
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
chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content']+' [/user][assistant]'}}{% else %}{{   message['content'] +' [/assistant]'  }}{% endif %}{% endfor %}"
times=[]
messages_record = []
clinet_index = 0
scheduler = GlobalScheduler(2) #GlobalSchedulerWithTime(num_nodes=2)
gtokenizer = AutoTokenizer.from_pretrained("/hy-tmp/")
modelConfig = ModelConfig("/hy-tmp/")
app = FastAPI()
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

    clinet_index = scheduler.runtime_selector(request_id,prompt_inputs['prompt_token_ids'],request.model)
    print(clinet_index,request.model)
    request_dict["prompt"] = prompt
    request_dict["prompt_token_ids"] = prompt_inputs['prompt_token_ids']


    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    api_url = "http://localhost:8000/v1/chat/completions"
    
    async def get_requests():
        chunks = []
        send_out_time = time.perf_counter()
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
        # generated_text = ""
        # ttft = 0.0
        # for chunk in chunks:
        #     #chunk = chunk.strip()
        #     chunk = remove_prefix(chunk,"data:")
        #     chunk = chunk.strip().replace("\n\n", "")
        #     if chunk == "[DONE]":
        #         break
        #     try:
        #         data = json.loads(chunk)
        #         if data["choices"][0]["delta"].get("content",None):
        #             if ttft == 0.0:
        #                 ttft = time.perf_counter() - send_out_time
        #             generated_text += data["choices"][0]["delta"]["content"]
        #     except Exception as e:
        #         print(e)
            
        # latency = time.perf_counter()-send_out_time
        # req_out = RequestFuncOutput(
        #         rid=request_id,
        #         prompt_text=text,
        #         prompt_len=len(prompt_inputs['prompt_token_ids']), 
        #         # generated_text=generated_text,
        #         send_out_time=send_out_time,
        #         route_dest=clinet_index,
        #         scheduling_overhead=metrics_dict["overhead"],
        #         runtime_selected=clinet_index,
        #         # max_new_tokens=len(generated_text),
        #         ttft=ttft,
        #         request_latency=latency,
        #         output_len=len(generated_text),
        #         tpot=(latency-ttft)/(len(generated_text)-1),
        #         success=True,
        #         global_time=time.time()
        #     )
        scheduler.finish_request(request_id,clinet_index) # 到时候可以改成异步
    
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