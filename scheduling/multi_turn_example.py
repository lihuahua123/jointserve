import time
from openai import OpenAI
user_messages = [
    "Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality?", # noqa: E501
    "Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport"
    # "Who directed each of these films?",
    # "Which director has the most experience?",
    # "What other films has this director directed?",
    # "Do these films have anything in common?",
    # "Which of those films is the oldest?",
    # "How old was the director when this was released?",
]

client = OpenAI(api_key="api_key", base_url="http://localhost:8000/v1")
models = client.models.list() # 第一个是llm，后面是lora模型
model = "lora1" #models.data[0].id

times=[]
messages = []
for i,user_message in enumerate(user_messages):
    messages.append(dict(role="user", content=user_message))
    start = time.perf_counter()
    chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[user] '+message['content']+' [/user][assistant]'}}{% else %}{{   message['content'] +' [/assistant]'  }}{% endif %}{% endfor %}"
            
    output = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0,
        max_tokens=200,
        extra_body={"chat_template":chat_template,}
    )
    stop = time.perf_counter()
    print(f"{stop - start = }")
    times.append(stop - start)
    print(output.usage)
    assistant_message = output.choices[0].message
    print(assistant_message.content)
    messages.append(dict(role=assistant_message.role, content=assistant_message.content))

print(sum(times))