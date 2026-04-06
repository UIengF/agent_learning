import os

from openai import OpenAI


dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key:
    raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen-max",
    messages=[
        {"role": "user", "content": "你是谁？"},
    ],
)

print(response.choices[0].message.content)
