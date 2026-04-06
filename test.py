from openai import OpenAI
aliyun_api_key = 'sk-d18ec22172af4ad2aa8fa11e82e480c0'
client = OpenAI(
    api_key=aliyun_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-max",
    messages=[
        {'role': 'user', 'content': "你是谁？"}
    ]
)

# 打印完整回答内容
print(response.choices[0].message.content)