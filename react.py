from openai import OpenAI
import re,json

# 这里直接写死了 API Key，便于示例演示；实际使用更建议放到环境变量里。
aliyun_api_key = 'sk-d18ec22172af4ad2aa8fa11e82e480c0'
client = OpenAI(
    api_key=aliyun_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Agent 负责维护对话历史，并在每一轮把历史消息发给模型。
class Agent():
    def __init__(self,system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role":"system","content":system})

    # 调用 Agent("问题") 时，自动把用户输入加到历史里，再请求模型。
    def __call__(self, message):
        self.messages.append({"role":"user","content":message})
        result =self.excute()
        self.messages.append({"role":"assistant","content":result})
        return result
    
    # 真正调用大模型接口的地方。
    def excute(self):
        response = client.chat.completions.create(
            model="qwen-max",
            messages =self.messages
        )
        return response.choices[0].message.content

# 一个最简单的工具：让模型把数学表达式交给 Python 计算。
def calculate(what):
    return eval(what)

# 另一个工具：根据犬种返回平均体重，用来演示“模型调用外部工具”。
def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
# abot=Agent("你是我的人工智能助手，协助我完成任务")
# print(abot("你是谁？"))

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

# 用原始字符串写正则，避免 `\w` 触发 Python 的转义警告。
action_re = re.compile(r'^Action: (\w+): (.*)$')

# ReAct 主循环：
# 1. 把问题发给模型
# 2. 解析模型是否要求调用工具
# 3. 执行工具并得到 Observation
# 4. 把 Observation 再喂回模型，继续下一轮
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # 如果模型输出了 Action，就执行对应工具。
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            # 如果没有 Action，说明模型已经给出了最终答案。
            return result
        
# 示例问题：模型需要先查两个犬种的平均体重，再尝试给出总和。
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
