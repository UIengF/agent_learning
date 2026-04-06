# agent_learning

这是一个用于学习和实验 Python Agent 基础模式的示例仓库，当前包含几类典型实践：

- 直接调用大模型接口
- 基于 ReAct 模式进行工具调用
- 使用 LangGraph 构建带工具调用的 Agent
- 使用 SQLite checkpoint 为 LangGraph Agent 增加记忆能力

## 文件说明

- `test.py`
  最小化的大模型调用示例，演示如何通过兼容 OpenAI 接口访问模型。

- `react.py`
  一个简化版 ReAct Agent 示例。模型通过 `Thought / Action / Observation` 循环决定是否调用工具。

- `graph_begin.py`
  使用 LangGraph 构建基础 Agent 图，包含：
  - 状态定义
  - LLM 节点
  - 工具调用节点
  - 条件边控制

- `graph_mem.py`
  在 `graph_begin.py` 基础上增加 SQLite checkpoint，用于保存对话线程状态，演示带记忆的 LangGraph Agent。

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装常见依赖：

```bash
pip install openai langchain-openai langchain-core langgraph pydantic requests
```

如果运行 `graph_mem.py` 相关代码，还需要确保本地环境支持 `langgraph.checkpoint.sqlite`。

## 运行方式

直接运行对应脚本即可：

```bash
python test.py
python react.py
python graph_begin.py
python graph_mem.py
```

## 配置说明

当前示例代码中包含模型与搜索服务相关配置。更合理的做法是：

- 不要在代码中硬编码 API Key
- 改为使用环境变量读取密钥
- 提交前检查是否包含敏感信息

例如可以改成：

```bash
set OPENAI_API_KEY=your_key
set BOCHA_API_KEY=your_key
```

然后在 Python 中通过环境变量读取。

## 当前状态

这个仓库目前以学习和验证思路为主，代码更接近实验脚本，而不是完整工程项目。后续可以继续补充：

- `requirements.txt`
- 环境变量配置示例
- 更清晰的目录结构
- 单元测试
- 更安全的密钥管理方式
