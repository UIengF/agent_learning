# 基于 LangGraph 的多源证据增强 RAG Agent

## 一、简历项目概括

### 项目名称

基于 LangGraph 的多源证据增强 RAG Agent

### 一句话描述

设计并实现一个面向知识库问答和研究检索的 RAG Agent，支持本地知识库混合检索、联网搜索、网页正文抽取、Google Scholar 论文检索、上下文压缩、SQLite checkpoint 断点恢复和用户偏好记忆。

### 技术栈

Python、LangGraph、LangChain、OpenAI-compatible API / DashScope、SQLite、BM25、TF-IDF、Dense Vector、DuckDuckGo HTML Search、SerpAPI Google Scholar、unittest

### 简历版项目描述

独立开发基于 LangGraph 的多源证据增强 RAG Agent，用于解决本地知识库证据不足、长对话上下文膨胀和研究类问题资料来源分散的问题。系统以 Python 实现 CLI 工具链，支持对 Markdown / DOCX 知识库构建 SQLite 检索索引，并提供本地 RAG、联网搜索、网页正文抓取和 Google Scholar 检索等工具供 Agent 按需调用。

在检索侧，实现了基于 BM25 稀疏召回与 Dense Vector 相似度融合的 hybrid retrieval，并结合 source、title、section 等元数据进行重排，提高对具体文档、标题和章节查询的命中稳定性。索引构建阶段会对文档分块、提取 Markdown 章节、补充 chunk header 元数据，并将 token 统计、IDF、BM25 IDF、向量和 chunk 文本持久化到 SQLite，便于复用和离线检索。

在 Agent 编排侧，基于 LangGraph 构建 LLM 节点和工具执行节点，支持 local_rag_retrieve、web_search、web_fetch、scholar_search 等工具调用，并通过工具轮次限制、证据缓存、反思提示、上下文预算、长对话压缩和用户记忆机制控制幻觉与上下文成本。系统还提供 checkpoint / resume 能力，将会话状态保存到 SQLite，支持中断后继续执行。

### 可直接放到简历中的条目

- 独立实现基于 LangGraph 的 RAG Agent，集成本地知识库检索、联网搜索、网页正文抽取与 Google Scholar 检索，支持在本地证据不足时自动切换到外部证据源。
- 设计 SQLite 持久化索引，支持 Markdown / DOCX 文档解析、章节级分块、chunk 元数据增强、BM25 / TF-IDF 统计和向量存储，避免每次问答重复构建索引。
- 实现 hybrid retrieval，将 BM25 稀疏召回与 Dense Vector 相似度按权重融合，并基于 source / title / section 元数据进行重排，提升针对文档名、标题和章节查询的召回质量。
- 基于 LangGraph 实现 LLM 节点与工具节点的循环编排，加入工具调用轮次上限、证据缓存、官方链接优先抓取、搜索结果去重和 fetch-before-answer 策略，降低无依据回答风险。
- 构建上下文管理模块，支持会话摘要、任务状态、用户偏好记忆、实时消息压缩、token / char 预算控制和上下文指标记录，缓解长对话和多工具调用带来的上下文膨胀。
- 完成 CLI 工具链与单元测试覆盖，提供索引构建、索引检查、直接检索、Agent 问答、网页搜索、网页抓取、Scholar 检索和 Markdown 导出等命令。

### 简历精简版

基于 Python + LangGraph 独立实现多源证据增强 RAG Agent，支持本地知识库 hybrid retrieval、Web Search / Fetch、Google Scholar 检索、SQLite checkpoint 断点恢复和上下文压缩。系统将 Markdown / DOCX 文档构建为 SQLite 持久化索引，融合 BM25、Dense Vector 和元数据重排提升检索质量，并通过工具调用限制、证据缓存、反思提示和 fetch-before-answer 策略降低大模型幻觉风险。

### 项目亮点

- 多源证据融合：本地 RAG、网页搜索、网页正文抓取、论文检索统一封装为 Agent 工具。
- 检索链路完整：从文档解析、分块、元数据增强、索引持久化到 hybrid retrieval 和评估指标均有实现。
- Agent 工程化：具备 LangGraph 状态流转、工具调用、日志、checkpoint、resume、上下文压缩和用户记忆。
- 可测试性较强：项目包含检索、上下文构建、Web 工具、Scholar 工具、CLI 和 Agent 行为相关测试模块。
- 面试表达要点：这是一个“LangGraph-based RAG Agent”，不是传统意义上基于知识图谱数据库的 GraphRAG；项目名中的 Graph 主要来自 LangGraph 的状态图编排。

## 二、面试官可能会问的问题与回答要点

### 1. 这个项目解决的核心问题是什么？

回答要点：

本项目解决的是 RAG 问答中“本地知识不完整、外部信息需要实时获取、长对话上下文容易膨胀、工具调用容易失控”的问题。系统通过本地知识库检索提供稳定证据，通过 Web / Scholar 补充外部证据，并用 LangGraph 管理 LLM 与工具调用流程。

### 2. 为什么使用 LangGraph，而不是简单的 ReAct 循环？

回答要点：

LangGraph 更适合把 Agent 流程显式建模成状态图。项目中将 LLM 调用和工具执行拆成节点，并通过条件边判断是否继续调用工具或结束回答。这样比手写 while loop 更容易接入 checkpoint、resume、状态检查、流式执行和中断恢复。

### 3. 本地 RAG 检索流程是怎样的？

回答要点：

先解析 Markdown / DOCX 文档，再按章节和长度进行 chunk 切分；每个 chunk 会补充 source、title、section 元数据。索引构建时会计算 token counts、TF-IDF、BM25 IDF 和 dense vector，并持久化到 SQLite。查询时先计算 BM25 分数和 dense 相似度，再做加权融合并重排。

### 4. 为什么使用 hybrid retrieval？

回答要点：

BM25 对关键词、专有名词、文档标题匹配更稳定，Dense Vector 对语义相近但字面不同的问题更友好。项目将二者融合，可以兼顾精确匹配和语义召回，减少单一检索策略的失效场景。

### 5. hybrid score 是怎么计算的？

回答要点：

系统分别计算 sparse score 和 dense score，并对分数做归一化，然后按 keyword_weight 融合：

```text
score = keyword_weight * sparse_score + (1 - keyword_weight) * dense_score
```

默认 keyword_weight 为 0.5，也可以通过配置调整。

### 6. 如果没有外部 embedding API，系统还能运行吗？

回答要点：

可以。项目中如果没有配置 DashScope embedding，会退化为基于稳定 hash 的 dense vector 方案。它不能替代真实语义 embedding，但可以保证离线可运行、单元测试可执行，也能作为基础 dense signal 参与融合检索。

### 7. 为什么把索引存在 SQLite，而不是只放内存？

回答要点：

SQLite 适合本地 demo 和轻量级工程化场景。它能持久化 chunk、元数据、token counts、IDF 表和向量，避免每次启动重复解析文档和构建索引，同时便于 inspect、复用和 checkpoint 一起管理。

### 8. chunk 切分策略有哪些设计？

回答要点：

项目支持 Markdown 章节识别，会尽量保留 section title，并为 chunk 添加 source / title / section header。这样检索时不仅匹配正文，也能匹配文档名、标题和章节名。对于短章节开头，还做了首段扩展，避免 lead chunk 信息太少。

### 9. 元数据重排为什么重要？

回答要点：

很多用户问题会包含文档名、产品名、标题或章节关键词。如果只看正文相似度，可能把正文相似但来源不对的 chunk 排到前面。项目通过 source、title、section 的命中 bonus 修正排序，提高指定文档或指定章节查询的稳定性。

### 10. Agent 如何决定使用本地检索、Web Search 还是 Scholar？

回答要点：

系统 prompt 明确要求优先使用 local_rag_retrieve；当本地证据不足、问题涉及近期公开信息时使用 web_search；当问题涉及论文、综述、引用、相关工作时使用 scholar_search。工具结果返回后还有 reflection prompt，让模型判断当前证据是否足够，以及下一步最小必要动作。

### 11. 为什么不能直接依赖搜索摘要回答？

回答要点：

搜索摘要信息不完整且可能被截断或误导。项目 prompt 明确要求不能仅凭 snippet 回答详细事实问题；如果 snippet 指向可能有证据的页面，需要进一步调用 web_fetch 获取正文后再回答。

### 12. Web Search 做了哪些工程处理？

回答要点：

项目实现了 DuckDuckGo HTML 搜索解析、搜索查询扩展、URL 归一化、结果去重、官方域名识别和文档路径加权。对于 OpenAI、Google 等技术问题，还会优先识别官方文档域名，减少引用非权威来源的概率。

### 13. Web Fetch 如何抽取网页正文？

回答要点：

Web Fetch 会设置 User-Agent 和基础请求头，仅接受 text/html 内容；解析时过滤 script、style、nav、footer 等噪声标签，并优先抽取 article 或 main 标签内容。还支持最大字节数、最大字符数、重试和截断标记。

### 14. Scholar 检索模块有什么作用？

回答要点：

Scholar 模块用于处理论文、综述、引用、相关工作类问题。它可以通过 LLM planner 将主题扩展成 2 到 4 个 Google Scholar 查询，再调用 SerpAPI 获取论文元数据，并基于标题、URL、查询覆盖数、排名、引用量和年份做合并排序。

### 15. 如何降低 Agent 的工具调用失控问题？

回答要点：

项目设置了 max_rounds 工具轮次上限。如果达到上限，Agent 会停止继续调用工具，转而基于已有证据回答并说明不确定性。另外还有证据缓存，避免同一个工具参数重复调用。

### 16. 上下文压缩是怎么做的？

回答要点：

系统把上下文分为 system prompt、用户记忆、会话摘要、实时消息、证据缓存、任务状态和 reflection prompt 等层。构建上下文时会根据 char / token 预算进行截断或丢弃，并对较早的 tool message 做压缩，只保留 query、URL、标题、少量正文等关键信息。

### 17. 用户记忆模块记录什么？

回答要点：

用户记忆模块会从历史 human message 中提取偏好语言、回答风格、常用 index-dir、稳定约束和 recurring topics，并保存到 JSON 文件。后续问答时会把这些偏好作为上下文注入，提升多轮使用体验。

### 18. checkpoint / resume 是怎么实现的？

回答要点：

项目使用 LangGraph 的 SQLite checkpointer，将指定 session_id 的图状态保存到 SQLite。运行时可以通过 session_id 获取 graph state，如果存在 pending node，可以用 resume 参数从中断处继续执行。

### 19. 你如何评估检索效果？

回答要点：

项目实现了 hit_rate@k、MRR、NDCG@k 等检索评估函数。给定 query 和 relevant_source_paths 后，可以评估 top-k 检索结果是否命中目标文档，以及相关结果排序是否靠前。

### 20. 项目里哪些地方体现了可测试性？

回答要点：

项目包含多个 unittest 测试模块，覆盖 retrieval、chunking、context_builder、context_metrics、web_search、web_fetch、web_tools、scholar_search、scholar_export、CLI 和 Agent 工具限制等。很多模块对外部依赖做了 fallback 或 mock-friendly 设计，便于在没有真实网络和 API key 的情况下测试核心逻辑。

### 21. 这个项目和普通 RAG Demo 最大区别是什么？

回答要点：

普通 RAG Demo 往往只做“向量检索 + LLM 回答”。这个项目更接近一个工程化研究助手：有本地索引持久化、hybrid retrieval、外部 Web / Scholar 工具、LangGraph 状态编排、上下文预算、证据缓存、用户记忆、日志、checkpoint 和 CLI 工具链。

### 22. 项目当前有什么不足？

回答要点：

可以坦诚说明：当前项目更偏本地实验和工程原型，缺少 Web UI、部署配置、依赖文件和真实线上监控；fallback dense vector 不等价于真实 embedding；还没有接入专业向量数据库；README 中部分路径可能需要更新。后续可补充 requirements / pyproject、Docker、API 服务、向量数据库、权限控制和更系统的评测集。

### 23. 如果要上线这个项目，你会怎么改？

回答要点：

首先补齐依赖管理、配置管理和密钥管理；其次把 CLI 封装成 API 服务，增加异步任务、索引构建队列和多用户隔离；再接入向量数据库或 SQLite-vec；最后增加观测能力，包括 tool call trace、检索命中率、回答引用、失败重试、成本统计和安全过滤。

### 24. 如何进一步降低幻觉？

回答要点：

可以增加强制引用机制，让最终回答必须引用具体 chunk / URL / paper metadata；对低分检索结果设置阈值；对冲突证据做显式比较；要求模型在证据不足时拒答；对 Web 来源做白名单或权威域名优先；引入 answer verification 或 second-pass fact checking。

### 25. 如果面试官问“为什么叫 Graph RAG”，怎么回答？

回答要点：

这个项目名里的 Graph 主要来自 LangGraph 的状态图编排，而不是知识图谱数据库或微软 GraphRAG 那类社区摘要算法。更准确的表述是“基于 LangGraph 的多工具 RAG Agent”。简历中建议这样写，避免面试官误解为实现了 Knowledge Graph RAG。

## 三、面试时建议强调的主线

1. 先讲业务目标：让 Agent 在本地知识、实时网页和论文资料之间选择可靠证据。
2. 再讲检索链路：文档解析、chunk、SQLite 索引、BM25、dense、hybrid、metadata rerank。
3. 再讲 Agent 编排：LangGraph 节点、工具调用、反思、工具轮次限制、checkpoint。
4. 最后讲工程化：CLI、日志、上下文预算、用户记忆、测试、可扩展方向。

## 四、30 秒口述版本

我做了一个基于 LangGraph 的多源 RAG Agent。它可以先从本地 Markdown / DOCX 知识库中做 hybrid retrieval，如果本地证据不足，再调用 Web Search、网页正文抓取或 Google Scholar 检索补充证据。检索部分我实现了 SQLite 持久化索引、BM25 和 Dense Vector 融合、元数据重排；Agent 部分实现了工具调用编排、证据缓存、反思提示、上下文压缩、用户记忆和 checkpoint 断点恢复。这个项目主要体现我对 RAG 检索链路、Agent 工程化和长上下文控制的理解。
