---
name: agent-evaluation
description: Evaluate retrieval, tool trajectory, sources, and final answers with layered criteria.
---

Use this skill when the user asks to assess retrieval quality, compare evaluation
runs, interpret metrics, or debug failing evaluation cases.

Recommended workflow:

1. Identify the dataset, case id, expected sources, expected entities, and
   assertion layers.
2. Evaluate retrieval before judging the answer.
3. Check tool trajectory: whether the agent used local retrieval, web search,
   web fetch, scholar search, and research planning in a reasonable order.
4. Check source quality: whether cited local chunks, URLs, or scholar results
   actually support the answer.
5. Summarize failures as actionable fixes, such as query rewrite, metadata
   rerank, chunking, source extraction, or answer synthesis.

Metric interpretation:

- Hit@1 measures whether the top result is relevant.
- Recall@5 measures whether relevant evidence appears in the top five.
- NDCG@10 rewards relevant evidence appearing earlier in the ranking.
- A passing final answer still needs grounded sources and a reasonable tool path.
