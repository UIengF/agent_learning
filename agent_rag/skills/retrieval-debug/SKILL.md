---
name: retrieval-debug
description: Diagnose local RAG retrieval, chunking, indexing, and metadata reranking issues.
---

Use this skill when the user asks why local retrieval missed an expected document,
why a result ranked poorly, or how to improve knowledge-base matching.

Recommended workflow:

1. Inspect the index before changing retrieval behavior.
2. Compare the user question with source path, title, section, and chunk text.
3. Check whether the query needs domain-specific rewrite or title/section terms.
4. Prefer `local_rag_retrieve` with focused queries before using web tools.
5. If local evidence is weak, explain whether the likely cause is missing corpus
   content, chunk boundaries, keyword mismatch, dense-vector mismatch, or rerank
   weighting.

Evidence rules:

- Do not claim a document exists unless retrieval or index inspection supports it.
- Treat low-score local results as weak evidence.
- When relevant, suggest rebuilding the index after changing documents or chunking
  parameters.
