---
name: grounded-answering
description: Synthesize final answers from local, web, and scholar evidence without unsupported claims.
---

Use this skill before final answer synthesis when the task involves multiple
tool results, mixed evidence quality, or long context.

Recommended workflow:

1. Start from the user's exact question and success criteria.
2. Group evidence by source type: local chunks, fetched web pages, scholar
   metadata, cached evidence, and prior research plan.
3. Prefer high-confidence local chunks and fetched primary sources over snippets.
4. State missing evidence explicitly instead of filling gaps with assumptions.
5. Keep the answer organized around the user's requested comparison, decision,
   explanation, or implementation steps.

Grounding rules:

- Do not cite a source that was not retrieved or fetched.
- Do not mention paper titles unless they appear in allowed scholar results or
  fetched content.
- When evidence conflicts, describe the conflict and avoid overclaiming.
- If the tool limit was reached, answer from available evidence and call out the
  remaining uncertainty.
