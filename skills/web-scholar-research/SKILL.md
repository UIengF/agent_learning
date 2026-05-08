---
name: web-scholar-research
description: Plan reliable web and Google Scholar research with fetch-before-answer discipline.
---

Use this skill for recent public information, papers, literature reviews,
citations, related work, or questions where the local knowledge base is
insufficient.

Recommended workflow:

1. Use `scholar_search` for papers, surveys, citations, benchmarks, and related
   work.
2. Use `web_search` for recent public information, product documentation,
   release notes, and official-source discovery.
3. Do not answer detailed factual claims from search snippets alone. Fetch the
   page with `web_fetch` when the snippet points to evidence that matters.
4. Prefer official documentation, papers, project repositories, or primary
   sources over summaries and reposts.
5. Merge duplicate search hits by URL and title before synthesizing.

Evidence rules:

- Clearly separate local evidence, web evidence, and scholar metadata.
- Mention uncertainty when search results are incomplete or conflicting.
- For academic answers, cite paper titles only when they appear in scholar
  results or fetched evidence.
