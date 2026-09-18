"""Live web-search fallback (Architecture.md §3.6).

Used only when the corpus cannot ground an answer (empty retrieval, grounding
failure, or a low-confidence route). Results become `RetrievedChunk`s with
`source="web"` and a live URL, then re-enter generation → grounding →
confidence, so a web answer is held to the exact same citation gate as a
knowledge-base one (Rule 5: fallback is honest, sources labelled and linked).
"""
