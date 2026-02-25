# Interface level

`LLM`, `Embedder` - interface level. Argument type for
various components. Example:

```
@dataclass
class MyCommunitySummarizer:
    fast_llm: LLM,
    heavy_llm: LLM,
    query_embedder: Embedder
    key_embedder: Embedder
```

# Network level

`ResponseCachingMixin`, `CachedAsyncOpenAI` work on network level.
They handle caching, retrying, rate limiting, and woring with
OpenAI API.

# Logic level

`LLMOpenAI`, `EmbedderOpenAI` implement `LLM` and `Embedder`
interfaces, using `CachedAsyncOpenAI` backend. Example:

```
client = CachedAsyncOpenAI()
summarizer = MyCommunitySummarizer(
    fast_llm=LLMOpenAI(client, 'qweno-tiny'),
    heavy_llm=LLMOpenAI(client, 'claude-mighty'),
    query_embedder=EmbedderOpenAI(client, 'bert', dim=768),
    key_embedder=EmbedderOpenAI(client, 'bert', dim=768),
)
```

Thus, rate limiting is shared between all models, since they use
the same client, caching and retrying is handled correctly.