import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam

from ragu.utils.ragu_utils import FLOATS, get_disk_cache
from ragu.common.logger import logger


# LLM Interfaces

T = TypeVar('T', BaseModel, str)

class LLM:
    """Abstract LLM able to respond with texts, structured schemas
    and/or embeddings.
    
    This base class is made to enable backend-agnostic response caching.
    It defines two public methods that act as caching wrappers for
    their private counterparts:

    - `chat_completion` (wrapper) and `_chat_completion` (abstract)
    - `embed_text` (wrapper) and `_embed_text` (abstract)
    
    ### How caching works
    
    This class uses abstract dict (str -> Any) as cache, typically this may
    be a dict() for in-memory caching, or diskcache.Index for disk
    caching.

    Caching key is calculated by combining `chat_completion` or
    `embed_text` arguments and `cache_prefix`.

    ### Subclassing rules

    1. Override `_chat_completion` and/or `_embed_text` in subclass,
       while `chat_completion` and `embed_text` in base class serve as
       a caching wrapper.
    2. Call `super().__init__(cache, prefix)` in constructor if you
       need to enable caching.
    3. Optionally may add more keyword arguments to `chat_completion`,
       such as `temperature`, `tools` etc, they will also be added in
       the caching key calculation.
    4. If you have object-level parameters, such as `temperature`,
       consider moving them into `chat_completion` arguments, so that
       temperature value is cached cofrrectly, or add them as `cache_prefix`.
       The `cache_prefix` may also be used if the same cache is reused by
       multiple `StructuredOutputLLM` subclasses that return different
       results for the same input parameters in `chat_completion`.
    """

    cache: MutableMapping[str, Any] | None = None

    def __init__(
        self,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = '',
    ):
        self.cache_prefix = cache_prefix
        match cache:
            case None:
                self.cache = None
            case str() | Path():
                self.cache = get_disk_cache(cache)
            case _:
                self.cache = cache

    async def chat_completion(  # with caching
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        is_str = issubclass(output_schema, str)
        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'chat_completion',
            'conversation': conversation,
            'output_schema': 'str' if is_str else output_schema.model_json_schema(),
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if self.cache is not None and (value := self.cache.get(key, None)):
            logger.debug(f'Cache hit for {model_name}! Returning from cache.')
            cached: str | dict[str, Any]
            _args, cached = value
            result = cached if is_str else output_schema.model_validate(cached)
            return cast(T, result)
        
        if self.cache is not None:
            logger.debug(f'Cache miss for {model_name}! Doing a request.')
        
        response = await self._chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            **kwargs,
        )

        cached = response if is_str else response.model_dump() # type: ignore

        if self.cache is not None:
            self.cache[key] = args, cached

        return response

    async def _chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        # should be overridden only if a subcclass supports chat completions
        # kwargs are here to add custom arguments that will also be cached
        raise NotImplementedError()
    
    async def embed_text(  # with caching
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'embed_text',
            'text': text,
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if self.cache is not None and (value := self.cache.get(key, None)):
            logger.debug(f'Cache hit for {model_name}! Returning from cache.')
            cached: list[float] | FLOATS
            _args, cached = value
            return cached
        
        if self.cache is not None:
            logger.debug(f'Cache miss for {model_name}! Doing a request.')
        
        response = await self._embed_text(
            model_name=model_name,
            text=text,
            **kwargs,
        )

        if self.cache is not None:
            self.cache[key] = args, response

        return response

    async def _embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        # should be overridden only if a subcclass supports embeddings
        # kwargs are here to add custom arguments that will also be cached
        raise NotImplementedError()


# LLM Implementations

