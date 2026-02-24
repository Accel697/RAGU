import asyncio
import logging
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, TypeVar, cast
from pydantic import BaseModel
from typing_extensions import override

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, before_sleep_log
from aiolimiter import AsyncLimiter

from ragu.llm.llm import LLM
from ragu.utils.ragu_utils import FLOATS, LoguruAdapter, attach_async_contexts
from ragu.common.logger import logger


T = TypeVar('T', BaseModel, str)

@dataclass
class CachedOpenAI(LLM):
    """OpenAI client able to respond with structured outputs and
    embeddings, with response caching, rate limiting and request retrying.

    If `client` is provided, the arguments `base_url` and `api_key`
    are not used. Otherwise, a new `AsyncOpenAI` client is constructed.

    ### Schema handling

    If `output_schema == str`, runs `client.chat.completions.create`
    and returns the `response.choices[0].message.content`.

    If `output_schema != str`, then an additional parameter `as_tool`
    offers two different ways to handle the `output_schema`. The
    correctness and quality of the responses is model-dependent and
    provider-dependent:
    
    - If `as_tool=True`: calls `client.chat.completions.create` and
      passed `tool_definition` that contain the output format schema.
    - If `as_tool=False`: calls `client.beta.chat.completions.parse` and
      passed the `response_format` argument.

    ### Rate limits and retrying
    
    Rates can be controlled by:
    - `rate_min_delay`: min delay in seconds between requests
    - `rate_max_per_minute`: max requests per minute
    - `rate_max_simultaneous`: max simultaneous requests

    Allows retrying: for example, if `retry_times=(4, 8, 16)`, will
    retry in 4, then 8, then 16 seconds on exception, and finally
    raise it. In rate limiting, each retrying attempt is considered
    a new request.

    So, these mechanisms are independent: rate limiting delays
    requests, and retrying handles exceptions.

    ### Response caching

    Typically, pass `cache="my_cache_dir/"` to enable caching. For
    details see the base class.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        rate_min_delay: float | None = None,
        rate_max_per_minute: int | None = None,
        rate_max_simultaneous: int | None = None,
        retry_times_sec: Sequence[float] | None = None,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = 'openai',
    ):
        super().__init__(cache=cache, cache_prefix=cache_prefix)

        self.client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Should add retrying after attaching limiters, so that
        # every retry increments counter in the limiters.

        # Thus, handlers/wrappers will be called in this order:
        # 1. Caching
        # 2. Retrying
        # 3. Rate limiting

        # add rate limiter contexts
        contexts: list[AbstractAsyncContextManager[Any]] = []
        if rate_max_per_minute:
            contexts.append(AsyncLimiter(rate_max_per_minute, time_period=60))
        if rate_max_simultaneous:
            contexts.append(asyncio.Semaphore(rate_max_simultaneous))
        if rate_min_delay:
            contexts.append(AsyncLimiter(1, time_period=rate_min_delay))
        if contexts:
            self._chat_completion = attach_async_contexts(
                self._chat_completion, *contexts
            )
            self._embed_text = attach_async_contexts(
                self._embed_text, *contexts
            )

        # add retrying decorators
        if retry_times_sec:
            retrying_decorator = retry(
                stop=stop_after_attempt(len(retry_times_sec) + 1),
                wait=wait_chain(*[wait_fixed(t) for t in retry_times_sec]),
                before_sleep=before_sleep_log(
                    LoguruAdapter('logger'), logging.DEBUG
                ),
                reraise=True
            )
            self._chat_completion = retrying_decorator(self._chat_completion)
            self._embed_text = retrying_decorator(self._embed_text)

    @override
    async def _chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        as_tool: bool = False,
        **kwargs: Any,
    ) -> T:
        logger.debug(f'Sending chat_completion API request...')
        if issubclass(output_schema, str):
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
            )
            content = response.choices[0].message.content
            return cast(T, content if content is not None else '')

        model_schema = output_schema
        
        if not as_tool:
            # use response_format
            parsed_completion = await self.client.beta.chat.completions.parse(
                model=model_name,
                messages=conversation,
                response_format=model_schema, 
            )
            
            parsed_result = parsed_completion.choices[0].message.parsed
            
            if parsed_result is None:
                raise ValueError('OpenAI refused to output structured data.')
            return cast(T, parsed_result)

        else:
            # use tool calling to define schema, as in pydantic_ai
            function_name = model_schema.__name__
            tool_definition: ChatCompletionFunctionToolParam = {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": f"Output data in the structure of {function_name}",
                    "parameters": model_schema.model_json_schema(), # type: ignore
                },
            }

            response = await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": function_name}},
            )

            message = response.choices[0].message
            
            if not message.tool_calls:
                raise ValueError('Model did not call the expected tool.')
            
            # Parse the arguments from the tool call back into the Pydantic model
            arguments_json = cast(str, message.tool_calls[0].function.arguments) # type: ignore
            return cast(T, model_schema.model_validate_json(arguments_json))

    @override
    async def _embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        logger.debug(f'Sending embed_text API request...')
        response = await self.client.embeddings.create(
            model=model_name, input=text,
        )
        return response.data[0].embedding