from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import override

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI


T = TypeVar('T', BaseModel, str)


class LLM(ABC):
    """LLM interface to support various backends (openai, transformers etc.)."""

    @abstractmethod
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """Returns LLM response, in form or string or BaseModel. Subclasses
        may add more kwargs, such as temperature.
        """

    async def batch_chat_completion(
        self,
        conversations: list[list[ChatCompletionMessageParam]],
        output_schema: type[T] = str,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Sequence[T]:
        """Parallel async processing of multiple chat_completion calls."""
        logger.debug(f'Calling batch_chat_completion with size {len(conversations)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.chat_completion(
                conversation=conversation,
                output_schema=output_schema,
                **kwargs,
            )
            for conversation in conversations
        ], desc=desc)
        


class LLMOpenAI(LLM):
    """Fixes model name and (possisbly) kwargs for CachedAsyncOpenAI client,
    to match LLM interface.
    
    Example:
    ```
    llm = LLMOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='gpt-5',
    )
    ```
    """

    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        **kwargs: Any,
    ):
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
    
    @override
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        return await self.client.chat_completion(
            model_name=self.model_name,
            conversation=conversation,
            output_schema=output_schema,
            **(self.kwargs | kwargs),
        )