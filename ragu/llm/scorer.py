from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import override

from pydantic import BaseModel
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from ragu.common.logger import logger
from ragu.common.batch_generator import BatchGenerator
from ragu.llm.openai import CachedAsyncOpenAI


T = TypeVar('T', BaseModel, str)


class Scorer(ABC):
    """Scorer interface to support various backends (openai, transformers etc.)."""

    @abstractmethod
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Scores text similarity with a list of other texts. Returns tuples
        of (index, score) sorted by score in descending order.
        
        Subclasses may add more kwargs."""

    async def batch_score(
        self,
        texts: list[tuple[str, list[str]]],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[tuple[int, float]]]:
        """Parallel async processing of multiple score calls."""
        logger.debug(f'Calling batch_score with size {len(texts)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.score(
                text_1=text_1,
                text_2=text_2,
                **kwargs,
            )
            for text_1, text_2 in texts
        ], desc=desc)


class ScorerOpenAI(Scorer):
    """Fixes model name and (possisbly) kwargs for CachedAsyncOpenAI client,
    to match Scorer interface.
    
    Example:
    ```
    scorer = ScorerOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='my-model',
    )
    ```
    """
    
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        dim: int,
        **kwargs: Any,
    ):
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
        self._dim = dim
    
    @override
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        return await self.client.score(
            model_name=self.model_name,
            text_1=text_1,
            text_2=text_2,
            **(self.kwargs | kwargs),
        )

class ScorerCrossEncoder(Scorer):
    """Scorer (reranker) that uses Sentence Transformers
    CrossEncoder to compute relevance scores.
    """

    def __init__(self, model: CrossEncoder, batch_size: int = 16):
        self.model = model
        self.batch_size = batch_size
    
    @override
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        pairs = [(text_1, doc) for doc in text_2]
        batch_generator = BatchGenerator(pairs, batch_size=batch_size or self.batch_size)
        
        scores_list: list[float] = []
        for batch in batch_generator.get_batches():
            batch_scores = self.model.predict(batch, show_progress_bar=False) # type: ignore
            scores_list.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores)) # type: ignore

        indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores
