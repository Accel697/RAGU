from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from typing_extensions import override

from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.llm.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import FLOATS


T = TypeVar('T', BaseModel, str)


class Embedder(ABC):
    """Embedder interface to support various backends (openai, transformers etc.)."""

    @property
    def dim(self) -> int: ...

    @abstractmethod
    async def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS: ...

    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]] | FLOATS:
        logger.debug(f'Calling batch_embed_text with size {len(texts)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.embed_text(
                text=text,
                **kwargs,
            )
            for text in texts
        ], desc=desc)


class EmbedderOpenAI(Embedder):
    """Fixes model name and (possisbly) kwargs for CachedAsyncOpenAI client,
    to match Embedder interface.
    
    Example:
    ```
    embedder = EmbedderOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='my-embedding-model',
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
    async def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        return await self.client.embed_text(
            model_name=self.model_name,
            text=text,
            **(self.kwargs | kwargs),
        )
    
    @property
    @override
    def dim(self) -> int:
        return self._dim


# class STEmbedder(BaseEmbedder):
#     """
#     Embedder that uses Sentence Transformers to compute text embeddings.

#     Warning:
#     This embedder currently has limited support and can be unstable. Use OpenAIEmbedder instead.
#     """

#     def __init__(self, model_name_or_path: str, dim: int=None, *args, **kwargs):
#         """
#         Initializes the STEmbedder with a specified model.

#         :param model_name_or_path: Path or name of the Sentence Transformer model.
#         """

#         raise ImportError(
#             "[STEmbedder] Current support is limited and unstable. "
#             "Please, use OpenAIEmbedder for now."
#         )
#         try:
#             from sentence_transformers import SentenceTransformer
#         except ImportError:
#             raise ImportError(
#                 "RAGU needs SentenceTransformer to use this class. Please install it using `pip install sentence-transformers`."
#             )
#         super().__init__()
#         self.model = SentenceTransformer(model_name_or_path, **kwargs)
#         self.dim = dim or self.model.get_sentence_embedding_dimension()

#     async def embed(self, texts: list[str], batch_size: int=16) -> list[list[float]] | FLOATS:
#         """
#         Computes embeddings for a list of strings.

#         :param texts: Input text(s) to embed.
#         :param batch_size: Batch size.
#         :return: Embeddings for the input text(s).
#         """

#         batch_generator = BatchGenerator(texts, batch_size=batch_size)
#         embeddings_list = [
#             self.model.encode(batch, show_progress_bar=False)
#             for batch in batch_generator.get_batches()
#         ]

#         return np.concatenate(embeddings_list)
