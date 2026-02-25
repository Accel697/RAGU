import asyncio
from collections.abc import Awaitable, Collection, MutableMapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
import functools
from hashlib import md5
import logging
from pathlib import Path
from typing import Callable, Any, TypeVar, cast
from typing import List

from diskcache import Index # pyright: ignore[reportMissingTypeStubs]
import loguru
import numpy as np
import numpy.typing as npt

from ragu.common.logger import logger


FLOATS = npt.NDArray[np.floating[Any]]
"""A typization for numpy array of floats"""

INTS = npt.NDArray[np.integer[Any]]
"""A typization for numpy array of integers"""

_dish_caches: dict[str, Index] = {}

def get_disk_cache(dir: str | Path) -> MutableMapping[str, Any]:
    """Get or create a DiskCache by a directory name.
    Cache is shared between multiple `get_disk_cache` calls.
    """
    path = str(Path(dir).resolve())
    if (cache := _dish_caches.get(path, None)):
        return cache
    _dish_caches[path] = cache = Index(path)
    return cache


T_fn = TypeVar('T_fn', bound=Callable[..., Awaitable[Any]])

def acontexts(
    func: T_fn,
    *contexts: AbstractAsyncContextManager[Any],
) -> T_fn:
    """Wraps the `func` into the given async contexts."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug('attach_async_contexts: entering context...')
        async with AsyncExitStack() as stack:
            for mgr in contexts:
                await stack.enter_async_context(mgr)
            logger.debug('attach_async_contexts: entered context!')
            return await func(*args, **kwargs)
            
    return cast(T_fn, wrapper)


class LoguruAdapter(logging.Logger):
    # is neeed where some tool requires a logging.Logger, but we have loguru
    def __init__(self, name: str):
        super().__init__(name)
        
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False): # type: ignore
        # We override the internal _log method to intercept standard logging calls
        # and redirect them to loguru.
        
        # Map integer levels to loguru level names/values
        # Note: stacklevel=2 usually helps valid source file reporting
        loguru_opts = loguru.logger.opt(depth=2, exception=exc_info) # type: ignore
        
        # Handle standard logging's printf style formatting (%s) 
        # vs Loguru's mechanism
        try:
             # Standard logging expands args eagerly usually, 
             # but here we might just pass the message formatted
             formatted_msg = msg % args if args else msg # type: ignore
        except TypeError:
             formatted_msg = msg # Fallback # type: ignore

        loguru_opts.log(level, formatted_msg) # type: ignore


def compute_mdhash_id(*args: str, prefix: str = '', **kwargs: str) -> str:
    """A unique string hash for the given combination of arguments.
    Invariant to kwargs order.
    """
    string = ''
    for x in args:
        assert isinstance(x, str)
        string += '\0' + x
    for key, x in sorted(kwargs.items(), key=lambda item: item[0]):
        assert isinstance(x, str)
        string += '\0' + key + '\1' + x
    return prefix + md5(string.encode()).hexdigest()


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError()
        return current_loop

    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def read_text_from_files(directory: str | Path, file_extensions: Collection[str] | None = None) -> List[str]:
    texts: list[str] = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Cannot read file {file_path}: {e}")

    return texts

