from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """
    Batch data into tuples of length n. The last batch may be shorter.
    Backport of itertools.batched (Python 3.12+) for older versions.
    """
    if n < 1:
        msg = "n must be at least one"
        raise ValueError(msg)
    iterator = iter(iterable)
    while current_batch := tuple(islice(iterator, n)):
        yield current_batch


async def async_batched(iterable: AsyncIterable[T], n: int) -> AsyncIterator[tuple[T, ...]]:
    """
    Asynchronous version of batched for streaming large async datasets.
    """
    if n < 1:
        msg = "n must be at least one"
        raise ValueError(msg)

    batch = []
    async for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield tuple(batch)
            batch.clear()

    if batch:
        yield tuple(batch)
