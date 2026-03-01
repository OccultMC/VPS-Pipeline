"""Utility functions for fetching JSON from URLs."""

import json
from typing import Callable, Optional

from aiohttp import ClientSession


async def get_json_async(
    url: str,
    session: ClientSession,
    json_function_parameters: dict = None,
    preprocess_function: Callable = None,
    headers: dict = None,
) -> dict:
    """Fetch JSON from a URL using aiohttp."""
    if headers is None:
        headers = {}
    async with session.get(url, headers=headers) as response:
        if preprocess_function:
            text = await response.text()
            processed = preprocess_function(text)
            return json.loads(processed)
        else:
            if json_function_parameters:
                return await response.json(**json_function_parameters)
            return await response.json()


def try_get(accessor):
    """Safely access nested list/dict elements. Returns None on IndexError/TypeError/KeyError."""
    try:
        return accessor()
    except (IndexError, TypeError, KeyError):
        return None
