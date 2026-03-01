"""Google Street View API endpoints — coverage tile discovery only."""

import aiohttp
from ..util import get_json_async


def build_coverage_tile_request_url(tile_x: int, tile_y: int) -> str:
    """Build the URL for fetching a coverage tile at zoom level 17."""
    return f"https://www.google.com/maps/photometa/ac/v1?pb=!1m1!1smaps_sv.tactile!6m3!1i{tile_x}!2i{tile_y}!3i17!8b1"


async def get_coverage_tile_async(tile_x: int, tile_y: int, session: aiohttp.ClientSession) -> dict:
    """Fetch a coverage tile and return the raw parsed JSON response."""
    return await get_json_async(
        build_coverage_tile_request_url(tile_x, tile_y),
        session,
        preprocess_function=lambda text: text[4:],
    )
