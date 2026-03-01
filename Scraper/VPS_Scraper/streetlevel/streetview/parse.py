"""Parse coverage tile responses into panorama metadata.

Simplified: only extracts (id, lat, lon, heading, elevation) — no links, no full metadata.
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PanoMetadata:
    """Minimal panorama metadata from a coverage tile."""
    id: str
    lat: float
    lon: float
    heading: float = 0.0  # radians
    pitch: float = 0.0    # radians
    roll: float = 0.0     # radians
    elevation: float = 0.0


def parse_coverage_tile_response(tile: list) -> List[PanoMetadata]:
    """Parse a coverage tile API response into a list of PanoMetadata.

    Returns an empty list if the tile contains no panoramas.
    """
    if tile is None:
        return []

    panos = []
    try:
        if tile[1] is not None and len(tile[1]) > 0:
            for raw_pano in tile[1][1]:
                try:
                    if raw_pano[0][0] == 1:
                        continue
                    panos.append(
                        PanoMetadata(
                            id=raw_pano[0][0][1],
                            lat=raw_pano[0][2][0][2],
                            lon=raw_pano[0][2][0][3],
                            heading=math.radians(raw_pano[0][2][2][0]),
                            pitch=math.radians(90 - raw_pano[0][2][2][1]),
                            roll=math.radians(raw_pano[0][2][2][2]),
                            elevation=raw_pano[0][2][1][0],
                        )
                    )
                except (IndexError, TypeError, KeyError):
                    continue
    except (IndexError, TypeError, KeyError):
        pass

    return panos
