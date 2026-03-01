"""Reverse geocoding to generate hierarchical paths like US/California/Sacramento."""

import asyncio
import logging
import re
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

logger = logging.getLogger(__name__)

_geocoder = Nominatim(user_agent="vpr-scraper/1.0")

# US state abbreviation to full name
STATE_ABBREVS = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "NewHampshire", "NJ": "NewJersey", "NM": "NewMexico", "NY": "NewYork",
    "NC": "NorthCarolina", "ND": "NorthDakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "RhodeIsland", "SC": "SouthCarolina",
    "SD": "SouthDakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "WestVirginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "DistrictOfColumbia",
}


def sanitize_path_segment(s: str) -> str:
    """Remove special characters from a path segment, keep alphanumeric."""
    s = s.strip()
    # Remove common suffixes
    for suffix in [" County", " Parish", " Borough", " Census Area", " Municipality"]:
        s = s.replace(suffix, "")
    # Remove non-alphanumeric except spaces
    s = re.sub(r"[^a-zA-Z0-9 ]", "", s)
    # CamelCase: capitalize each word, remove spaces
    return "".join(word.capitalize() for word in s.split())


async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Reverse geocode a lat/lon to a hierarchical path.

    Returns a path like "US/California/Sacramento" or None on failure.
    """
    try:
        loop = asyncio.get_event_loop()
        location = await loop.run_in_executor(
            None, lambda: _geocoder.reverse(f"{lat}, {lon}", language="en", addressdetails=True)
        )

        if not location or not location.raw.get("address"):
            return None

        addr = location.raw["address"]

        # Extract country
        country_code = addr.get("country_code", "").upper()
        if country_code == "US":
            country = "US"
        elif country_code == "CA":
            country = "CA"
        else:
            country = sanitize_path_segment(addr.get("country", country_code))

        # Extract state/province
        state = addr.get("state", "")
        if country_code == "US":
            # Try to use full state name
            state_abbrev = addr.get("ISO3166-2-lvl4", "").replace("US-", "")
            if state_abbrev in STATE_ABBREVS:
                state = STATE_ABBREVS[state_abbrev]
            else:
                state = sanitize_path_segment(state)
        else:
            state = sanitize_path_segment(state)

        # Extract city/place
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or addr.get("county")
            or ""
        )
        city = sanitize_path_segment(city)

        if not city:
            return None

        return f"{country}/{state}/{city}"

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed for ({lat}, {lon}): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected geocoding error for ({lat}, {lon}): {e}")
        return None


def get_city_name_from_path(path: str) -> str:
    """Extract city name from the last segment of a path."""
    parts = path.strip("/").split("/")
    return parts[-1] if parts else "Unknown"
