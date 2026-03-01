def is_third_party_panoid(panoid: str) -> bool:
    """Returns whether a pano ID is from a third-party rather than Google."""
    return panoid.startswith("CIHM0og") or len(panoid) > 22
