# schema_tools.py
from typing import Dict, Any, List
import json
from cache import ttl_cache   # 앞서 드린 ttl_cache 데코레이터

SCHEMA_PATH = "Chinook_Column_Config.json"

@ttl_cache(ttl_seconds=600)
def get_schema_cache() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@ttl_cache(ttl_seconds=600)
def get_relations_cache() -> List[str]:
    return [
        "Album.ArtistId -> Artist.ArtistId",
        "Track.AlbumId -> Album.AlbumId",
        "Track.GenreId -> Genre.GenreId",
        "InvoiceLine.TrackId -> Track.TrackId",
        "InvoiceLine.InvoiceId -> Invoice.InvoiceId",
        "Invoice.CustomerId -> Customer.CustomerId"
    ]
