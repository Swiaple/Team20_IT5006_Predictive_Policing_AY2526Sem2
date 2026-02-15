import os
import math
import pandas as pd
import requests
from tqdm import tqdm

# Chicago Crimes dataset (Socrata)
BASE = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

def fetch_chunk(where: str, limit: int, offset: int) -> pd.DataFrame:
    params = {
        "$select": "date,primary_type,latitude,longitude",
        "$where": where,
        "$limit": limit,
        "$offset": offset,
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    # Socrata returns CSV text
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def fetch_count(where: str) -> int:
    params = {"$select": "count(*) as n", "$where": where}
    r = requests.get("https://data.cityofchicago.org/resource/ijzp-q8t2.json", params=params, timeout=60)
    r.raise_for_status()
    return int(r.json()[0]["n"])

def download_period(start_iso: str, end_iso: str, out_csv: str, chunk_size: int = 50000):
    # only downloading that having coordinate
    where = (
        f"date between '{start_iso}' and '{end_iso}' "
        "AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )

    total = fetch_count(where)
    print(f"Total rows in period: {total}")
    if total == 0:
        print("No data for this period.")
        return

    chunks = math.ceil(total / chunk_size)
    print(f"Downloading in {chunks} chunk(s) of {chunk_size}...")

    first = True
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    for i in tqdm(range(chunks)):
        offset = i * chunk_size
        df = fetch_chunk(where, limit=chunk_size, offset=offset)
        if df.empty:
            break
        df.to_csv(out_csv, mode="w" if first else "a", index=False, header=first)
        first = False

    print(f"Saved to: {out_csv}")

if __name__ == "__main__":
    start = "2024-01-01T00:00:00.000"
    end   = "2025-01-01T00:00:00.000"
    download_period(start, end, out_csv="data/crimes_2024.csv")
