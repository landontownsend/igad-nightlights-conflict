"""Shared helpers: settings, file paths, and downloads."""

import time
from pathlib import Path

import requests
import yaml


def load_config(path="config.yaml"):
    """Read config.yaml and return it as a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def interim(cfg, name):
    """Path of a file in data/interim/, creating the folder if needed."""
    folder = Path(cfg["paths"]["interim"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def processed(cfg, name):
    """Path of a file in data/processed/, creating the folder if needed."""
    folder = Path(cfg["paths"]["processed"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def downloads(cfg, subfolder):
    """A folder inside downloads/ for large raw files that are never committed."""
    folder = Path(cfg["paths"]["downloads"]) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def download(url, dest, tries=4, timeout=300):
    """
    Download a file to dest, retrying a few times if the server is busy.
    Returns dest, or None if the file does not exist on the server (404).
    """
    dest = Path(dest)
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
                return dest
        except requests.RequestException as e:
            if attempt == tries:
                raise
            print(f"   download failed ({e}); retrying in {10 * attempt} s")
            time.sleep(10 * attempt)


def country_codes(cfg):
    """The ISO3 codes of the study countries, e.g. ['SDN', 'SSD', ...]."""
    return [c["iso3"] for c in cfg["countries"]]
