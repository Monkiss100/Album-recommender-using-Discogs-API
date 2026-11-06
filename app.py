"""Album recommender powered by the Discogs API.

Discogs has no "similar albums" endpoint, so this app builds its own:
  1. Look up each album the user enters and read its genres, styles and year.
  2. Search Discogs for other albums sharing the most common styles.
  3. Score every candidate by style/genre overlap, closeness in year and
     popularity (how many Discogs users own it), then show the best matches.

Run:  DISCOGS_TOKEN=your_token python app.py
"""
import math
import os
import time
from collections import Counter

import requests
from flask import Flask, render_template, request

API_ROOT = "https://api.discogs.com"
USER_AGENT = "SimilarAlbums/1.0 +https://github.com/example/similar-albums"
CACHE_TTL = 3600          # seconds to remember a search result
MAX_SEEDS = 1             # how many albums a user may enter at once
STYLES_TO_SEARCH = 3      # how many of the seed styles search for
RESULTS_SHOWN = 5

app = Flask(__name__)
_cache = {}


class DiscogsError(Exception):
    """Raised with a message that is safe to show to the user."""


def discogs_search(**params):
    """Call /database/search with caching and friendly error messages."""
    token = os.environ.get("DISCOGS_TOKEN")
    if not token:
        raise DiscogsError(
            "DISCOGS_TOKEN is not set."
        )

    key = tuple(sorted(params.items()))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    try:
        resp = requests.get(
            f"{API_ROOT}/database/search",
            params=params,
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Discogs token={token}",
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        raise DiscogsError(f"Could not reach Discogs: {exc}") from exc

    if resp.status_code == 401:
        raise DiscogsError("Discogs rejected the token. Check DISCOGS_TOKEN.")
    if resp.status_code == 429:
        raise DiscogsError("Discogs rate limit hit (60 requests/minute). Try again shortly.")
    if not resp.ok:
        raise DiscogsError(f"Discogs returned HTTP {resp.status_code}.")

    results = resp.json().get("results", [])
    _cache[key] = (time.time(), results)
    return results


def split_title(full_title):
    """Discogs titles look like 'Artist - Album'."""
    if " - " in full_title:
        artist, album = full_title.split(" - ", 1)
        return artist.strip(), album.strip()
    return "", full_title.strip()


def parse_year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_seed(query):
    """Return the best matching master release for a line the user typed."""
    results = []
    if " - " in query:
        artist, album = (part.strip() for part in query.split(" - ", 1))
        results = discogs_search(artist=artist, release_title=album,
                                 type="master", per_page=5)
    if not results:
        results = discogs_search(q=query, type="master", per_page=5)
    return results[0] if results else None


def normalise(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def recommend(queries):
    seeds, missing = [], []
    for q in queries:
        found = find_seed(q)
        if found:
            seeds.append(found)
        else:
            missing.append(q)

    if not seeds:
        return [], [], missing

    # Build a taste profile from all seeds.
    style_counts, genre_counts, years = Counter(), Counter(), []
    for s in seeds:
        style_counts.update(s.get("style", []))
        genre_counts.update(s.get("genre", []))
        year = parse_year(s.get("year"))
        if year:
            years.append(year)
    seed_styles, seed_genres = set(style_counts), set(genre_counts)
    seed_year = sum(years) / len(years) if years else None

    # Pull candidates for the most common styles (falling back to genres).
    candidates = {}
    searches = [("style", s) for s, _ in style_counts.most_common(STYLES_TO_SEARCH)]
    if not searches:
        searches = [("genre", g) for g, _ in genre_counts.most_common(2)]
    for field, value in searches:
        for c in discogs_search(**{field: value}, type="master",
                                format="Album", per_page=100):
            candidates.setdefault(c["id"], c)

    # Anything the user already entered, or by the same artists, is excluded.
    seed_ids = {s["id"] for s in seeds}
    seed_keys = {normalise(s["title"]) for s in seeds}
    seed_artists = {normalise(split_title(s["title"])[0]) for s in seeds}

    scored = []
    for c in candidates.values():
        artist, album = split_title(c["title"])
        if c["id"] in seed_ids or normalise(c["title"]) in seed_keys:
            continue
        if normalise(artist) in seed_artists:
            continue

        c_styles, c_genres = set(c.get("style", [])), set(c.get("genre", []))
        shared = [s for s, _ in style_counts.most_common() if s in c_styles]
        if not shared and not (c_genres & seed_genres):
            continue

        score = jaccard(c_styles, seed_styles) * 5 + jaccard(c_genres, seed_genres) * 1.5
        # Reward styles that appear in several of the user's albums.
        score += sum(style_counts[s] for s in shared) / (len(seeds) * 4)

        year = parse_year(c.get("year"))
        if seed_year and year:
            score += max(0.0, 1 - abs(year - seed_year) / 30) * 1.5

        have = (c.get("community") or {}).get("have", 0)
        score += math.log10(have + 1) * 0.4

        scored.append({
            "artist": artist,
            "album": album,
            "year": year,
            "url": "https://www.discogs.com" + c.get("uri", ""),
            "image": c.get("cover_image") or c.get("thumb"),
            "shared_styles": shared[:4],
            "genres": c.get("genre", []),
            "owners": have,
            "score": score,
            "artist_key": normalise(artist),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)

    # At most one album per artist keeps the list varied.
    picked, seen_artists = [], set()
    for rec in scored:
        if rec["artist_key"] in seen_artists:
            continue
        seen_artists.add(rec["artist_key"])
        picked.append(rec)
        if len(picked) == RESULTS_SHOWN:
            break

    seed_info = []
    for s in seeds:
        artist, album = split_title(s["title"])
        seed_info.append({
            "artist": artist,
            "album": album,
            "year": parse_year(s.get("year")),
            "styles": s.get("style", []),
            "image": s.get("thumb"),
        })
    return picked, seed_info, missing


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"query": "", "recs": None, "seeds": [], "missing": [], "error": None}
    if request.method == "POST":
        raw = request.form.get("albums", "")
        ctx["query"] = raw
        queries = [line.strip() for line in raw.splitlines() if line.strip()][:MAX_SEEDS]
        if not queries:
            ctx["error"] = "Enter at least one album."
        else:
            try:
                ctx["recs"], ctx["seeds"], ctx["missing"] = recommend(queries)
            except DiscogsError as exc:
                ctx["error"] = str(exc)
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
