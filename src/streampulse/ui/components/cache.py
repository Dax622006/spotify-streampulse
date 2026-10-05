# spotify-streampulse/src/streampulse/ui/components/cache.py
"""
Cached wrappers around warehouse reads and ML training for the UI.

Without caching, Streamlit re-runs the whole page on every click or slider move,
which re-queried SQLite and re-trained every model each time. These wrappers
compute once and reuse the result until the warehouse file changes (the file's
modification time is part of the cache key, so rebuilding the pipeline
automatically invalidates everything).
"""
from __future__ import annotations

import functools

import streamlit as st

from streampulse.config import WAREHOUSE_FILE
from streampulse.ml import association as _assoc
from streampulse.ml import classification as _clf
from streampulse.ml import clustering as _clu
from streampulse.ml import regression as _reg
from streampulse.warehouse import repository as _repo


def _stamp() -> tuple[int, int]:
    """Cache-busting key that changes whenever the warehouse file is rebuilt."""
    try:
        s = WAREHOUSE_FILE.stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return (0, 0)


# --- Warehouse reads (no DataFrame argument) --------------------------------
def _cached_reader(fn):
    # `name` is part of the cache key: every wrapper shares this inner function,
    # so without it different functions would collide and return each other's results.
    @st.cache_data(show_spinner=False)
    def _inner(name, _stamp_key, args, kwargs):
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    def outer(*args, **kwargs):
        return _inner(fn.__name__, _stamp(), args, kwargs)

    return outer


get_analytics_frame = _cached_reader(_repo.get_analytics_frame)
olap_aggregate = _cached_reader(_repo.olap_aggregate)


# --- ML training (first argument is the analytics DataFrame) ----------------
def _cached_trainer(fn):
    """The DataFrame is always the warehouse analytics frame, so it is excluded
    from hashing (leading underscore); the warehouse stamp + parameters are the key."""

    @st.cache_data(show_spinner=False)
    def _inner(name, _df, _stamp_key, args, kwargs):
        return fn(_df, *args, **kwargs)

    @functools.wraps(fn)
    def outer(df, *args, **kwargs):
        return _inner(fn.__name__, df, _stamp(), args, kwargs)

    return outer


train_decision_tree = _cached_trainer(_clf.train_decision_tree)
train_naive_bayes = _cached_trainer(_clf.train_naive_bayes)
train_slr = _cached_trainer(_reg.train_slr)
train_mlr = _cached_trainer(_reg.train_mlr)
evaluate_kmeans_k_range = _cached_trainer(_clu.evaluate_kmeans_k_range)
train_kmeans = _cached_trainer(_clu.train_kmeans)
train_agglomerative = _cached_trainer(_clu.train_agglomerative)
run_apriori_pipeline = _cached_trainer(_assoc.run_apriori_pipeline)
