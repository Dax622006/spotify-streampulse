# spotify-streampulse/src/streampulse/ml/association.py
"""
Apriori Association Rule Mining.
R-ML-27: Transaction grain = User_ID; basket items = distinct Genres listened to by user.
100 users x 10 genres boolean basket matrix.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from streampulse.config import (
    APRIORI_MAX_LEN,
    APRIORI_MIN_CONFIDENCE,
    APRIORI_MIN_LIFT,
    APRIORI_MIN_SUPPORT,
)
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AssociationResult:
    basket_matrix: pd.DataFrame
    summary: dict[str, Any]
    itemsets: pd.DataFrame
    rules: pd.DataFrame


def build_genre_basket_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build boolean user-genre basket matrix (100 users x 10 genres)."""
    # Group genres per user
    user_genres = df.groupby("User_ID")["Genre"].unique()

    # One-hot encoded transaction matrix
    basket = pd.crosstab(df["User_ID"], df["Genre"]).astype(bool)

    sizes = basket.sum(axis=1)
    item_frequencies = basket.sum(axis=0).to_dict()

    summary = {
        "num_transactions": len(basket),
        "num_items": basket.shape[1],
        "basket_size_min": int(sizes.min()),
        "basket_size_max": int(sizes.max()),
        "basket_size_mean": round(float(sizes.mean()), 2),
        "item_frequencies": item_frequencies,
    }
    return basket, summary


def mine_association_rules(
    basket: pd.DataFrame,
    min_support: float = APRIORI_MIN_SUPPORT,
    min_confidence: float = APRIORI_MIN_CONFIDENCE,
    max_len: int = APRIORI_MAX_LEN,
    min_lift: float = APRIORI_MIN_LIFT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine frequent itemsets and association rules using mlxtend.

    Returns:
        (itemsets_df, rules_df)
    """
    if basket.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Frequent itemsets
    itemsets = apriori(
        basket,
        min_support=max(0.01, min(1.0, min_support)),
        use_colnames=True,
        max_len=max_len,
    )

    if itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets", "length"]), pd.DataFrame()

    itemsets["length"] = itemsets["itemsets"].apply(lambda s: len(s))
    itemsets["itemsets_str"] = itemsets["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    itemsets.sort_values(by="support", ascending=False, inplace=True)

    # Association rules
    try:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=max(0.01, min(1.0, min_confidence)),
        )
    except ValueError:
        # Occurs if itemsets has only length 1
        return itemsets, pd.DataFrame()

    if rules.empty:
        return itemsets, pd.DataFrame()

    # Filter by min_lift
    rules = rules[rules["lift"] >= min_lift].copy()

    # Format antecedents and consequents
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    # Clean display columns
    rules = rules[
        [
            "antecedents_str",
            "consequents_str",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]
    ].copy()

    rules.sort_values(by="lift", ascending=False, inplace=True)
    return itemsets, rules


def run_apriori_pipeline(
    df: pd.DataFrame,
    min_support: float = APRIORI_MIN_SUPPORT,
    min_confidence: float = APRIORI_MIN_CONFIDENCE,
    max_len: int = APRIORI_MAX_LEN,
    min_lift: float = APRIORI_MIN_LIFT,
) -> AssociationResult:
    """Run end-to-end Apriori pipeline."""
    basket, summary = build_genre_basket_matrix(df)
    itemsets, rules = mine_association_rules(
        basket,
        min_support=min_support,
        min_confidence=min_confidence,
        max_len=max_len,
        min_lift=min_lift,
    )
    return AssociationResult(
        basket_matrix=basket,
        summary=summary,
        itemsets=itemsets,
        rules=rules,
    )
