from pathlib import Path

import pandas as pd
import numpy as np


def load_p4_data():
    acts = pd.read_csv(
        "interim/clean/p4_ready/acts_clean.csv"
    )

    etm = pd.read_csv(
        "interim/clean/p4_ready/etm_clean.csv"
    )

    registry = pd.read_parquet(
        "interim/registry.parquet"
    )

    acts["date"] = pd.to_datetime(
        acts["date"],
        errors="coerce"
    )

    etm["date"] = pd.to_datetime(
        etm["date"],
        errors="coerce"
    )

    registry["ts"] = pd.to_datetime(
        registry["ts"],
        errors="coerce"
    )

    acts["period"] = (
        acts["date"]
        .dt.to_period("M")
        .astype(str)
    )

    etm["period"] = (
        etm["date"]
        .dt.to_period("M")
        .astype(str)
    )

    registry["period"] = (
        registry["ts"]
        .dt.to_period("M")
        .astype(str)
    )

    return acts, etm, registry


def prepare_acts(acts):
    a = acts.copy()

    a["tickets10"] = (
        a["tickets10"]
        .fillna("")
        .astype(str)
    )

    a = a[
        a["tickets10"] != ""
    ].copy()

    a["ticket10"] = (
        a["tickets10"]
        .str.split()
    )

    a = a.explode(
        "ticket10"
    )

    a["amount_1c"] = (
        a["debt_delta"] /
        a["n_tickets"].replace(
            0,
            np.nan
        )
    )

    a["op_type"] = (
        a["line_type"]
        .map({
            "sale": "sale",
            "service_fee": "sale",
            "refund": "refund"
        })
    )

    return a[
        [
            "subagent_key",
            "period",
            "ticket10",
            "op_type",
            "amount_1c",
            "pnr",
            "pay_doc",
            "date",
            "doc"
        ]
    ]


def prepare_etm(etm):
    e = etm.copy()

    e["tickets10"] = (
        e["tickets10"]
        .fillna("")
        .astype(str)
    )

    e = e[
        e["tickets10"] != ""
    ].copy()

    e["ticket10"] = (
        e["tickets10"]
        .str.split()
    )

    e = e.explode(
        "ticket10"
    )

    e["amount_etm"] = (
        e["debt_delta"] /
        e["n_tickets"].replace(
            0,
            np.nan
        )
    )

    e["op_type"] = (
        e["kind_en"]
        .map({
            "purchase": "sale",
            "refund": "refund",
            "void": "void"
        })
    )

    return e[
        [
            "agent_key",
            "period",
            "ticket10",
            "op_type",
            "amount_etm",
            "pnr",
            "pay_doc",
            "date",
            "txn_id"
        ]
    ]


def match_operations(acts, etm, registry):
    a = prepare_acts(acts)
    e = prepare_etm(etm)

    a = a.rename(
        columns={
            "subagent_key": "subagent_id"
        }
    )

    e = e.rename(
        columns={
            "agent_key": "subagent_id"
        }
    )

    keys = [
        "subagent_id",
        "period",
        "ticket10",
        "op_type"
    ]

    result = a.merge(
        e,
        on=keys,
        how="outer",
        suffixes=(
            "_1c",
            "_etm"
        ),
        indicator=True
    )

    result["amount_1c"] = (
        result["amount_1c"]
        .fillna(0)
    )

    result["amount_etm"] = (
        result["amount_etm"]
        .fillna(0)
    )

    result["difference_1c_etm"] = (
        result["amount_1c"] -
        result["amount_etm"]
    ).round(2)

    result["match_status"] = np.select(
        [
            (
                result["_merge"].eq("both") &
                result[
                    "difference_1c_etm"
                ].abs().le(1)
            ),
            result["_merge"].eq("both"),
            result["_merge"].eq("left_only"),
            result["_merge"].eq("right_only")
        ],
        [
            "matched",
            "amount_difference",
            "only_1c",
            "only_etm"
        ],
        default="unknown"
    )

    result = result.drop(
        columns="_merge"
    )

    r = registry.copy()

    r = r[
        (r["party_type"] == "subagent") &
        r["subagent_id"].notna() &
        r["ticket10"].notna()
    ].copy()

    r = (
        r.groupby(
            keys,
            as_index=False
        )
        .agg(
            amount_registry=(
                "amount_kgs",
                "sum"
            ),
            registry_rows=(
                "row_id",
                "count"
            ),
            party_match=(
                "party_match",
                "first"
            ),
            parse_status=(
                "parse_status",
                "first"
            )
        )
    )

    # Registry uses the opposite sign.
    r["amount_registry"] = (
        -r["amount_registry"]
    )

    result = result.merge(
        r,
        on=keys,
        how="left"
    )

    result["amount_registry"] = (
        result["amount_registry"]
        .fillna(0)
    )

    result["difference_1c_registry"] = (
        result["amount_1c"] -
        result["amount_registry"]
    ).round(2)

    result["difference_etm_registry"] = (
        result["amount_etm"] -
        result["amount_registry"]
    ).round(2)

    return result


def build_balance(acts, etm):
    # 1C balance
    act_balance = (
        acts.groupby(
            [
                "subagent_key",
                "period"
            ],
            as_index=False
        )
        .agg(
            saldo_start=(
                "saldo_start",
                "first"
            ),
            saldo_end=(
                "saldo_end",
                "first"
            ),
            turnover_1c=(
                "debt_delta",
                "sum"
            )
        )
        .rename(
            columns={
                "subagent_key":
                    "subagent_id"
            }
        )
    )

    # ETM transaction data
    e = etm.copy()

    e["date"] = pd.to_datetime(
        e["date"],
        errors="coerce"
    )

    e = e.sort_values(
        [
            "agent_key",
            "date",
            "txn_id"
        ]
    )

    # ETM turnover in balance terms
    etm_turnover = (
        e.groupby(
            [
                "agent_key",
                "period"
            ],
            as_index=False
        )
        .agg(
            turnover_etm=(
                "amount_kgs",
                "sum"
            )
        )
        .rename(
            columns={
                "agent_key":
                    "subagent_id"
            }
        )
    )

    # ETM month-end balances from P2
    etm_month_end = pd.read_csv(
        "interim/clean/etm_month_end_balance.csv"
    )

    etm_month_end["period"] = (
        etm_month_end["period"]
        .astype(str)
    )

    # Map ETM display name to agent_key
    name_map = (
        e[
            [
                "subagent",
                "agent_key"
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    etm_month_end = etm_month_end.merge(
        name_map,
        on="subagent",
        how="left"
    )

    etm_month_end = etm_month_end.rename(
        columns={
            "agent_key":
                "subagent_id"
        }
    )

    # Previous month closing balance
    etm_balance = (
        etm_month_end
        .sort_values(
            [
                "subagent_id",
                "period"
            ]
        )
        .copy()
    )

    etm_balance[
        "etm_balance_start"
    ] = (
        etm_balance
        .groupby(
            "subagent_id"
        )[
            "etm_balance_end"
        ]
        .shift(1)
        .fillna(0)
    )

    etm_balance = etm_balance.merge(
        etm_turnover,
        on=[
            "subagent_id",
            "period"
        ],
        how="left"
    )

    etm_balance[
        "turnover_etm"
    ] = (
        etm_balance[
            "turnover_etm"
        ].fillna(0)
    )

    etm_balance = etm_balance[
        [
            "subagent_id",
            "period",
            "etm_balance_start",
            "etm_balance_end",
            "turnover_etm"
        ]
    ]

    # Combine 1C and ETM
    balance = act_balance.merge(
        etm_balance,
        on=[
            "subagent_id",
            "period"
        ],
        how="outer"
    )

    numeric_columns = [
        "saldo_start",
        "saldo_end",
        "turnover_1c",
        "etm_balance_start",
        "etm_balance_end",
        "turnover_etm"
    ]

    for column in numeric_columns:
        balance[column] = (
            balance[column]
            .fillna(0)
        )

    # Internal 1C balance check
    balance["check_1c"] = (
        balance["saldo_start"] +
        balance["turnover_1c"] -
        balance["saldo_end"]
    ).round(2)

    # Internal ETM balance check
    balance["check_etm"] = (
        balance["etm_balance_start"] +
        balance["turnover_etm"] -
        balance["etm_balance_end"]
    ).round(2)

    # Difference between 1C and ETM
    balance["opening_difference"] = (
        balance["saldo_start"] +
        balance["etm_balance_start"]
    ).round(2)

    balance["turnover_difference"] = (
        balance["turnover_1c"] +
        balance["turnover_etm"]
    ).round(2)

    balance["closing_difference"] = (
        balance["saldo_end"] +
        balance["etm_balance_end"]
    ).round(2)

    # Overall bridge check
    balance["bridge_check"] = (
        balance["opening_difference"] +
        balance["turnover_difference"] -
        balance["closing_difference"]
    ).round(2)

    return balance


def build_summary(matches):
    summary = (
        matches.groupby(
            "subagent_id",
            as_index=False
        )
        .agg(
            operations=(
                "ticket10",
                "size"
            ),
            matched=(
                "match_status",
                lambda x:
                    (x == "matched").sum()
            ),
            amount_difference=(
                "difference_1c_etm",
                "sum"
            ),
            only_1c=(
                "match_status",
                lambda x:
                    (x == "only_1c").sum()
            ),
            only_etm=(
                "match_status",
                lambda x:
                    (x == "only_etm").sum()
            )
        )
    )

    summary["matched_share"] = (
        summary["matched"] /
        summary["operations"]
    ).round(4)

    return summary


def run_p4():
    print(
        "[P4] Загрузка данных..."
    )

    acts, etm, registry = (
        load_p4_data()
    )

    print(
        "[P4] Сопоставление операций..."
    )

    matches = match_operations(
        acts,
        etm,
        registry
    )

    print(
        "[P4] Построение баланса..."
    )

    balance = build_balance(
        acts,
        etm
    )

    summary = build_summary(
        matches
    )

    output = Path(
        "interim/p4"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    matches.to_csv(
        output /
        "p4_operation_matches.csv",
        index=False,
        encoding="utf-8-sig"
    )

    balance.to_csv(
        output /
        "p4_balance_bridge.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        output /
        "p4_subagent_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[P4] Готово."
    )

    print(
        f"[P4] Результаты: {output}"
    )




if __name__ == "__main__":
    run_p4()