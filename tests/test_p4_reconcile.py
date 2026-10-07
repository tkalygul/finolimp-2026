"""Тесты сопоставления операций и моста баланса (P4).

Как запустить:
    pytest tests/test_p4_reconcile.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaning import etm_month_end_balance
from p4_reconcile import (
    build_balance, build_bridge, etm_balances, match_operations, match_payment_rows, run_p4,
)

REPO = Path(__file__).resolve().parent.parent
INTERIM = REPO / "interim"


# ================================================================
# Помощники: маленькие таблицы в формате P2/P3
# ================================================================

def act(sub, line_type, tickets="", debet=0.0, credit=0.0, period="2026-01", date=None, doc="",
        saldo_start=0.0, saldo_end=0.0):
    return {
        "subagent": sub, "subagent_key": sub, "period": period,
        "date": pd.Timestamp(date or f"{period}-10"), "doc": doc or f"{line_type} {tickets}",
        "tickets10": tickets or np.nan, "debet": debet, "credit": credit,
        "debt_delta": round(debet - credit, 2), "line_type": line_type, "pnr": "PNR001",
        "is_payment": line_type in ("payment_bank", "payment_cash"), "pay_doc": None,
        "saldo_start": saldo_start, "saldo_end": saldo_end,
    }


def etm_op(sub, txn, kind_en, tickets="", amount_kgs=0.0, balance_after=0.0, agreement="AGR-1",
           period="2026-01", date=None):
    return {
        "subagent": sub, "agent_key": sub, "agreement_id": agreement, "txn_id": txn, "kind_en": kind_en,
        "tickets10": tickets or np.nan, "amount_kgs": amount_kgs, "debt_delta": -amount_kgs,
        "balance_after": balance_after, "period": period,
        "date": pd.Timestamp(date or f"{period}-10 12:00:00"), "pnr": "PNR001", "pay_doc": None,
        "comment": "", "creator_en": "bot",
    }


def reg(sub, row_id, ticket, op_type, amount_kgs):
    return {
        "party_type": "subagent", "subagent_id": sub, "ticket10": ticket, "op_type": op_type,
        "amount_kgs": amount_kgs, "row_id": row_id, "party_match": "exact", "parse_status": "ok",
    }


EMPTY_REG = pd.DataFrame([reg("x", "r:0", "0000000000", "sale", 0.0)]).iloc[:0]


def statuses(acts, etm, registry=EMPTY_REG):
    acts = pd.DataFrame(acts, columns=list(act("x", "sale").keys()))
    m = match_operations(acts, pd.DataFrame(etm), registry)
    return m.set_index(["ticket10", "op_type"])


# ================================================================
# Сопоставление операций
# ================================================================

def test_sale_and_service_fee_are_one_operation():
    """Продажа и сервисный сбор 1С против одной строки ETM — одна сошедшаяся операция, а не две."""
    m = statuses(
        [act("alpha", "sale", "2420845724", debet=19345.23),
         act("alpha", "service_fee", "2420845724", debet=800.0)],
        [etm_op("alpha", 1, "purchase", "2420845724", -20145.23)],
    )
    assert len(m) == 1
    row = m.loc[("2420845724", "sale")]
    assert row["match_status"] == "matched"
    assert row["amount_1c"] == pytest.approx(20145.23)
    assert row["lines_1c"] == 2


def test_key_is_unique():
    m = match_operations(pd.DataFrame([
        act("alpha", "sale", "1000000001 1000000002", debet=2000.0),
        act("alpha", "service_fee", "1000000001 1000000002", debet=200.0),
    ]), pd.DataFrame([etm_op("alpha", 1, "purchase", "1000000001 1000000002", -2200.0)]), EMPTY_REG)
    assert not m.duplicated(["subagent_id", "ticket10", "op_type"]).any()
    assert (m["match_status"] == "matched").all()


def test_void_nets_purchase():
    m = statuses([], [etm_op("alpha", 1, "purchase", "1000000004", -700.0),
                      etm_op("alpha", 2, "void", "1000000004", 700.0)])
    assert list(m["match_status"]) == ["voided"]


def test_voided_ticket_posted_in_1c_is_a_difference():
    m = statuses([act("alpha", "sale", "1000000004", debet=700.0)],
                 [etm_op("alpha", 1, "purchase", "1000000004", -700.0),
                  etm_op("alpha", 2, "void", "1000000004", 700.0)])
    assert m.loc[("1000000004", "sale"), "match_status"] == "amount_difference"


def test_other_month_in_1c_is_one_row():
    m = statuses([act("alpha", "sale", "1000000005", debet=300.0, period="2026-02")],
                 [etm_op("alpha", 1, "purchase", "1000000005", -300.0, period="2026-01")])
    assert list(m["match_status"]) == ["period_mismatch"]


def test_etm_duplicate_is_summed_not_dropped():
    m = statuses([act("alpha", "sale", "1000000006", debet=500.0)],
                 [etm_op("alpha", 1, "purchase", "1000000006", -500.0),
                  etm_op("alpha", 2, "purchase", "1000000006", -500.0)])
    row = m.loc[("1000000006", "sale")]
    assert (row["match_status"], row["lines_etm"]) == ("amount_difference", 2)
    assert row["difference_1c_etm"] == pytest.approx(-500.0)


def test_refund_is_separate_from_sale():
    m = statuses([act("alpha", "sale", "1000000007", debet=900.0),
                  act("alpha", "refund", "1000000007", credit=600.0)],
                 [etm_op("alpha", 1, "purchase", "1000000007", -900.0),
                  etm_op("alpha", 2, "refund", "1000000007", 600.0)])
    assert set(m["match_status"]) == {"matched"}
    assert len(m) == 2


def test_registry_sign_and_void():
    m = statuses([act("alpha", "sale", "1000000008", debet=1000.0)],
                 [etm_op("alpha", 1, "purchase", "1000000008", -1000.0)],
                 pd.DataFrame([reg("alpha", "r:1", "1000000008", "sale", -1100.0),
                               reg("alpha", "r:2", "1000000008", "void", 100.0)]))
    row = m.loc[("1000000008", "sale")]
    assert row["amount_registry"] == pytest.approx(1000.0)
    assert row["registry_rows"] == 2
    assert row["difference_1c_registry"] == pytest.approx(0.0)


# ================================================================
# Баланс ETM: депозит один на субагента
# ================================================================

def _shared_deposit_etm():
    # Два договора, но остаток идёт одной цепочкой; остаток до первой операции −200
    return pd.DataFrame([
        etm_op("alpha", 1, "purchase", "1000000001", -1100.0, -1300.0, "AGR-1", date="2026-01-05 10:00"),
        etm_op("alpha", 2, "purchase", "1000000003", -300.0, -1600.0, "AGR-2", date="2026-01-06 10:00"),
        etm_op("alpha", 3, "purchase", "1000000004", -400.0, -2000.0, "AGR-1", date="2026-01-07 10:00"),
        etm_op("alpha", 4, "void", "1000000004", 400.0, -1600.0, "AGR-2", date="2026-01-08 10:00"),
        etm_op("alpha", 5, "payment", "", 250.0, -1350.0, "AGR-1", date="2026-01-09 10:00"),
        etm_op("alpha", 6, "purchase", "1000000009", -50.0, -1400.0, "AGR-2", period="2026-03",
               date="2026-03-02 10:00"),
    ])


def test_etm_balance_is_per_subagent_with_real_opening():
    b = etm_balances(_shared_deposit_etm(), ["2026-01", "2026-02", "2026-03"]).set_index("period")
    assert b.loc["2026-01", "etm_balance_start"] == pytest.approx(-200.0)
    assert b.loc["2026-01", "etm_balance_end"] == pytest.approx(-1350.0)
    # В феврале операций нет — остаток переносится
    assert b.loc["2026-02", "etm_balance_start"] == pytest.approx(-1350.0)
    assert b.loc["2026-02", "etm_balance_end"] == pytest.approx(-1350.0)
    assert b.loc["2026-03", "etm_balance_end"] == pytest.approx(-1400.0)


def test_p2_month_end_does_not_sum_agreements():
    me = etm_month_end_balance(_shared_deposit_etm()).set_index("period")
    assert me.loc["2026-01", "etm_balance_end"] == pytest.approx(-1350.0)


# ================================================================
# Мост: разница на начало + причины = разница на конец
# ================================================================

def test_bridge_explains_whole_gap():
    acts = pd.DataFrame([
        act("alpha", "sale", "1000000001", debet=1000.0, saldo_start=0.0, saldo_end=1000.0),
        act("alpha", "service_fee", "1000000001", debet=100.0, saldo_start=0.0, saldo_end=1000.0),
        act("alpha", "sale", "1000000002", debet=500.0, saldo_start=0.0, saldo_end=1000.0),     # нет в ETM
        act("alpha", "payment_bank", credit=600.0, saldo_start=0.0, saldo_end=1000.0),          # нет в ETM
    ])
    etm = _shared_deposit_etm()
    etm = etm[etm["period"] == "2026-01"]
    matches = match_operations(acts, etm, EMPTY_REG)
    payments = match_payment_rows(acts, etm)
    balance = build_balance(acts, etm)
    bridge = build_bridge(balance, acts, etm, matches, payments).set_index("period").loc["2026-01"]

    assert bridge["check_1c"] == pytest.approx(0.0)
    assert bridge["check_etm"] == pytest.approx(0.0)
    assert bridge["opening_difference"] == pytest.approx(-200.0)
    assert bridge["closing_difference"] == pytest.approx(-350.0)
    assert bridge["ops_only_1c"] == pytest.approx(500.0)
    assert bridge["ops_only_etm"] == pytest.approx(-300.0)
    assert bridge["ops_voided"] == pytest.approx(0.0)
    assert bridge["payments_only_1c"] == pytest.approx(-600.0)
    assert bridge["payments_only_etm"] == pytest.approx(250.0)
    assert bridge["unexplained"] == pytest.approx(0.0)


# ================================================================
# Реальные данные (если пайплайн уже запускали)
# ================================================================

@pytest.mark.skipif(not (INTERIM / "clean" / "p4_ready" / "acts_clean.csv").exists()
                    or not (INTERIM / "registry.parquet").exists(),
                    reason="нет interim: сначала запустите reconcile.py")
def test_real_data_smoke(tmp_path):
    res = run_p4(INTERIM / "clean", INTERIM / "registry.parquet", tmp_path)
    m, b = res["matches"], res["bridge"]
    assert not m.duplicated(["subagent_id", "ticket10", "op_type"]).any()
    assert (m["match_status"] != "unknown").all()
    # Без слияния «многие ко многим» расхождений сумм около тысячи, а не тысячи
    assert (m["match_status"] == "amount_difference").sum() < 2000
    assert b["check_etm"].abs().le(1).all()
    assert b["unexplained"].abs().le(1).all()
    for name in ("p4_operation_matches.csv", "p4_payment_matches.csv", "p4_balance_bridge.csv",
                 "p4_subagent_summary.csv"):
        assert (tmp_path / name).exists()
