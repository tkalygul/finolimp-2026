"""Тесты классификации ошибок и аномалий (P5).

Как запустить:
    pytest tests/test_classify_errors.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classify_errors import (
    ERROR_TYPES, amount_issue, build_ticket_ledger, classify_payments, classify_row,
    classify_tickets, detect_anomalies, etm_duplicate_ops, match_payments, run_p5, saldo_gaps,
)

REPO = Path(__file__).resolve().parent.parent
INTERIM = REPO / "interim"


# ================================================================
# Помощники: маленькие таблицы в формате P2/P3
# ================================================================

def act(sub, period, doc, tickets="", debet=0.0, credit=0.0, line_type="sale", pay_doc=None, date=None,
        saldo_start=0.0, saldo_end=0.0):
    return {
        "subagent": sub, "subagent_key": sub, "period": period, "date": date or f"{period}-10",
        "doc": doc, "tickets10": tickets, "debet": debet, "credit": credit,
        "debt_delta": round(debet - credit, 2), "line_type": line_type,
        "is_payment": line_type in ("payment_bank", "payment_cash"), "pay_doc": pay_doc,
        "saldo_start": saldo_start, "saldo_end": saldo_end,
    }


def etm_op(sub, txn, kind_en, tickets="", amount_kgs=0.0, period="2026-01", date=None, creator="bot",
           currency="KGS", pay_doc=None, comment=""):
    t13 = " ".join("000" + t for t in tickets.split()) if tickets else np.nan
    sign_ok = amount_kgs < 0 if kind_en == "purchase" else amount_kgs > 0
    return {
        "agent_key": sub, "txn_id": txn, "kind_en": kind_en, "kind": kind_en, "tickets10": tickets or np.nan,
        "tickets13": t13, "amount_kgs": amount_kgs, "debt_delta": -amount_kgs, "period": period,
        "date": date or f"{period}-10 12:00:00", "creator_en": creator, "currency": currency,
        "pay_doc": pay_doc, "comment": comment, "sign_ok": sign_ok, "chain_break": False, "chain_gap": 0.0,
    }


def reg(sub, row_id, ticket, op_type, amount_kgs, period="2026-01", status="ok", dup=False, who="a.user"):
    return {
        "party_type": "subagent", "subagent_id": sub, "ticket10": ticket, "op_type": op_type,
        "amount_kgs": amount_kgs, "ts": pd.Timestamp(f"{period}-10"), "row_id": row_id, "currency": "KGS",
        "is_dup_extra": dup, "parse_status": status, "created_by": who, "dup_group": None,
        "pax_count_mismatch": False,
    }


def ledger_row(**kw):
    """Строка леджера с разумными значениями по умолчанию."""
    row = {
        "subagent_id": "s", "ticket10": "1111111111", "grp": "sale",
        "amount_1c": np.nan, "n_1c": 0, "period_1c": np.nan,
        "amount_etm": np.nan, "n_etm": 0, "period_etm": np.nan, "kinds_etm": np.nan, "currency_etm": np.nan,
        "amount_reg": np.nan, "n_reg": 0, "period_reg": np.nan, "kinds_reg": np.nan, "currency_reg": np.nan,
        "reg_dup": False, "reg_parse_bad": False, "period": "2026-01", "act_exists": True,
    }
    if "amount_1c" in kw:
        row.update(n_1c=1, period_1c="2026-01")
    if "amount_etm" in kw:
        row.update(n_etm=1, period_etm="2026-01", kinds_etm="purchase", currency_etm="KGS")
    if "amount_reg" in kw:
        row.update(n_reg=1, period_reg="2026-01", kinds_reg="sale", currency_reg="KGS")
    row.update(kw)
    return row


def etype(**kw):
    return classify_row(ledger_row(**kw))[0]


# ================================================================
# Уточнение ошибки суммы
# ================================================================

@pytest.mark.parametrize("value, ref, n, cur, bad, expected", [
    (2000, 1000, 2, "KGS", False, "duplicate"),
    (2033.7, 1016.85, 1, "KGS", False, "other"),  # вдвое, но строка одна — не дубль
    (500, 1000, 1, "KGS", False, "half"),
    (-1000, 1000, 1, "KGS", False, "wrong_sign"),
    (0, 1000, 1, "KGS", True, "parse_error"),
    (0, 1000, 1, "KGS", False, "zero"),
    (19284, 91284, 1, "KGS", False, "round_typo"),
    (17193.29, 17175.46, 1, "KGS", False, "fx_rate"),  # 0.1% — округление/курс
    (11494.35, 10947.00, 1, "USD", False, "fx_rate"),  # 5% в валюте — курс
    (11494.35, 10947.00, 1, "KGS", False, "other"),    # 5% в сомах — не курс
])
def test_amount_issue(value, ref, n, cur, bad, expected):
    assert amount_issue(value, ref, n, cur, bad) == expected


# ================================================================
# Три источника: виноват тот, кто расходится с двумя другими
# ================================================================

def test_all_three_agree_is_ok():
    assert etype(amount_1c=100.0, amount_etm=100.4, amount_reg=99.8) == "ok"


def test_registry_differs_is_agent():
    t, owner, issue, *_ = classify_row(ledger_row(amount_1c=100.0, amount_etm=100.0, amount_reg=170.0))
    assert (t, owner) == ("registry_wrong_amount", "agent")


def test_etm_differs_is_bot():
    t, owner, *_ = classify_row(ledger_row(amount_1c=100.0, amount_etm=150.0, amount_reg=100.0))
    assert (t, owner) == ("etm_wrong_amount", "bot")


def test_1c_differs_is_1c():
    t, owner, issue, *_ = classify_row(ledger_row(amount_1c=19284.0, amount_etm=91284.0, amount_reg=91284.0))
    assert (t, owner, issue) == ("1c_wrong_amount", "1c", "round_typo")


def test_etm_double_purchase_is_bot_duplicate():
    assert etype(amount_1c=100.0, amount_etm=200.0, n_etm=2, amount_reg=100.0) == "etm_duplicate"


def test_registry_double_row_is_agent_duplicate():
    assert etype(amount_1c=100.0, amount_etm=100.0, amount_reg=200.0, n_reg=2) == "registry_duplicate"


def test_registry_unparsed_amount():
    assert etype(amount_1c=100.0, amount_etm=100.0, amount_reg=0.0, reg_parse_bad=True) == "registry_parse_error"


def test_all_differ_needs_manual_review():
    t, owner, *_ = classify_row(ledger_row(amount_1c=100.0, amount_etm=150.0, amount_reg=130.0))
    assert (t, owner) == ("all_differ", "unclear")


def test_same_amounts_different_month_is_1c_period():
    assert etype(amount_1c=100.0, amount_etm=100.0, amount_reg=100.0,
                 period_1c="2026-02", period_etm="2026-01") == "1c_wrong_period"


# ================================================================
# Войды
# ================================================================

def test_void_in_etm_and_registry_is_ok():
    assert etype(amount_etm=0.0, n_etm=2, kinds_etm="purchase+void",
                 amount_reg=0.0, n_reg=2, kinds_reg="sale+void") == "ok_voided"


def test_void_with_fee_left_in_registry_is_ok():
    assert etype(amount_etm=0.0, kinds_etm="purchase+void", amount_reg=500.0, kinds_reg="sale+void") == "ok_void_fee"


def test_void_not_executed_by_bot():
    t, owner, *_ = classify_row(ledger_row(amount_etm=10878.0, kinds_etm="purchase",
                                           amount_reg=1000.0, kinds_reg="sale+void"))
    assert (t, owner) == ("etm_void_not_executed", "bot")


def test_voided_ticket_posted_in_1c():
    assert etype(amount_1c=5000.0, amount_etm=0.0, kinds_etm="purchase+void",
                 amount_reg=0.0, kinds_reg="sale+void") == "1c_voided_posted"


def test_void_missing_in_registry():
    assert etype(amount_etm=0.0, kinds_etm="purchase+void", amount_reg=5000.0, kinds_reg="sale") == \
        "registry_void_missing"


# ================================================================
# Двух источников нет или один отсутствует
# ================================================================

def test_self_service_ok_and_mismatch():
    assert etype(amount_1c=100.0, amount_etm=100.0) == "ok_self_service"
    t, owner, _, _, conf, _ = classify_row(ledger_row(amount_1c=80.0, amount_etm=100.0))
    assert (t, owner, conf) == ("1c_wrong_amount", "1c", "medium")


def test_missing_in_1c_with_and_without_act():
    assert etype(amount_etm=100.0, amount_reg=100.0) == "1c_not_posted"
    assert etype(amount_etm=100.0, amount_reg=100.0, act_exists=False) == "1c_act_missing"
    assert etype(amount_etm=100.0) == "1c_not_posted"


def test_missing_in_etm_is_bot():
    assert etype(amount_1c=100.0, amount_reg=100.0) == "etm_not_executed"


def test_registry_only_and_1c_only():
    assert etype(amount_reg=100.0) == "etm_registry_not_executed"
    assert etype(amount_1c=100.0) == "1c_only"


def test_every_type_is_in_catalog():
    assert all(owner in ("bot", "agent", "1c", "unclear", "none") for owner, _ in ERROR_TYPES.values())


# ================================================================
# Леджер из маленьких таблиц
# ================================================================

def _small_sources():
    acts = pd.DataFrame([
        # продажа на 2 билета + сервисный сбор отдельной строкой
        act("alpha", "2026-01", "Реализация KV AAAAAA", "1000000001 1000000002", debet=2000.0),
        act("alpha", "2026-01", "Реализация сервисный сбор AAAAAA", "1000000001 1000000002", debet=200.0,
            line_type="service_fee"),
        # опечатка 1С в сумме
        act("alpha", "2026-01", "Реализация KV BBBBBB", "1000000003", debet=1900.0),
    ])
    etm = pd.DataFrame([
        etm_op("alpha", 1, "purchase", "1000000001 1000000002", -2200.0),
        etm_op("alpha", 2, "purchase", "1000000003", -1000.0),
        # выкуп и войд: в 1С ничего нет
        etm_op("alpha", 3, "purchase", "1000000004", -700.0),
        etm_op("alpha", 4, "void", "1000000004", 700.0),
    ])
    registry = pd.DataFrame([
        reg("alpha", "r:1", "1000000001", "sale", -1100.0),
        reg("alpha", "r:1", "1000000002", "sale", -1100.0),
        reg("alpha", "r:2", "1000000003", "sale", -1000.0),
        reg("alpha", "r:3", "1000000004", "sale", -800.0),
        reg("alpha", "r:4", "1000000004", "void", 700.0),
    ])
    return acts, etm, registry


def test_ledger_sums_fee_and_splits_tickets():
    acts, etm, registry = _small_sources()
    led = build_ticket_ledger(acts, etm, registry).set_index("ticket10")
    assert led.loc["1000000001", "amount_1c"] == pytest.approx(1100.0)
    assert led.loc["1000000001", "amount_etm"] == pytest.approx(1100.0)
    assert led.loc["1000000001", "amount_reg"] == pytest.approx(1100.0)
    assert led.loc["1000000004", "amount_etm"] == pytest.approx(0.0)
    assert pd.isna(led.loc["1000000004", "amount_1c"])


def test_classify_tickets_end_to_end():
    acts, etm, registry = _small_sources()
    out = classify_tickets(build_ticket_ledger(acts, etm, registry)).set_index("ticket10")
    assert out.loc["1000000001", "error_type"] == "ok"
    assert out.loc["1000000003", "error_type"] == "1c_wrong_amount"
    assert out.loc["1000000003", "amount_at_stake"] == pytest.approx(900.0)
    assert out.loc["1000000004", "error_type"] == "ok_void_fee"
    assert out["is_error"].sum() == 1


# ================================================================
# Оплаты
# ================================================================

def _payment_sources():
    acts = pd.DataFrame([
        act("alpha", "2026-01", "п/п ЦБ-С000001", credit=1000.0, line_type="payment_bank", pay_doc="ЦБ-С000001",
            date="2026-01-05"),
        act("alpha", "2026-01", "п/п ЦБ-С000002", credit=2000.0, line_type="payment_bank", pay_doc="ЦБ-С000002",
            date="2026-01-07"),
        act("alpha", "2026-01", "п/п ЦБ-С000003", credit=3000.0, line_type="payment_bank", pay_doc="ЦБ-С000003",
            date="2026-01-09"),
    ])
    etm = pd.DataFrame([
        etm_op("alpha", 10, "payment", amount_kgs=1000.0, date="2026-01-05 10:00:00", pay_doc="ЦБ-С000001"),
        etm_op("alpha", 11, "payment", amount_kgs=1000.0, date="2026-01-07 10:00:00"),   # повтор
        etm_op("alpha", 12, "payment", amount_kgs=2000.0, date="2026-01-08 10:00:00"),   # на день позже
        etm_op("alpha", 13, "payment", amount_kgs=5000.0, date="2026-01-20 10:00:00"),   # нет в 1С
        etm_op("alpha", 14, "payment", amount_kgs=700.0, period="2026-02", date="2026-02-03 10:00:00"),
    ])
    return acts, etm


def test_payment_matching_and_classes():
    acts, etm = _payment_sources()
    p = classify_payments(match_payments(acts, etm), acts)
    by_txn = p.dropna(subset=["txn_id"]).set_index("txn_id")["error_type"]
    assert by_txn[10] == "ok_payment"
    assert by_txn[11] == "etm_double_credit"
    assert by_txn[12] == "ok_payment"
    assert by_txn[13] == "1c_payment_not_posted"
    assert by_txn[14] == "1c_act_missing"
    lone_1c = p[p["match"] == "only_1c"]
    assert list(lone_1c["error_type"]) == ["etm_payment_not_credited"]
    assert lone_1c["amount_1c"].iloc[0] == 3000.0


# ================================================================
# Аномалии
# ================================================================

def test_etm_duplicate_ops_finds_second_copy():
    etm = pd.DataFrame([
        etm_op("alpha", 1, "purchase", "1000000001", -500.0, date="2026-01-05 10:00:00"),
        etm_op("alpha", 2, "purchase", "1000000001", -500.0, date="2026-01-06 10:00:00"),
        etm_op("alpha", 3, "purchase", "1000000002", -500.0),
    ])
    d = etm_duplicate_ops(etm)
    assert list(d["txn_id"]) == [2]
    assert d["hours_after_first"].iloc[0] == pytest.approx(24.0)


def test_saldo_gap_and_transfer_suspect():
    acts = pd.DataFrame([
        act("alpha", "2026-01", "x", saldo_start=0.0, saldo_end=100.0),
        act("alpha", "2026-02", "x", saldo_start=150.0, saldo_end=150.0),   # +50
        act("beta", "2026-01", "x", saldo_start=0.0, saldo_end=300.0),
        act("beta", "2026-02", "x", saldo_start=250.0, saldo_end=250.0),    # −50
        act("gamma", "2026-01", "x", saldo_start=0.0, saldo_end=10.0),
        act("gamma", "2026-03", "x", saldo_start=99.0, saldo_end=99.0),     # пропущен месяц — не разрыв
    ])
    g = saldo_gaps(acts)
    assert sorted(g["subagent_key"]) == ["alpha", "beta"]

    etm = pd.DataFrame([etm_op("alpha", 1, "purchase", "1000000001", -10.0)])
    registry = pd.DataFrame([reg("alpha", "r:1", "1000000001", "sale", -10.0)])
    payments = classify_payments(match_payments(acts, etm), acts)
    an = detect_anomalies(acts, etm, registry, payments)
    assert (an["anomaly_type"] == "1c_saldo_transfer_suspect").sum() == 2


# ================================================================
# Реальные данные (если пайплайн уже запускали)
# ================================================================

@pytest.mark.skipif(not (INTERIM / "clean" / "reconciliation_ready" / "acts_clean.csv").exists()
                    or not (INTERIM / "registry.parquet").exists(),
                    reason="нет interim: сначала запустите reconcile.py")
def test_real_data_smoke(tmp_path):
    res = run_p5(INTERIM / "clean", INTERIM / "registry.parquet", tmp_path)
    t = res["tickets"]
    assert t["error_type"].isin(ERROR_TYPES).all()
    # Большинство операций должно сходиться
    assert (t["is_error"] == 0).mean() > 0.8
    # Строк леджера без метки быть не должно
    assert t["error_owner"].notna().all()
    for name in ("p5_ticket_classified.csv", "p5_payments.csv", "p5_anomalies.csv", "p5_summary_subagent.csv"):
        assert (tmp_path / name).exists()
