"""P5: Типы ошибок и аномалии.

Для каждого расхождения между 1С, ETM и реестром агентов решаем, чья это ошибка
(бот ETM, агент в реестре или 1С), и ищем дубли, двойные оплаты и частые сбои.

Вход (результаты P2 и P3):
    interim/clean/p4_ready/acts_clean.csv   — акты 1С
    interim/clean/p4_ready/etm_clean.csv    — транзакции ETM
    interim/registry.parquet                — реестр агентов (одна строка на билет)

Выход (interim/p5):
    p5_ticket_classified.csv  — каждая пара «субагент + билет + продажа/возврат» с меткой ошибки
    p5_payments.csv           — сверка оплат 1С и ETM с меткой ошибки
    p5_anomalies.csv          — дубли, двойные зачисления, разрывы сальдо и прочие аномалии
    p5_summary_subagent.csv   — ошибки по субагентам, флаг частых сбоев
    p5_summary_type.csv       — ошибки по типам
    p5_summary_employee.csv   — ошибки агентов по сотрудникам реестра
    p5_error_types.csv        — справочник типов ошибок

Как запустить:
    python -m src.classify_errors --clean interim/clean --registry interim/registry.parquet --out interim/p5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize import subagent_key

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPORATE_CLIENTS_CSV = REPO_ROOT / "reference" / "corporate_clients.csv"

# Допуск сверки сумм, сом
TOL = 1.0
# Сколько дней может пройти между оплатой в 1С и зачислением в ETM
PAYMENT_MAX_DAYS = 5
# Окно, в котором повторное зачисление той же суммы считаем двойным
DOUBLE_CREDIT_DAYS = 7
# Частый сбой: доля ошибок выше среднего на столько сигм и не меньше стольких ошибок
FREQUENT_SIGMA = 2.0
FREQUENT_MIN_ERRORS = 5

OWNER_RU = {
    "bot": "бот ETM",
    "agent": "агент (реестр)",
    "1c": "1С",
    "unclear": "неясно, ручная проверка",
    "none": "нет ошибки",
}

# Справочник типов: код -> (виновник, описание)
ERROR_TYPES = {
    "ok": ("none", "Совпадает во всех источниках"),
    "ok_self_service": ("none", "Операция самого субагента (без реестра), 1С = ETM"),
    "ok_voided": ("none", "Войд: выкуп отменён в ETM, в 1С не проводится"),
    "ok_void_fee": ("none", "Войд: в реестре остался сервисный сбор, это ожидаемо"),
    "ok_payment": ("none", "Оплата есть и в 1С, и в ETM"),
    # бот ETM
    "etm_not_executed": ("bot", "Операция есть в реестре и в 1С, но её нет в ETM"),
    "etm_registry_not_executed": ("bot", "Операция есть только в реестре: бот её не исполнил "
                                         "(или агент ошибся в номере билета)"),
    "etm_void_not_executed": ("bot", "В реестре войд, а в ETM выкуп не отменён"),
    "etm_wrong_amount": ("bot", "Сумма в ETM отличается от 1С и реестра"),
    "etm_duplicate": ("bot", "Бот провёл операцию дважды"),
    "etm_wrong_sign": ("bot", "Операция в ETM с неверным знаком"),
    "etm_payment_not_credited": ("bot", "Оплата есть в 1С, но не зачислена в ETM"),
    "etm_double_credit": ("bot", "Оплата зачислена в ETM дважды"),
    # агент (реестр)
    "registry_wrong_amount": ("agent", "Сумма в реестре отличается от 1С и ETM"),
    "registry_duplicate": ("agent", "Операция внесена в реестр дважды"),
    "registry_parse_error": ("agent", "Сумма в реестре не распознана (например, номер билета в поле суммы)"),
    "registry_void_missing": ("agent", "В ETM войд, а в реестре он не отмечен"),
    # 1С
    "1c_not_posted": ("1c", "Операция есть в ETM, но не проведена в 1С"),
    "1c_act_missing": ("1c", "За этот месяц у субагента нет акта 1С"),
    "1c_wrong_amount": ("1c", "Сумма в 1С отличается от ETM (и реестра)"),
    "1c_duplicate": ("1c", "Операция проведена в 1С дважды"),
    "1c_voided_posted": ("1c", "Билет отменён (войд), а в 1С проведена продажа"),
    "1c_wrong_period": ("1c", "Операция проведена в 1С в другом месяце, чем в ETM"),
    "1c_only": ("1c", "Проводка в 1С без операции в ETM и в реестре"),
    "1c_payment_not_posted": ("1c", "Оплата зачислена в ETM, но не проведена в 1С"),
    # неясно
    "all_differ": ("unclear", "Все три источника дают разные суммы"),
    "sources_differ": ("unclear", "Два источника расходятся, третьего для арбитража нет"),
}

# Уточнение для ошибок в сумме
ISSUE_RU = {
    "duplicate": "сумма вдвое больше — дубль",
    "half": "сумма вдвое меньше — потеряна половина",
    "wrong_sign": "неверный знак",
    "parse_error": "сумма не распознана",
    "zero": "нулевая сумма",
    "round_typo": "круглая разница — похоже на опечатку",
    "fx_rate": "небольшая разница — курс, конвертация или округление",
    "other": "прочая разница",
}

FOREIGN_CURRENCIES = {"USD", "EUR", "RUB", "KZT"}


# ================================================================
# ЧАСТЬ 1. ЗАГРУЗКА
# ================================================================

def load_inputs(clean_dir="interim/clean", registry_path="interim/registry.parquet"):
    """Читает очищенные акты, ETM (P2) и реестр (P3)."""
    clean_dir = Path(clean_dir)
    ready = clean_dir / "p4_ready"
    acts = pd.read_csv(ready / "acts_clean.csv", dtype={"tickets10": str, "pay_doc": str},
                       encoding="utf-8-sig")
    etm = pd.read_csv(ready / "etm_clean.csv", dtype={"tickets10": str, "tickets13": str, "pay_doc": str},
                      encoding="utf-8-sig")
    registry = pd.read_parquet(registry_path)
    return acts, etm, registry


def _ticket_list(cell) -> list:
    """Билеты в ячейке: список (из памяти) или строка через пробел (из CSV)."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(t) for t in cell]
    if pd.isna(cell):
        return []
    return str(cell).split()


def _explode_tickets(df, col="tickets10"):
    """Одна строка на билет, сумма делится поровну между билетами строки."""
    d = df.copy()
    d["ticket10"] = d[col].map(_ticket_list)
    d["n_in_row"] = d["ticket10"].map(len)
    d = d[d["n_in_row"] > 0].explode("ticket10")
    d["ticket10"] = d["ticket10"].astype(str)
    return d


# ================================================================
# ЧАСТЬ 2. ЛЕДЖЕР ПО БИЛЕТАМ
# ================================================================

# Войд гасит выкуп, поэтому оба попадают в группу «sale»; возврат сверяется отдельно
_ACT_GROUP = {"sale": "sale", "service_fee": "sale", "refund": "refund"}
_ETM_GROUP = {"purchase": "sale", "void": "sale", "refund": "refund"}
_REG_GROUP = {"sale": "sale", "void": "sale", "refund": "refund"}


def _join_sorted(values) -> str:
    return "+".join(sorted(str(v) for v in values if pd.notna(v)))


def _foreign(values) -> str:
    """Иностранная валюта операции, если есть, иначе KGS."""
    for v in values:
        if pd.notna(v) and str(v) in FOREIGN_CURRENCIES:
            return str(v)
    return "KGS"


def build_ticket_ledger(acts, etm, registry) -> pd.DataFrame:
    """Сводит три источника в одну строку на (субагент, билет, группа операции).

    Суммы приведены к знаку «долг субагента»: продажа +, возврат −.
    Если источника нет, его сумма NaN (а не 0) — это важно для классификации.
    """
    keys = ["subagent_id", "ticket10", "grp"]

    a = acts[acts["line_type"].isin(_ACT_GROUP)].copy()
    a = _explode_tickets(a)
    a["subagent_id"] = a["subagent_key"]
    a["grp"] = a["line_type"].map(_ACT_GROUP)
    a["amt"] = a["debt_delta"] / a["n_in_row"]
    a["doc_line"] = np.arange(len(a))
    ga = a.groupby(keys).agg(
        amount_1c=("amt", "sum"), n_1c=("doc_line", "size"), period_1c=("period", "min"),
        doc_1c=("doc", lambda s: " | ".join(sorted(set(map(str, s))))[:200]),
        lines_1c=("line_type", _join_sorted),
    )

    e = etm[etm["kind_en"].isin(_ETM_GROUP)].copy()
    e = _explode_tickets(e)
    e["subagent_id"] = e["agent_key"]
    e["grp"] = e["kind_en"].map(_ETM_GROUP)
    e["amt"] = e["debt_delta"] / e["n_in_row"]
    ge = e.groupby(keys).agg(
        amount_etm=("amt", "sum"), n_etm=("txn_id", "nunique"), period_etm=("period", "min"),
        kinds_etm=("kind_en", _join_sorted), creator_etm=("creator_en", lambda s: _join_sorted(set(s))),
        currency_etm=("currency", _foreign), txn_ids=("txn_id", lambda s: " ".join(str(int(x)) for x in sorted(s))),
    )

    r = registry[(registry["party_type"] == "subagent") & registry["subagent_id"].notna()
                 & registry["ticket10"].notna()].copy()
    r["subagent_id"] = r["subagent_id"].astype(str)
    r["ticket10"] = r["ticket10"].astype(str)
    r["grp"] = r["op_type"].map(_REG_GROUP)
    r["amt"] = -r["amount_kgs"].astype(float)
    r["period_r"] = pd.to_datetime(r["ts"]).dt.to_period("M").astype(str)
    r["is_dup_extra"] = r["is_dup_extra"].fillna(False).astype(bool)
    r["parse_bad"] = r["parse_status"].astype(str).ne("ok")
    gr = r.groupby(keys).agg(
        amount_reg=("amt", "sum"), n_reg=("row_id", "nunique"), period_reg=("period_r", "min"),
        kinds_reg=("op_type", _join_sorted), currency_reg=("currency", _foreign),
        reg_dup=("is_dup_extra", "any"), reg_parse_bad=("parse_bad", "any"),
        parse_status=("parse_status", lambda s: _join_sorted(set(s))),
        created_by=("created_by", lambda s: _join_sorted(set(s))),
    )

    ledger = ga.join(ge, how="outer").join(gr, how="outer").reset_index()
    ledger["period"] = ledger["period_etm"].fillna(ledger["period_reg"]).fillna(ledger["period_1c"])

    act_periods = set(zip(acts["subagent_key"], acts["period"]))
    ledger["act_exists"] = [(k, p) in act_periods for k, p in zip(ledger["subagent_id"], ledger["period"])]
    for c in ("reg_dup", "reg_parse_bad"):
        ledger[c] = ledger[c].astype("boolean").fillna(False).astype(bool)
    for c in ("n_1c", "n_etm", "n_reg"):
        ledger[c] = ledger[c].fillna(0).astype(int)
    return ledger


# ================================================================
# ЧАСТЬ 3. КЛАССИФИКАЦИЯ БИЛЕТОВ
# ================================================================

def _eq(x, y) -> bool:
    return abs(x - y) <= TOL


def amount_issue(value, reference, n_rows=1, currency="KGS", parse_bad=False) -> str:
    """Уточняет, как именно сумма value отличается от верной суммы reference."""
    if abs(value) <= TOL:
        return "parse_error" if parse_bad else "zero"
    if abs(reference) <= TOL:
        return "other"
    ratio = value / reference
    if abs(ratio - 2) <= 0.01 and n_rows >= 2:
        return "duplicate"
    if abs(ratio - 0.5) <= 0.01:
        return "half"
    if abs(ratio + 1) <= 0.01:
        return "wrong_sign"
    diff = abs(value - reference)
    if diff >= 100 and abs(diff - round(diff, -2)) <= TOL:
        return "round_typo"
    # До 1% — курс/округление в любой валюте, до 20% — только если операция была в валюте
    if abs(ratio - 1) <= 0.01 or (abs(ratio - 1) <= 0.2 and currency in FOREIGN_CURRENCIES):
        return "fx_rate"
    return "other"


def _result(error_type, reason, confidence, at_stake=0.0, issue=""):
    owner = ERROR_TYPES[error_type][0]
    return error_type, owner, issue, reason, confidence, round(float(at_stake), 2)


def _wrong_amount(side, value, reference, row, confidence="high"):
    """Ошибка суммы у одного источника: тип по виновнику, уточнение по сумме."""
    if side == "bot":
        issue = amount_issue(value, reference, row["n_etm"], row["currency_etm"] or "KGS")
        error_type = "etm_duplicate" if issue == "duplicate" else "etm_wrong_amount"
        src = "ETM"
    elif side == "agent":
        currency = row["currency_reg"] if isinstance(row["currency_reg"], str) else "KGS"
        issue = amount_issue(value, reference, row["n_reg"], currency, row["reg_parse_bad"])
        if issue == "duplicate" or (issue == "half" and row["reg_dup"]):
            error_type = "registry_duplicate"
        elif issue in ("parse_error", "zero") and row["reg_parse_bad"]:
            error_type = "registry_parse_error"
        else:
            error_type = "registry_wrong_amount"
        src = "реестре"
    else:
        currency = row["currency_etm"] if isinstance(row["currency_etm"], str) else "KGS"
        issue = amount_issue(value, reference, row["n_1c"], currency)
        error_type = "1c_duplicate" if issue == "duplicate" else "1c_wrong_amount"
        src = "1С"
    reason = f"в {src} {value:,.2f}, в остальных {reference:,.2f} ({ISSUE_RU[issue]})"
    return _result(error_type, reason, confidence, value - reference, issue)


def classify_row(row) -> tuple:
    """Решает, чья ошибка в строке леджера.

    Возвращает (error_type, error_owner, issue, reason, confidence, amount_at_stake).
    Главное правило: если источников три, виноват тот, кто расходится с двумя другими.
    """
    h1, he, hr = pd.notna(row["amount_1c"]), pd.notna(row["amount_etm"]), pd.notna(row["amount_reg"])
    c1 = row["amount_1c"] if h1 else 0.0
    ce = row["amount_etm"] if he else 0.0
    cr = row["amount_reg"] if hr else 0.0
    kinds_e = row["kinds_etm"] if isinstance(row["kinds_etm"], str) else ""
    kinds_r = row["kinds_reg"] if isinstance(row["kinds_reg"], str) else ""
    void_e, void_r = "void" in kinds_e, "void" in kinds_r

    # --- Войды: выкуп + войд в ETM дают ноль, в 1С ничего быть не должно
    if row["grp"] == "sale" and (void_e or void_r):
        if void_r and not void_e and he and abs(ce) > TOL:
            return _result("etm_void_not_executed",
                           f"в реестре войд, а в ETM выкуп {ce:,.2f} не отменён", "high", ce)
        if void_e and abs(ce) <= TOL:
            if h1 and abs(c1) > TOL:
                return _result("1c_voided_posted",
                               f"в ETM билет отменён, а в 1С проведено {c1:,.2f}", "high", c1)
            if hr and not void_r and abs(cr) > TOL:
                return _result("registry_void_missing",
                               f"в ETM войд, а в реестре только продажа {cr:,.2f}", "high", cr)
            if hr and abs(cr) > TOL:
                return _result("ok_void_fee", f"в реестре остался сбор {cr:,.2f}", "high")
            return _result("ok_voided", "выкуп и войд в ETM", "high")

    # --- Все три источника есть
    if h1 and he and hr:
        e1e, e1r, eer = _eq(c1, ce), _eq(c1, cr), _eq(ce, cr)
        if e1e and e1r:
            if pd.notna(row["period_1c"]) and pd.notna(row["period_etm"]) and row["period_1c"] != row["period_etm"]:
                return _result("1c_wrong_period",
                               f"в 1С {row['period_1c']}, в ETM {row['period_etm']}", "high", c1)
            return _result("ok", "", "high")
        if e1e:
            return _wrong_amount("agent", cr, c1, row)
        if e1r:
            return _wrong_amount("bot", ce, c1, row)
        if eer:
            return _wrong_amount("1c", c1, ce, row)
        return _result("all_differ", f"1С {c1:,.2f}, ETM {ce:,.2f}, реестр {cr:,.2f}", "low",
                       max(abs(c1 - ce), abs(c1 - cr), abs(ce - cr)))

    # --- Только 1С и ETM: операция самого субагента, ETM — первичная система
    if h1 and he:
        if _eq(c1, ce):
            if row["period_1c"] != row["period_etm"]:
                return _result("1c_wrong_period",
                               f"в 1С {row['period_1c']}, в ETM {row['period_etm']}", "medium", c1)
            return _result("ok_self_service", "", "high")
        return _wrong_amount("1c", c1, ce, row, confidence="medium")

    # --- Есть в ETM, нет в 1С
    if he and not h1:
        if abs(ce) <= TOL and not hr:
            return _result("ok_voided", "операция в ETM обнулена", "high")
        if hr and not _eq(ce, cr):
            return _result("sources_differ",
                           f"нет в 1С; ETM {ce:,.2f}, реестр {cr:,.2f}", "low", ce)
        if not row["act_exists"]:
            return _result("1c_act_missing", f"нет акта 1С за {row['period']}, в ETM {ce:,.2f}",
                           "high", ce)
        return _result("1c_not_posted", f"в ETM {ce:,.2f}, в 1С нет", "high" if hr else "medium", ce)

    # --- Есть в 1С и реестре, нет в ETM
    if h1 and hr:
        if _eq(c1, cr):
            return _result("etm_not_executed", f"в 1С и реестре {c1:,.2f}, в ETM нет", "high", c1)
        return _result("sources_differ", f"нет в ETM; 1С {c1:,.2f}, реестр {cr:,.2f}", "low", c1)

    # --- Только в реестре
    if hr:
        if abs(cr) <= TOL and row["reg_parse_bad"]:
            return _result("registry_parse_error", "сумма в реестре не распознана, в ETM и 1С нет",
                           "medium")
        return _result("etm_registry_not_executed", f"в реестре {cr:,.2f}, в ETM и 1С нет", "low", cr)

    # --- Только в 1С
    return _result("1c_only", f"в 1С {c1:,.2f}, в ETM и реестре нет", "medium", c1)


def classify_tickets(ledger) -> pd.DataFrame:
    """Добавляет к леджеру метки ошибки (по строке)."""
    out = ledger.copy()
    cols = ["error_type", "error_owner", "issue", "reason", "confidence", "amount_at_stake"]
    if out.empty:
        for c in cols:
            out[c] = pd.Series(dtype=object)
        out["is_error"] = pd.Series(dtype=int)
        return out
    res = [classify_row(r) for r in out.to_dict("records")]
    out[cols] = pd.DataFrame(res, index=out.index, columns=cols)
    out["error_owner_ru"] = out["error_owner"].map(OWNER_RU)
    out["error_type_ru"] = out["error_type"].map(lambda t: ERROR_TYPES[t][1])
    out["is_error"] = (~out["error_owner"].eq("none")).astype(int)
    return out


# ================================================================
# ЧАСТЬ 4. ОПЛАТЫ
# ================================================================

def match_payments(acts, etm, max_days=PAYMENT_MAX_DAYS) -> pd.DataFrame:
    """Сопоставляет оплаты 1С и ETM: тот же субагент, та же сумма, даты рядом.

    Номер п/п (ЦБ-С...) повторяется у разных субагентов и месяцев, а в ETM он есть
    не всегда, поэтому он только приоритет, а не ключ.
    """
    pa = acts[acts["is_payment"].astype(bool)].copy()
    pa["amount"] = (pa["credit"].fillna(0) - pa["debet"].fillna(0)).round(2)
    pa["date_1c"] = pd.to_datetime(pa["date"]).dt.normalize()
    pa["i"] = np.arange(len(pa))

    pe = etm[etm["kind_en"] == "payment"].copy()
    pe["amount"] = pe["amount_kgs"].round(2)
    pe["date_etm"] = pd.to_datetime(pe["date"])
    pe["j"] = np.arange(len(pe))

    cand = pa[["subagent_key", "amount", "date_1c", "pay_doc", "i"]].merge(
        pe[["agent_key", "amount", "date_etm", "pay_doc", "j"]].rename(columns={"agent_key": "subagent_key"}),
        on=["subagent_key", "amount"], suffixes=("_1c", "_etm"))
    cand["days"] = (cand["date_etm"].dt.normalize() - cand["date_1c"]).dt.days
    cand = cand[cand["days"].abs() <= max_days].copy()
    cand["doc_match"] = (cand["pay_doc_1c"] == cand["pay_doc_etm"]).astype(int)
    cand["abs_days"] = cand["days"].abs()
    cand = cand.sort_values(["doc_match", "abs_days", "date_etm"], ascending=[False, True, True])

    used_i, used_j, pairs = set(), set(), []
    for i, j in zip(cand["i"], cand["j"]):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))

    pair = pd.DataFrame(pairs, columns=["i", "j"])
    cols_1c = ["i", "subagent_key", "period", "date_1c", "doc", "pay_doc", "amount"]
    cols_e = ["j", "agent_key", "period", "date_etm", "txn_id", "comment", "pay_doc", "amount", "creator_en"]
    left = pa[cols_1c].rename(columns={"subagent_key": "subagent_id", "period": "period_1c",
                                       "pay_doc": "pay_doc_1c", "amount": "amount_1c"})
    right = pe[cols_e].rename(columns={"agent_key": "subagent_id_etm", "period": "period_etm",
                                       "pay_doc": "pay_doc_etm", "amount": "amount_etm"})
    both = pair.merge(left, on="i").merge(right, on="j")
    only_1c = left[~left["i"].isin(used_i)]
    only_etm = right[~right["j"].isin(used_j)].copy()
    only_etm["subagent_id"] = only_etm["subagent_id_etm"]

    res = pd.concat([both.assign(match="both"), only_1c.assign(match="only_1c"),
                     only_etm.assign(match="only_etm")], ignore_index=True)
    res["subagent_id"] = res["subagent_id"].fillna(res["subagent_id_etm"])
    res["period"] = res["period_etm"].fillna(res["period_1c"])
    return res.drop(columns=["i", "j", "subagent_id_etm"])


def classify_payments(payments, acts) -> pd.DataFrame:
    """Размечает оплаты: двойное зачисление, не зачислено, не проведено в 1С."""
    p = payments.copy()
    act_periods = set(zip(acts["subagent_key"], acts["period"]))
    p["date_etm"] = pd.to_datetime(p["date_etm"])

    # Двойное зачисление: та же сумма у того же субагента уже сопоставлена с 1С рядом по дате
    matched = p[p["match"] == "both"][["subagent_id", "amount_etm", "date_etm", "txn_id"]]
    lone = p[p["match"] == "only_etm"][["subagent_id", "amount_etm", "date_etm", "txn_id"]].reset_index()
    sib = lone.merge(matched, on=["subagent_id", "amount_etm"], suffixes=("", "_orig"))
    sib = sib[(sib["date_etm"] - sib["date_etm_orig"]).abs() <= pd.Timedelta(days=DOUBLE_CREDIT_DAYS)]
    double_orig = sib.drop_duplicates("index").set_index("index")["txn_id_orig"]

    types, reasons, conf, stake = [], [], [], []
    for idx, r in p.iterrows():
        if r["match"] == "both":
            if r["period_1c"] != r["period_etm"]:
                t, why, c = "1c_wrong_period", f"в 1С {r['period_1c']}, в ETM {r['period_etm']}", "high"
            else:
                t, why, c = "ok_payment", "", "high"
            amt = 0.0 if t == "ok_payment" else r["amount_1c"]
        elif r["match"] == "only_1c":
            t, why, c = "etm_payment_not_credited", f"{r['doc']} на {r['amount_1c']:,.2f} нет в ETM", "medium"
            amt = r["amount_1c"]
        elif idx in double_orig.index:
            t = "etm_double_credit"
            why = f"повтор зачисления {r['amount_etm']:,.2f}, первое — txn {int(double_orig[idx])}"
            c, amt = "high", r["amount_etm"]
        elif (r["subagent_id"], r["period"]) not in act_periods:
            t, why, c = "1c_act_missing", f"нет акта 1С за {r['period']}", "high"
            amt = r["amount_etm"]
        else:
            t, why, c = "1c_payment_not_posted", f"зачисление {r['amount_etm']:,.2f} в ETM, в 1С нет", "medium"
            amt = r["amount_etm"]
        types.append(t)
        reasons.append(why)
        conf.append(c)
        stake.append(round(float(amt), 2))

    p["error_type"] = types
    p["error_owner"] = p["error_type"].map(lambda t: ERROR_TYPES[t][0])
    p["error_owner_ru"] = p["error_owner"].map(OWNER_RU)
    p["error_type_ru"] = p["error_type"].map(lambda t: ERROR_TYPES[t][1])
    p["reason"] = reasons
    p["confidence"] = conf
    p["amount_at_stake"] = stake
    p["is_error"] = (~p["error_owner"].eq("none")).astype(int)
    return p


# ================================================================
# ЧАСТЬ 5. АНОМАЛИИ
# ================================================================

ANOMALY_COLUMNS = ["anomaly_type", "error_owner", "severity", "subagent_id", "period", "ref",
                   "amount_kgs", "detail"]


def _anom(df, anomaly_type, owner, severity, **cols) -> pd.DataFrame:
    out = pd.DataFrame({"anomaly_type": anomaly_type, "error_owner": owner, "severity": severity},
                       index=df.index)
    for k, v in cols.items():
        out[k] = v
    return out.reindex(columns=ANOMALY_COLUMNS)


def etm_duplicate_ops(etm) -> pd.DataFrame:
    """Повторные операции бота: те же билеты, тот же вид, та же сумма."""
    e = etm[(etm["kind_en"] != "payment") & etm["tickets13"].notna()].copy()
    e["date"] = pd.to_datetime(e["date"])
    k = ["agent_key", "tickets13", "kind_en", "amount_kgs"]
    e = e.sort_values(k + ["date"])
    e["copy_no"] = e.groupby(k).cumcount()
    e["first_txn"] = e.groupby(k)["txn_id"].transform("first")
    e["hours_after_first"] = (e["date"] - e.groupby(k)["date"].transform("first")).dt.total_seconds() / 3600
    return e[e["copy_no"] > 0]


def saldo_gaps(acts) -> pd.DataFrame:
    """Сальдо на начало месяца не равно сальдо на конец прошлого акта."""
    h = (acts.groupby(["subagent_key", "subagent", "period"], as_index=False)
         .agg(saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first")))
    h = h.sort_values(["subagent_key", "period"])
    h["prev_period"] = h.groupby("subagent_key")["period"].shift()
    h["prev_saldo_end"] = h.groupby("subagent_key")["saldo_end"].shift()
    exp_prev = (pd.PeriodIndex(h["period"], freq="M") - 1).astype(str)
    h["consecutive"] = h["prev_period"].eq(pd.Series(exp_prev, index=h.index))
    h["gap"] = (h["saldo_start"] - h["prev_saldo_end"]).round(2)
    return h[h["prev_period"].notna() & h["consecutive"] & (h["gap"].abs() > 0.01)]


def corporate_in_sources(acts, etm, path=CORPORATE_CLIENTS_CSV) -> pd.DataFrame:
    """Корпоративные клиенты не должны встречаться в ETM и 1С как субагенты."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["name", "source"])
    corp = pd.read_csv(path, encoding="utf-8-sig")
    corp["key"] = corp["name"].map(subagent_key)
    keys = set(corp["key"])
    hits = []
    for src, col in (("1С", acts["subagent_key"]), ("ETM", etm["agent_key"])):
        for k in sorted(set(col) & keys):
            hits.append({"name": corp.loc[corp["key"] == k, "name"].iloc[0], "key": k, "source": src})
    return pd.DataFrame(hits, columns=["name", "key", "source"])


def detect_anomalies(acts, etm, registry, payments) -> pd.DataFrame:
    """Собирает все аномалии в одну длинную таблицу."""
    parts = []

    d = etm_duplicate_ops(etm)
    parts.append(_anom(d, "etm_duplicate_op", "bot", "high", subagent_id=d["agent_key"], period=d["period"],
                       ref=d["txn_id"].astype("Int64").astype(str), amount_kgs=d["amount_kgs"],
                       detail="повтор txn " + d["first_txn"].astype("Int64").astype(str) + " через "
                              + d["hours_after_first"].round(1).astype(str) + " ч, билеты " + d["tickets13"]))

    dc = payments[payments["error_type"] == "etm_double_credit"]
    parts.append(_anom(dc, "etm_double_credit", "bot", "high", subagent_id=dc["subagent_id"],
                       period=dc["period"], ref=dc["txn_id"].astype("Int64").astype(str),
                       amount_kgs=dc["amount_etm"], detail=dc["reason"]))

    ws = etm[~etm["sign_ok"].astype(bool)]
    parts.append(_anom(ws, "etm_wrong_sign", "bot", "high", subagent_id=ws["agent_key"], period=ws["period"],
                       ref=ws["txn_id"].astype("Int64").astype(str), amount_kgs=ws["amount_kgs"],
                       detail=ws["kind"] + " с суммой " + ws["amount_kgs"].astype(str)))

    if "chain_break" in etm.columns:
        cb = etm[etm["chain_break"].astype(bool)]
        parts.append(_anom(cb, "etm_balance_chain_break", "bot", "low", subagent_id=cb["agent_key"],
                           period=cb["period"], ref=cb["txn_id"].astype("Int64").astype(str),
                           amount_kgs=cb["chain_gap"],
                           detail="остаток после операции не сходится с предыдущим (возможен порядок внутри дня)"))

    reg = registry
    rd = reg[reg["is_dup_extra"].fillna(False).astype(bool)]
    parts.append(_anom(rd, "registry_duplicate_row", "agent", "medium", subagent_id=rd["subagent_id"],
                       period=pd.to_datetime(rd["ts"]).dt.to_period("M").astype(str), ref=rd["row_id"],
                       amount_kgs=rd["amount_kgs"], detail="лишняя копия строки, группа " + rd["dup_group"].astype(str)))

    rp = reg[reg["parse_status"].astype(str).ne("ok")]
    parts.append(_anom(rp, "registry_parse_problem", "agent", "medium", subagent_id=rp["subagent_id"],
                       period=pd.to_datetime(rp["ts"]).dt.to_period("M").astype(str), ref=rp["row_id"],
                       amount_kgs=rp["amount_kgs"], detail=rp["parse_status"].astype(str)))

    pm = reg[reg["pax_count_mismatch"].fillna(False).astype(bool)]
    parts.append(_anom(pm, "registry_pax_mismatch", "agent", "low", subagent_id=pm["subagent_id"],
                       period=pd.to_datetime(pm["ts"]).dt.to_period("M").astype(str), ref=pm["row_id"],
                       amount_kgs=pm["amount_kgs"], detail="пассажиров не столько, сколько билетов"))

    # Нет акта 1С за месяц, в котором у субагента есть операции в ETM
    act_periods = set(zip(acts["subagent_key"], acts["period"]))
    em = etm.groupby(["agent_key", "period"], as_index=False).agg(n=("txn_id", "size"),
                                                                 turnover=("debt_delta", "sum"))
    em = em[[(k, p) not in act_periods for k, p in zip(em["agent_key"], em["period"])]]
    parts.append(_anom(em, "1c_act_missing", "1c", "high", subagent_id=em["agent_key"], period=em["period"],
                       ref="", amount_kgs=em["turnover"].round(2),
                       detail="нет акта 1С, операций в ETM: " + em["n"].astype(str)))

    g = saldo_gaps(acts)
    parts.append(_anom(g, "1c_saldo_carryover", "1c", "high", subagent_id=g["subagent_key"], period=g["period"],
                       ref=g["prev_period"], amount_kgs=g["gap"],
                       detail="сальдо на начало " + g["saldo_start"].astype(str) + " ≠ сальдо на конец "
                              + g["prev_saldo_end"].astype(str)))

    # Одна и та же сумма разрыва у разных субагентов — похоже, сальдо перенесли не тому субагенту
    g2 = g.assign(abs_gap=g["gap"].abs())
    rep = g2[g2.groupby("abs_gap")["subagent_key"].transform("nunique") > 1]
    if not rep.empty:
        peers = rep.groupby("abs_gap")["subagent"].agg(lambda s: ", ".join(sorted(set(s))))
        parts.append(_anom(rep, "1c_saldo_transfer_suspect", "1c", "high", subagent_id=rep["subagent_key"],
                           period=rep["period"], ref=rep["abs_gap"].astype(str), amount_kgs=rep["gap"],
                           detail="такой же разрыв у: " + rep["abs_gap"].map(peers)))

    corp = corporate_in_sources(acts, etm)
    parts.append(_anom(corp, "corporate_as_subagent", "1c", "medium", subagent_id=corp["key"], period="",
                       ref=corp["source"], amount_kgs=np.nan,
                       detail="корпоративный клиент " + corp["name"] + " найден в " + corp["source"]))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["error_owner_ru"] = out["error_owner"].map(OWNER_RU)
    return out


# ================================================================
# ЧАСТЬ 6. СВОДКИ И ЧАСТЫЕ СБОИ
# ================================================================

def _frequent(rate, n_errors) -> pd.Series:
    """Частый сбой: доля ошибок заметно выше средней и ошибок не единицы."""
    mu, sd = rate.mean(), rate.std(ddof=0)
    return (rate > mu + FREQUENT_SIGMA * sd) & (n_errors >= FREQUENT_MIN_ERRORS)


def summarize(tickets, payments, anomalies) -> dict:
    """Сводки по субагентам, типам и сотрудникам реестра."""
    ops = pd.concat([
        tickets[["subagent_id", "error_type", "error_owner", "amount_at_stake", "is_error"]],
        payments[["subagent_id", "error_type", "error_owner", "amount_at_stake", "is_error"]],
    ], ignore_index=True)

    by_sub = ops.groupby("subagent_id").agg(operations=("is_error", "size"), errors=("is_error", "sum"))
    for owner in ("bot", "agent", "1c", "unclear"):
        sel = ops[ops["error_owner"] == owner]
        by_sub[f"errors_{owner}"] = sel.groupby("subagent_id").size()
        by_sub[f"amount_{owner}"] = sel.groupby("subagent_id")["amount_at_stake"].apply(lambda s: s.abs().sum())
    if not anomalies.empty:
        by_sub["anomalies"] = anomalies.groupby("subagent_id").size()
    by_sub = by_sub.fillna(0)
    by_sub["error_rate"] = (by_sub["errors"] / by_sub["operations"]).round(4)
    by_sub["frequent_failures"] = _frequent(by_sub["error_rate"], by_sub["errors"])
    by_sub = by_sub.reset_index().sort_values("errors", ascending=False)

    errs = ops[ops["is_error"] == 1]
    by_type = errs.groupby(["error_owner", "error_type"]).agg(
        count=("is_error", "size"), amount=("amount_at_stake", lambda s: s.abs().sum().round(2)),
        subagents=("subagent_id", "nunique")).reset_index()
    by_type["error_owner_ru"] = by_type["error_owner"].map(OWNER_RU)
    by_type["error_type_ru"] = by_type["error_type"].map(lambda t: ERROR_TYPES[t][1])
    by_type = by_type.sort_values("count", ascending=False)

    # Ошибки агентов по сотрудникам, которые вносили строки в реестр
    t = tickets[tickets["created_by"].notna() & tickets["created_by"].astype(str).ne("")].copy()
    t["agent_error"] = t["error_owner"].eq("agent").astype(int)
    by_emp = t.groupby("created_by").agg(rows=("agent_error", "size"), agent_errors=("agent_error", "sum"))
    by_emp["error_rate"] = (by_emp["agent_errors"] / by_emp["rows"]).round(4)
    by_emp["frequent_failures"] = _frequent(by_emp["error_rate"], by_emp["agent_errors"])
    by_emp = by_emp.reset_index().sort_values("agent_errors", ascending=False)

    return {"subagent": by_sub, "type": by_type, "employee": by_emp}


def error_types_table() -> pd.DataFrame:
    return pd.DataFrame([(code, owner, OWNER_RU[owner], text) for code, (owner, text) in ERROR_TYPES.items()],
                        columns=["error_type", "error_owner", "error_owner_ru", "description"])


# ================================================================
# ЧАСТЬ 7. ЗАПУСК
# ================================================================

def run_p5(clean_dir="interim/clean", registry_path="interim/registry.parquet", out_dir="interim/p5") -> dict:
    """Полный шаг P5: классификация, оплаты, аномалии, сводки, запись файлов."""
    acts, etm, registry = load_inputs(clean_dir, registry_path)

    ledger = build_ticket_ledger(acts, etm, registry)
    tickets = classify_tickets(ledger)
    payments = classify_payments(match_payments(acts, etm), acts)
    anomalies = detect_anomalies(acts, etm, registry, payments)
    summary = summarize(tickets, payments, anomalies)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(df, name):
        df.to_csv(out / name, index=False, encoding="utf-8-sig")

    save(tickets, "p5_ticket_classified.csv")
    save(payments, "p5_payments.csv")
    save(anomalies, "p5_anomalies.csv")
    save(summary["subagent"], "p5_summary_subagent.csv")
    save(summary["type"], "p5_summary_type.csv")
    save(summary["employee"], "p5_summary_employee.csv")
    save(error_types_table(), "p5_error_types.csv")

    return {"tickets": tickets, "payments": payments, "anomalies": anomalies, **summary}


def format_report(result) -> str:
    """Короткий текстовый отчёт для консоли."""
    t, p, a = result["tickets"], result["payments"], result["anomalies"]
    lines = [
        f"Билетных операций: {len(t)}, с ошибкой: {int(t['is_error'].sum())}",
        f"Оплат: {len(p)}, с ошибкой: {int(p['is_error'].sum())}",
        f"Аномалий: {len(a)}",
        "",
        "Ошибки по виновнику (билеты + оплаты):",
    ]
    both = pd.concat([t[["error_owner", "is_error"]], p[["error_owner", "is_error"]]])
    for owner, n in both[both["is_error"] == 1]["error_owner"].value_counts().items():
        lines.append(f"  {OWNER_RU[owner]:<26} {n}")
    lines += ["", "Ошибки по типам:"]
    for r in result["type"].itertuples():
        lines.append(f"  {r.error_type:<28} {r.count:>6}  {r.amount:>16,.2f}  {r.error_type_ru}")
    if not a.empty:
        lines += ["", "Аномалии:"]
        for k, n in a["anomaly_type"].value_counts().items():
            lines.append(f"  {k:<28} {n}")
    freq = result["subagent"]
    freq = freq[freq["frequent_failures"]]
    if not freq.empty:
        lines += ["", "Частые сбои (субагенты): " + ", ".join(freq["subagent_id"])]
    emp = result["employee"]
    emp = emp[emp["frequent_failures"]]
    if not emp.empty:
        lines += ["Частые сбои (сотрудники реестра): " + ", ".join(emp["created_by"])]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P5: типы ошибок и аномалии")
    parser.add_argument("--clean", default="interim/clean", help="Папка с результатами cleaning.py")
    parser.add_argument("--registry", default="interim/registry.parquet", help="Реестр после P3")
    parser.add_argument("--out", default="interim/p5", help="Куда сохранить результаты")
    args = parser.parse_args(argv)
    try:
        result = run_p5(args.clean, args.registry, args.out)
    except FileNotFoundError as exc:
        print(f"[P5] Нет входного файла: {exc}. Сначала запустите reconcile.py (шаги P2 и P3).")
        return 1
    print(format_report(result))
    print(f"\n[P5] Результаты: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
