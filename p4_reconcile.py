"""P4: Сопоставление операций и баланс.

Связывает операции 1С, ETM и реестра по билетам и для каждого субагента
показывает, из чего складывается разница между сальдо 1С и балансом ETM.

Вход (результаты P2 и P3):
    interim/clean/p4_ready/acts_clean.csv   — акты 1С
    interim/clean/p4_ready/etm_clean.csv    — транзакции ETM
    interim/registry.parquet                — реестр агентов

Выход (interim/p4):
    p4_operation_matches.csv  — одна строка на «субагент + билет + продажа/возврат»
    p4_payment_matches.csv    — сверка оплат 1С и ETM
    p4_balance_bridge.csv     — мост: разница на начало + причины = разница на конец
    p4_subagent_summary.csv   — сводка по субагентам

Как запустить:
    python p4_reconcile.py --clean interim/clean --registry interim/registry.parquet --out interim/p4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.classify_errors import match_payments

# Допуск сверки сумм, сом
TOL = 1.0

# Ключ операции. Месяц в ключ не входит: операция, проведённая в 1С в другом месяце,
# должна быть одной строкой «period_mismatch», а не двумя (only_1c + only_etm).
KEYS = ["subagent_id", "ticket10", "op_type"]

# Сервисный сбор 1С — часть продажи. Войд ETM и реестра гасит выкуп, поэтому тоже «sale»:
# выкуп + войд в сумме дают ноль.
ACT_OP = {"sale": "sale", "service_fee": "sale", "refund": "refund"}
ETM_OP = {"purchase": "sale", "void": "sale", "refund": "refund"}
REG_OP = {"sale": "sale", "void": "sale", "refund": "refund"}

OP_STATUSES = ["matched", "period_mismatch", "amount_difference", "voided", "only_1c", "only_etm"]
PAY_STATUSES = ["matched", "period_mismatch", "only_1c", "only_etm"]


# ================================================================
# ЧАСТЬ 1. ЗАГРУЗКА
# ================================================================

def load_p4_data(clean_dir="interim/clean", registry_path="interim/registry.parquet"):
    """Читает очищенные акты, ETM (P2) и реестр (P3)."""
    ready = Path(clean_dir) / "p4_ready"
    acts = pd.read_csv(ready / "acts_clean.csv", dtype={"tickets10": str, "pay_doc": str},
                       encoding="utf-8-sig")
    etm = pd.read_csv(ready / "etm_clean.csv", dtype={"tickets10": str, "tickets13": str, "pay_doc": str},
                      encoding="utf-8-sig")
    registry = pd.read_parquet(registry_path)

    acts["date"] = pd.to_datetime(acts["date"], errors="coerce")
    etm["date"] = pd.to_datetime(etm["date"], errors="coerce")
    registry["ts"] = pd.to_datetime(registry["ts"], errors="coerce")
    registry["period"] = registry["ts"].dt.to_period("M").astype(str)
    return acts, etm, registry


def _explode_tickets(df, amount_col):
    """Одна строка на билет, сумма строки делится поровну между её билетами."""
    d = df[df["tickets10"].fillna("").astype(str).str.strip().ne("")].copy()
    d["ticket10"] = d["tickets10"].astype(str).str.split()
    d["n_in_row"] = d["ticket10"].str.len()
    d = d.explode("ticket10")
    d[amount_col] = d["debt_delta"] / d["n_in_row"]
    return d


def prepare_acts(acts) -> pd.DataFrame:
    """Строки 1С по билетам (до сложения). Сумма — долг субагента: продажа +, возврат −."""
    a = _explode_tickets(acts[acts["line_type"].isin(ACT_OP)], "amount_1c")
    a["subagent_id"] = a["subagent_key"]
    a["op_type"] = a["line_type"].map(ACT_OP)
    return a[["subagent_id", "period", "ticket10", "op_type", "amount_1c", "line_type", "pnr", "date", "doc"]]


def prepare_etm(etm) -> pd.DataFrame:
    """Строки ETM по билетам (до сложения), в тех же знаках, что и 1С."""
    e = _explode_tickets(etm[etm["kind_en"].isin(ETM_OP)], "amount_etm")
    e["subagent_id"] = e["agent_key"]
    e["op_type"] = e["kind_en"].map(ETM_OP)
    return e[["subagent_id", "period", "ticket10", "op_type", "amount_etm", "kind_en", "pnr", "date", "txn_id"]]


# ================================================================
# ЧАСТЬ 2. СОПОСТАВЛЕНИЕ ОПЕРАЦИЙ
# ================================================================

def _join_unique(values) -> str:
    return " | ".join(sorted({str(v) for v in values if pd.notna(v)}))


def match_operations(acts, etm, registry) -> pd.DataFrame:
    """Одна строка на (субагент, билет, продажа/возврат) со суммами 1С, ETM и реестра.

    Строки каждого источника сначала складываются по ключу: продажа и сервисный сбор 1С,
    выкуп и войд ETM. Иначе merge идёт «многие ко многим» и одна строка ETM сравнивается
    с каждой строкой 1С отдельно.
    """
    a = prepare_acts(acts)
    e = prepare_etm(etm)

    ga = a.groupby(KEYS).agg(
        amount_1c=("amount_1c", "sum"), lines_1c=("amount_1c", "size"),
        period_1c=("period", "min"), date_1c=("date", "min"),
        doc_1c=("doc", _join_unique), pnr_1c=("pnr", "first"),
    )
    ge = e.groupby(KEYS).agg(
        amount_etm=("amount_etm", "sum"), lines_etm=("txn_id", "nunique"),
        period_etm=("period", "min"), date_etm=("date", "min"),
        kinds_etm=("kind_en", lambda s: "+".join(sorted(s))),
        txn_ids=("txn_id", lambda s: " ".join(str(int(x)) for x in sorted(s))),
        pnr_etm=("pnr", "first"),
    )
    result = ga.join(ge, how="outer").reset_index()
    for c in ("lines_1c", "lines_etm"):
        result[c] = result[c].fillna(0).astype(int)

    has_1c, has_etm = result["lines_1c"] > 0, result["lines_etm"] > 0
    result["difference_1c_etm"] = (result["amount_1c"].fillna(0) - result["amount_etm"].fillna(0)).round(2)
    same_amount = result["difference_1c_etm"].abs() <= TOL
    result["match_status"] = np.select(
        [has_1c & has_etm & same_amount & result["period_1c"].eq(result["period_etm"]),
         has_1c & has_etm & same_amount,
         has_1c & has_etm,
         has_etm & result["amount_etm"].abs().le(TOL),   # выкуп + войд = 0, в 1С ничего нет
         has_1c,
         has_etm],
        OP_STATUSES, default="unknown")
    result["period"] = result["period_etm"].fillna(result["period_1c"])

    r = registry[(registry["party_type"] == "subagent") & registry["subagent_id"].notna()
                 & registry["ticket10"].notna()].copy()
    r["subagent_id"] = r["subagent_id"].astype(str)
    r["ticket10"] = r["ticket10"].astype(str)
    r["op_type"] = r["op_type"].map(REG_OP)
    gr = r.groupby(KEYS, as_index=False).agg(
        amount_registry=("amount_kgs", "sum"), registry_rows=("row_id", "nunique"),
        party_match=("party_match", "first"), parse_status=("parse_status", _join_unique),
    )
    # В реестре знак обратный: продажа −, возврат +
    gr["amount_registry"] = -gr["amount_registry"]

    result = result.merge(gr, on=KEYS, how="left")
    result["registry_rows"] = result["registry_rows"].fillna(0).astype(int)
    # Разница с реестром — только там, где есть оба источника
    result["difference_1c_registry"] = (result["amount_1c"] - result["amount_registry"]).round(2)
    result["difference_etm_registry"] = (result["amount_etm"] - result["amount_registry"]).round(2)
    return result


def match_payment_rows(acts, etm) -> pd.DataFrame:
    """Сверка оплат: субагент + сумма + дата ±5 дней (та же логика, что в P5)."""
    p = match_payments(acts, etm)
    both = p["match"].eq("both")
    p["match_status"] = np.select(
        [both & p["period_1c"].eq(p["period_etm"]), both, p["match"].eq("only_1c")],
        ["matched", "period_mismatch", "only_1c"], default="only_etm")
    return p


# ================================================================
# ЧАСТЬ 3. БАЛАНС И МОСТ
# ================================================================

def etm_balances(etm, periods) -> pd.DataFrame:
    """Баланс ETM на начало и конец каждого месяца по субагенту.

    Депозит у субагента один: при нескольких договорах balance_after идёт сквозной цепочкой
    по всем договорам. Остаток на начало первого месяца = остаток после первой операции
    минус её сумма. В месяц без операций остаток переносится с прошлого месяца.
    """
    e = etm.sort_values(["agent_key", "date", "txn_id"])
    first = e.groupby("agent_key").first()
    opening = first["balance_after"] - first["amount_kgs"]

    end = e.groupby(["agent_key", "period"])["balance_after"].last().unstack().reindex(columns=periods)
    end = end.ffill(axis=1).apply(lambda col: col.fillna(opening))
    start = end.shift(1, axis=1)
    start[periods[0]] = opening

    turnover = e.groupby(["agent_key", "period"])["amount_kgs"].sum()
    ops = e.groupby(["agent_key", "period"]).size()
    out = pd.DataFrame({"etm_balance_start": start.stack(), "etm_balance_end": end.stack()})
    out.index.names = ["subagent_id", "period"]
    out["turnover_etm"] = turnover.reindex(out.index).fillna(0)
    out["etm_ops"] = ops.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


def build_balance(acts, etm) -> pd.DataFrame:
    """Сальдо 1С и баланс ETM по субагенту и месяцу, с внутренней проверкой каждого источника.

    Знаки: сальдо 1С > 0 — долг субагента, баланс ETM < 0 — долг субагента.
    Поэтому разница между системами = сальдо 1С + баланс ETM.
    """
    act_balance = (acts.groupby(["subagent_key", "period"], as_index=False)
                   .agg(saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first"),
                        turnover_1c=("debt_delta", "sum"))
                   .rename(columns={"subagent_key": "subagent_id"}))
    act_balance["act_exists"] = True

    periods = sorted(set(acts["period"].astype(str)) | set(etm["period"].astype(str)))
    balance = act_balance.merge(etm_balances(etm, periods), on=["subagent_id", "period"], how="outer")
    balance["act_exists"] = balance["act_exists"].fillna(False).astype(bool)
    for c in ("saldo_start", "saldo_end", "turnover_1c", "etm_balance_start", "etm_balance_end",
              "turnover_etm", "etm_ops"):
        balance[c] = balance[c].fillna(0)
    # Месяцы, где нет ни акта, ни операций ETM, в мост не берём
    balance = balance[balance["act_exists"] | balance["etm_ops"].gt(0)].copy()

    balance["check_1c"] = (balance["saldo_start"] + balance["turnover_1c"] - balance["saldo_end"]).round(2)
    balance["check_etm"] = (balance["etm_balance_start"] + balance["turnover_etm"]
                            - balance["etm_balance_end"]).round(2)
    balance["opening_difference"] = (balance["saldo_start"] + balance["etm_balance_start"]).round(2)
    balance["turnover_difference"] = (balance["turnover_1c"] + balance["turnover_etm"]).round(2)
    balance["closing_difference"] = (balance["saldo_end"] + balance["etm_balance_end"]).round(2)
    return balance.sort_values(["subagent_id", "period"]).reset_index(drop=True)


def build_bridge(balance, acts, etm, matches, payments) -> pd.DataFrame:
    """Мост: разница на начало + вклад каждой причины = разница на конец месяца.

    Вклад причины в месяц = сумма строк 1С этого месяца с этой причиной минус сумма строк ETM
    этого месяца с этой причиной (в знаках долга субагента). unexplained — то, что не объяснено
    ни одной причиной; при полной сверке он равен нулю.
    """
    status = matches[KEYS + ["match_status"]]
    a = prepare_acts(acts).merge(status, on=KEYS, how="left")
    e = prepare_etm(etm).merge(status, on=KEYS, how="left")
    parts = [
        a.assign(col="ops_" + a["match_status"], value=a["amount_1c"]),
        e.assign(col="ops_" + e["match_status"], value=-e["amount_etm"]),
    ]
    # Оплата уменьшает долг: в 1С это −сумма, в ETM баланс растёт на +сумму
    p1 = payments[payments["amount_1c"].notna()]
    pe = payments[payments["amount_etm"].notna()]
    parts += [
        pd.DataFrame({"subagent_id": p1["subagent_id"], "period": p1["period_1c"],
                      "col": "payments_" + p1["match_status"], "value": -p1["amount_1c"]}),
        pd.DataFrame({"subagent_id": pe["subagent_id"], "period": pe["period_etm"],
                      "col": "payments_" + pe["match_status"], "value": pe["amount_etm"]}),
    ]
    long = pd.concat([p[["subagent_id", "period", "col", "value"]] for p in parts], ignore_index=True)
    wide = long.pivot_table(index=["subagent_id", "period"], columns="col", values="value", aggfunc="sum")
    cols = [f"ops_{s}" for s in OP_STATUSES] + [f"payments_{s}" for s in PAY_STATUSES]
    wide = wide.reindex(columns=cols).fillna(0).round(2).reset_index()

    bridge = balance.merge(wide, on=["subagent_id", "period"], how="left")
    bridge[cols] = bridge[cols].fillna(0)
    bridge["explained"] = bridge[cols].sum(axis=1).round(2)
    bridge["unexplained"] = (bridge["closing_difference"] - bridge["opening_difference"]
                             - bridge["explained"]).round(2)
    return bridge


# ================================================================
# ЧАСТЬ 4. СВОДКА
# ================================================================

def build_summary(matches, payments, bridge) -> pd.DataFrame:
    """Сводка по субагентам. Ошибки сумм и пропущенные операции — в разных колонках."""
    m = matches
    summary = m.groupby("subagent_id").agg(operations=("match_status", "size"))
    for s in OP_STATUSES:
        summary[s] = m[m["match_status"] == s].groupby("subagent_id").size()
    summary["amount_difference_sum"] = (m[m["match_status"] == "amount_difference"]
                                        .groupby("subagent_id")["difference_1c_etm"].sum())
    summary["only_1c_amount"] = m[m["match_status"] == "only_1c"].groupby("subagent_id")["amount_1c"].sum()
    summary["only_etm_amount"] = m[m["match_status"] == "only_etm"].groupby("subagent_id")["amount_etm"].sum()

    summary["payments"] = payments.groupby("subagent_id").size()
    for s in ("only_1c", "only_etm"):
        summary[f"payments_{s}"] = payments[payments["match_status"] == s].groupby("subagent_id").size()

    last = bridge.sort_values("period").groupby("subagent_id").tail(1).set_index("subagent_id")
    summary["closing_difference_last"] = last["closing_difference"]
    summary["unexplained_total"] = bridge.groupby("subagent_id")["unexplained"].sum()

    summary = summary.fillna(0)
    count_cols = OP_STATUSES + ["payments", "payments_only_1c", "payments_only_etm"]
    summary[count_cols] = summary[count_cols].astype(int)
    # Сошедшимися считаем и войды: выкуп отменён, в 1С его быть не должно
    summary["matched_share"] = ((summary["matched"] + summary["voided"]) / summary["operations"]).round(4)
    return summary.round(2).reset_index().sort_values("subagent_id")


# ================================================================
# ЧАСТЬ 5. ЗАПУСК
# ================================================================

def run_p4(clean_dir="interim/clean", registry_path="interim/registry.parquet", out_dir="interim/p4") -> dict:
    """Полный шаг P4: сопоставление, баланс, мост, сводка, запись файлов."""
    acts, etm, registry = load_p4_data(clean_dir, registry_path)

    matches = match_operations(acts, etm, registry)
    payments = match_payment_rows(acts, etm)
    balance = build_balance(acts, etm)
    bridge = build_bridge(balance, acts, etm, matches, payments)
    summary = build_summary(matches, payments, bridge)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(df, name):
        df.to_csv(out / name, index=False, encoding="utf-8-sig")

    save(matches, "p4_operation_matches.csv")
    save(payments, "p4_payment_matches.csv")
    save(bridge, "p4_balance_bridge.csv")
    save(summary, "p4_subagent_summary.csv")
    return {"matches": matches, "payments": payments, "bridge": bridge, "summary": summary}


def format_report(result) -> str:
    """Короткий текстовый отчёт для консоли."""
    m, p, b = result["matches"], result["payments"], result["bridge"]
    lines = [f"Операций (субагент + билет + продажа/возврат): {len(m)}"]
    for s, n in m["match_status"].value_counts().items():
        lines.append(f"  {s:<20} {n}")
    lines.append(f"Оплат: {len(p)}")
    for s, n in p["match_status"].value_counts().items():
        lines.append(f"  {s:<20} {n}")
    lines += [
        f"Мост: субагенто-месяцев {len(b)}, "
        f"с расхождением на конец {int(b['closing_difference'].abs().gt(TOL).sum())}, "
        f"необъяснённых {int(b['unexplained'].abs().gt(TOL).sum())}",
        f"Внутренние проверки: 1С не сходится {int(b['check_1c'].abs().gt(TOL).sum())}, "
        f"ETM не сходится {int(b['check_etm'].abs().gt(TOL).sum())}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P4: сопоставление операций и баланс")
    parser.add_argument("--clean", default="interim/clean", help="Папка с результатами cleaning.py")
    parser.add_argument("--registry", default="interim/registry.parquet", help="Реестр после P3")
    parser.add_argument("--out", default="interim/p4", help="Куда сохранить результаты")
    args = parser.parse_args(argv)
    try:
        result = run_p4(args.clean, args.registry, args.out)
    except FileNotFoundError as exc:
        print(f"[P4] Нет входного файла: {exc}. Сначала запустите reconcile.py (шаги P2 и P3).")
        return 1
    print(format_report(result))
    print(f"\n[P4] Результаты: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
