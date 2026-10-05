"""P3: Загрузка и разбор реестра агентов.

Как запустить:
    python -m src.load_registry --data <папка_с_выгрузками> --out <папка_результата>
Пример:
    python -m src.load_registry --data data_2_final --out interim
"""
import argparse
import fnmatch
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.normalize import ParseError, normalize_pax, parse_pay, subagent_key, tickets10
from src.schema import ValidationError, validate

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPORATE_CLIENTS_CSV = REPO_ROOT / "reference" / "corporate_clients.csv"

SOURCE_COLUMNS = [
    "date", "employee", "kind", "party", "tickets", "pax", "pnr", "airline",
    "route", "pay_cell", "rate_usd", "rate_eur", "rate_rub", "rate_kzt",
]

OUTPUT_COLUMNS = [
    "source", "row_id", "subagent_id", "ts", "op_type", "ticket10",
    "n_tickets_in_row", "pnr", "passenger", "airline", "route",
    "amount_orig", "currency", "fee_orig", "fee_currency", "penalty_pct",
    "amount_kgs", "created_by", "parse_status",
    "ticket_seq", "party_raw", "party_key", "party_type", "party_match",
    "fx_rate", "fee_fx_rate", "amount_row_kgs", "dup_group", "is_dup_extra",
    "pax_count_mismatch",
]

KIND_TO_OP = {"продажа": "sale", "возврат": "refund", "войд": "void"}
RATE_COLUMN = {"USD": "rate_usd", "EUR": "rate_eur", "RUB": "rate_rub", "KZT": "rate_kzt"}

# Обрезанное имя ищем среди субагентов ETM, только если ключ не короче порога
MIN_PREFIX_LEN = 6


class InputFileError(Exception):
    """Нет нужного файла или их несколько."""


@dataclass
class LoadResult:
    registry: pd.DataFrame
    rejects: pd.DataFrame
    non_subagents: pd.DataFrame
    stats: dict = field(default_factory=dict)


# ================================================================
# ЧАСТЬ 1. ФАЙЛЫ И СПРАВОЧНИКИ
# ================================================================

def find_input_file(data_dir, pattern) -> Path:
    """Ищет ровно один файл по маске без учёта регистра."""
    folder = Path(data_dir)
    if not folder.is_dir():
        raise InputFileError(f"Папка с данными не найдена: {folder}")
    found = sorted(p for p in folder.iterdir()
                   if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern.lower()))
    if not found:
        raise InputFileError(f"В {folder} нет файла по маске {pattern}")
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise InputFileError(f"В {folder} несколько файлов по маске {pattern}: {names}")
    return found[0]


def read_csv_text(path) -> pd.DataFrame:
    """Читает все поля как текст, пустые строки оставляет, чтобы не сбить нумерацию."""
    options = dict(dtype=str, keep_default_na=False, skip_blank_lines=False)
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **options)
    except UnicodeDecodeError:
        # Файл мог быть пересохранён в Excel
        print(f"[Registry] Внимание: {path.name} не в UTF-8, читаю как cp1251", file=sys.stderr)
        return pd.read_csv(path, encoding="cp1251", **options)


def load_corporate_keys(path=CORPORATE_CLIENTS_CSV) -> set:
    """Загружает ключи корпоративных клиентов из справочника."""
    ref = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return {subagent_key(n) for n in ref["name"] if n.strip()}


def load_etm_reference(data_dir) -> tuple:
    """Берёт из ETM ключи субагентов и карту билет -> субагенты."""
    etm = read_csv_text(find_input_file(data_dir, "etm*.csv"))
    missing = [c for c in ("agent", "kind", "tickets") if c not in etm.columns]
    if missing:
        raise ValidationError(f"[Registry] В ETM нет колонок {missing}")
    keys = {subagent_key(a) for a in etm["agent"] if a.strip()}
    ticket_agents = {}
    for agent, cell in zip(etm["agent"], etm["tickets"]):
        if not cell.strip():
            continue
        try:
            tickets = tickets10(cell)
        except ParseError:
            continue
        for t in tickets:
            ticket_agents.setdefault(t, set()).add(subagent_key(agent))
    return keys, ticket_agents


# ================================================================
# ЧАСТЬ 2. КОНТРАГЕНТЫ
# ================================================================

def resolve_party(key, tickets, etm_keys, corporate_keys, ticket_agents) -> tuple:
    """Определяет тип контрагента: (party_type, subagent_id, party_match, reason)."""
    if not key:
        return "unresolved", None, "none", "party_empty"
    if key in etm_keys:
        return "subagent", key, "exact", None
    if key in corporate_keys:
        return "corporate", None, "corporate", None
    candidates = sorted(k for k in etm_keys if k.startswith(key)) if len(key) >= MIN_PREFIX_LEN else []
    if len(candidates) == 1:
        return "subagent", candidates[0], "prefix", None
    if len(candidates) > 1:
        # Несколько подходящих, выбираем по билету из ETM
        by_ticket = set()
        for t in tickets:
            by_ticket |= ticket_agents.get(t, set()) & set(candidates)
        if len(by_ticket) == 1:
            return "subagent", by_ticket.pop(), "prefix_ticket", None
        return "unresolved", None, "none", "party_ambiguous"
    return "unresolved", None, "none", "party_unknown"


# ================================================================
# ЧАСТЬ 3. РАЗБОР СТРОКИ
# ================================================================

def _rate(row, currency) -> float:
    """Берёт курс валюты из той же строки реестра."""
    if currency == "KGS":
        return 1.0
    text = row[RATE_COLUMN[currency]].strip().replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        raise ParseError("bad_rate", f"{RATE_COLUMN[currency]}={row[RATE_COLUMN[currency]]!r}")
    if not value > 0:
        raise ParseError("bad_rate", f"{RATE_COLUMN[currency]}={value}")
    return value


def parse_row(row, etm_keys, corporate_keys, ticket_agents) -> dict:
    """Разбирает одну строку реестра, проблемы складывает в reasons."""
    reasons = []
    details = []

    def problem(err):
        reasons.append(err.reason)
        if err.detail:
            details.append(f"{err.reason}: {err.detail}")

    if not any(str(v).strip() for v in row.values()):
        return {"reasons": ["empty_row"], "details": [], "tickets": []}

    ts = None
    try:
        ts = datetime.strptime(row["date"].strip(), "%d.%m.%Y")
    except ValueError:
        problem(ParseError("bad_date", row["date"]))

    op = KIND_TO_OP.get(row["kind"].strip().lower())
    if op is None:
        problem(ParseError("bad_kind", row["kind"]))

    tickets = []
    try:
        tickets = tickets10(row["tickets"])
    except ParseError as err:
        problem(err)

    pay = None
    try:
        pay = parse_pay(row["pay_cell"])
    except ParseError as err:
        problem(err)

    amount_row_kgs = fx = fee_fx = None
    if pay is not None:
        try:
            fx = _rate(row, pay.currency)
            fee_fx = _rate(row, pay.fee_currency) if pay.fee_currency else None
            gross = abs(pay.amount) * fx + (pay.fee or 0.0) * (fee_fx or 0.0)
            if op is not None:
                # Как в ETM: продажа уменьшает депозит, возврат и войд увеличивают
                # Штраф у возвратов уже вычтен в сумме реестра
                amount_row_kgs = -gross if op == "sale" else gross
        except ParseError as err:
            problem(err)
        # Проверяем знак суммы и лишние сбор/штраф
        if op == "sale" and pay.amount <= 0 or op in ("refund", "void") and pay.amount >= 0:
            problem(ParseError("sign_mismatch", f"{row['kind']}: {row['pay_cell']}"))
        if op in ("refund", "void") and pay.fee is not None:
            problem(ParseError("fee_on_non_sale", row["pay_cell"]))
        if op == "sale" and pay.penalty_pct is not None:
            problem(ParseError("penalty_on_sale", row["pay_cell"]))

    key = subagent_key(row["party"])
    party_type, subagent_id, party_match, party_reason = resolve_party(
        key, tickets, etm_keys, corporate_keys, ticket_agents)
    if party_reason:
        problem(ParseError(party_reason, row["party"]))

    pax = normalize_pax(row["pax"])
    return {
        "reasons": reasons, "details": details, "tickets": tickets, "ts": ts, "op": op,
        "pay": pay, "fx": fx, "fee_fx": fee_fx, "amount_row_kgs": amount_row_kgs,
        "party_key": key, "party_type": party_type, "subagent_id": subagent_id,
        "party_match": party_match, "pax": pax,
        "pax_count_mismatch": bool(tickets) and len(pax) != len(tickets),
    }


# ================================================================
# ЧАСТЬ 4. ЗАГРУЗКА И СОХРАНЕНИЕ
# ================================================================

def load_registry(data_dir, etm_keys=None, ticket_agents=None, corporate_keys=None) -> LoadResult:
    """Загружает реестр и возвращает таблицу «одна строка на билет»."""
    path = find_input_file(data_dir, "registry*.csv")
    src = read_csv_text(path)
    validate(src, dataset_type="registry")
    missing = [c for c in SOURCE_COLUMNS if c not in src.columns]
    if missing:
        raise ValidationError(f"[Registry] В реестре {path.name} нет колонок {missing}")

    if etm_keys is None or ticket_agents is None:
        ref_keys, ref_tickets = load_etm_reference(data_dir)
        etm_keys = ref_keys if etm_keys is None else etm_keys
        ticket_agents = ref_tickets if ticket_agents is None else ticket_agents
    if corporate_keys is None:
        corporate_keys = load_corporate_keys()

    # Номер строки равен номеру записи в CSV, заголовок это строка 1
    row_ids = [f"registry:{i + 2}" for i in range(len(src))]

    # Полные дубли не удаляем, а помечаем
    raw = src[SOURCE_COLUMNS]
    dup_mask = raw.duplicated(keep=False)
    first_of_group = {}
    dup_group, is_dup_extra = [], []
    for rid, values, is_dup in zip(row_ids, raw.itertuples(index=False, name=None), dup_mask):
        if not is_dup:
            dup_group.append(None)
            is_dup_extra.append(False)
            continue
        first = first_of_group.setdefault(values, rid)
        dup_group.append(first)
        is_dup_extra.append(first != rid)

    out_rows, reject_rows, corporate_rows = [], [], []
    reason_counts = Counter()
    for i, row in enumerate(src[SOURCE_COLUMNS].to_dict("records")):
        rid = row_ids[i]
        p = parse_row(row, etm_keys, corporate_keys, ticket_agents)
        status = "ok" if not p["reasons"] else ";".join(p["reasons"])
        reason_counts.update(p["reasons"] or ["ok"])
        if p["reasons"]:
            reject_rows.append({"row_id": rid, "parse_status": status,
                                "parse_detail": " | ".join(p["details"]), **row})
        if p.get("party_type") == "corporate":
            corporate_rows.append({"row_id": rid, "party_key": p["party_key"],
                                   "amount_row_kgs": p["amount_row_kgs"], **row})
        pay = p.get("pay")
        n = len(p["tickets"])
        for seq, ticket in enumerate(p["tickets"], start=1):
            out_rows.append({
                "source": "registry", "row_id": rid, "subagent_id": p["subagent_id"],
                "ts": p["ts"], "op_type": p["op"], "ticket10": ticket,
                "n_tickets_in_row": n, "pnr": row["pnr"].strip().upper(),
                "passenger": ", ".join(p["pax"]), "airline": row["airline"].strip().upper(),
                "route": row["route"].strip().upper(),
                "amount_orig": pay.amount if pay else None,
                "currency": pay.currency if pay else None,
                "fee_orig": pay.fee if pay else None,
                "fee_currency": pay.fee_currency if pay else None,
                "penalty_pct": pay.penalty_pct if pay else None,
                # Сумму строки делим поровну между билетами, как это делает бот
                "amount_kgs": p["amount_row_kgs"] / n if p["amount_row_kgs"] is not None else None,
                "created_by": row["employee"].strip(), "parse_status": status,
                "ticket_seq": seq, "party_raw": row["party"], "party_key": p["party_key"],
                "party_type": p["party_type"], "party_match": p["party_match"],
                "fx_rate": p["fx"], "fee_fx_rate": p["fee_fx"],
                "amount_row_kgs": p["amount_row_kgs"], "dup_group": dup_group[i],
                "is_dup_extra": is_dup_extra[i], "pax_count_mismatch": p["pax_count_mismatch"],
            })

    registry = _typed(pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS))
    rejects = pd.DataFrame(reject_rows, columns=["row_id", "parse_status", "parse_detail", *SOURCE_COLUMNS])
    non_subagents = pd.DataFrame(corporate_rows,
                                 columns=["row_id", "party_key", "amount_row_kgs", *SOURCE_COLUMNS])

    n_src = len(src)
    n_ok = n_src - len(rejects)
    stats = {
        "file": path.name,
        "source_rows": n_src,
        "output_rows": len(registry),
        "tickets_total": int(registry.groupby("row_id").size().sum()) if len(registry) else 0,
        "ok_rows": n_ok,
        "ok_share": n_ok / n_src if n_src else 0.0,
        "reject_rows": len(rejects),
        "reasons": dict(sorted(reason_counts.items())),
        "corporate_rows": len(non_subagents),
        "prefix_keys": sorted(registry.loc[registry.party_match.isin(["prefix", "prefix_ticket"]),
                                           "party_key"].unique()),
        "prefix_ticket_rows": int(registry.loc[registry.party_match == "prefix_ticket", "row_id"].nunique()),
        "dup_rows": int(sum(g is not None for g in dup_group)),
        "dup_extra_rows": int(sum(is_dup_extra)),
        "pax_mismatch_rows": int(registry.loc[registry.pax_count_mismatch, "row_id"].nunique()),
        "rows_without_tickets": int(sum(1 for r in reject_rows
                                        if any(x in r["parse_status"] for x in ("no_tickets", "bad_ticket_format", "duplicate_ticket_in_cell", "empty_row")))),
    }
    return LoadResult(registry, rejects, non_subagents, stats)


def _typed(df) -> pd.DataFrame:
    """Задаёт явные типы колонок, чтобы parquet не зависел от версии pandas."""
    text = ["source", "row_id", "subagent_id", "op_type", "ticket10", "pnr", "passenger",
            "airline", "route", "currency", "fee_currency", "created_by", "parse_status",
            "party_raw", "party_key", "party_type", "party_match", "dup_group"]
    floats = ["amount_orig", "fee_orig", "penalty_pct", "amount_kgs", "fx_rate",
              "fee_fx_rate", "amount_row_kgs"]
    df[text] = df[text].astype("string")
    df[floats] = df[floats].astype("float64")
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
    df[["n_tickets_in_row", "ticket_seq"]] = df[["n_tickets_in_row", "ticket_seq"]].astype("int64")
    df[["is_dup_extra", "pax_count_mismatch"]] = df[["is_dup_extra", "pax_count_mismatch"]].astype("bool")
    return df


def write_outputs(result, out_dir) -> dict:
    """Сохраняет таблицы и отчёт в папку результата."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "registry": out / "registry.parquet",
        "rejects": out / "registry_rejects.csv",
        "non_subagents": out / "registry_non_subagents.csv",
        "report": out / "registry_load_report.txt",
    }
    result.registry.to_parquet(paths["registry"], index=False)
    # utf-8-sig, чтобы Excel открывал кириллицу
    result.rejects.to_csv(paths["rejects"], index=False, encoding="utf-8-sig")
    result.non_subagents.to_csv(paths["non_subagents"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(format_stats(result.stats), encoding="utf-8")
    return paths


def format_stats(stats) -> str:
    """Собирает текстовый отчёт по загрузке."""
    lines = [
        f"Файл: {stats['file']}",
        f"Исходных строк: {stats['source_rows']}",
        f"Строк на выходе (одна на билет): {stats['output_rows']}",
        f"Без ошибок: {stats['ok_rows']} ({stats['ok_share']:.2%})",
        f"В списке проблемных: {stats['reject_rows']}",
        f"Корпоративных (не субагенты): {stats['corporate_rows']}",
        f"Обрезанных имён (ключей): {len(stats['prefix_keys'])}",
        f"Обрезанных, решённых по билету (строк): {stats['prefix_ticket_rows']}",
        f"Строк-дублей: {stats['dup_rows']} (лишних копий: {stats['dup_extra_rows']})",
        f"Строк, где пассажиров не столько, сколько билетов: {stats['pax_mismatch_rows']}",
        "Причины:",
        *[f"  {k}: {v}" for k, v in stats["reasons"].items()],
    ]
    return "\n".join(lines) + "\n"


# ================================================================
# ЧАСТЬ 5. САМОПРОВЕРКА
# ================================================================

# Эталон только для выданного набора, на других данных цифры не совпадут
EXPECTED_GIVEN_DATASET = {
    "source_rows": 12146,
    "output_rows": 19023,
    "corporate_rows": 423,
    "prefix_keys": 19,
    "ticket_in_amount": 65,
}


def compare_with_etm(registry, data_dir, tolerance=1.0) -> pd.DataFrame:
    """Сверяет строки бота в ETM с реестром по билету, виду операции и сумме."""
    etm = read_csv_text(find_input_file(data_dir, "etm*.csv"))
    etm_op = {"выкуп": "sale", "возврат": "refund", "войд": "void"}
    bot = etm[(etm["creator"].str.strip() == "etm-bot") & etm["kind"].isin(etm_op)]

    reg = registry.dropna(subset=["amount_kgs"])
    amounts = {}
    for t, op, a in zip(reg["ticket10"], reg["op_type"], reg["amount_kgs"]):
        amounts.setdefault((t, op), []).append(float(a))
    present = set(zip(registry["ticket10"], registry["op_type"]))

    rows = []
    for kind, cell, amount in zip(bot["kind"], bot["tickets"], bot["amount_kgs"]):
        try:
            tickets = tickets10(cell)
        except ParseError:
            rows.append({"op_type": etm_op[kind], "found": False, "amount_ok": False})
            continue
        per_ticket = float(amount) / len(tickets)
        for t in tickets:
            key = (t, etm_op[kind])
            ok = any(abs(per_ticket - a) <= tolerance for a in amounts.get(key, []))
            rows.append({"op_type": etm_op[kind], "found": key in present, "amount_ok": ok})
    df = pd.DataFrame(rows)
    return df.groupby("op_type").agg(etm_tickets=("found", "size"), found=("found", "mean"),
                                     amount_match=("amount_ok", "mean"))


def self_check(result, data_dir) -> bool:
    """Проверяет результат на выданном наборе и печатает итог."""
    s = result.stats
    reg = result.registry
    checks = [
        ("строк на выходе = сумма билетов", s["output_rows"] == s["tickets_total"], s["output_rows"]),
        ("ok + проблемные = исходные строки", s["ok_rows"] + s["reject_rows"] == s["source_rows"],
         f"{s['ok_rows']} + {s['reject_rows']}"),
        ("доля строк без ошибок >= 95%", s["ok_share"] >= 0.95, f"{s['ok_share']:.2%}"),
        ("у каждой ok-строки есть сумма",
         bool(reg.loc[reg.parse_status == "ok", "amount_kgs"].notna().all()), ""),
        ("у каждого субагента непустой subagent_id",
         bool(reg.loc[reg.party_type == "subagent", "subagent_id"].notna().all()), ""),
    ]
    exp = EXPECTED_GIVEN_DATASET
    actual = {
        "source_rows": s["source_rows"], "output_rows": s["output_rows"],
        "corporate_rows": s["corporate_rows"], "prefix_keys": len(s["prefix_keys"]),
        "ticket_in_amount": s["reasons"].get("ticket_in_amount", 0),
    }
    for k, v in exp.items():
        checks.append((f"эталон выданного набора: {k} = {v}", actual[k] == v, actual[k]))

    etm = compare_with_etm(reg, data_dir)
    sale = etm.loc["sale"] if "sale" in etm.index else None
    checks.append(("каждая продажа бота найдена в реестре по ticket10",
                   sale is not None and sale["found"] == 1.0,
                   f"{sale['found']:.2%}" if sale is not None else "нет продаж"))
    checks.append(("сумма на билет совпадает (+-1 сом) у >= 95% продаж бота",
                   sale is not None and sale["amount_match"] >= 0.95,
                   f"{sale['amount_match']:.2%}" if sale is not None else ""))

    print("\nСамопроверка:")
    for name, ok, value in checks:
        print(f"  [{'OK ' if ok else 'НЕТ'}] {name}  {value}")
    print("\nETM (строки бота) против реестра:")
    print(etm.to_string(float_format=lambda x: f"{x:.2%}"))
    return all(ok for _, ok, _ in checks)


# ================================================================
# ЧАСТЬ 6. ЗАПУСК
# ================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка реестра агентов")
    parser.add_argument("--data", required=True, help="папка с выгрузками")
    parser.add_argument("--out", default="interim", help="папка для результатов")
    parser.add_argument("--check", action="store_true",
                        help="самопроверка на выданном наборе")
    args = parser.parse_args(argv)

    try:
        result = load_registry(args.data)
    except (InputFileError, ValidationError) as err:
        print(f"[Registry] Ошибка: {err}", file=sys.stderr)
        return 1
    paths = write_outputs(result, args.out)
    print(format_stats(result.stats))
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.check:
        return 0 if self_check(result, args.data) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
