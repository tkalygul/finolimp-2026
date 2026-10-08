"""Контроль исходных строк: ошибки не заменяются финансовыми нулями."""
import numpy as np
import pandas as pd
from src.normalize import parse_number, ParseError, subagent_key
from src.schema import validate


def read_checked(path, kind):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    validate(raw, kind)
    raw["source_row"] = np.arange(2, len(raw) + 2)
    return raw


def prepare(raw, kind, numeric, dates, optional_zero=()):
    data = raw.copy()
    issues = []
    def issue(mask, field, reason):
        for i in data.index[mask]:
            issues.append({"source": kind, "source_row": int(raw.at[i, "source_row"]),
                           "field": field, "raw_value": raw.at[i, field], "reason": reason})
    for field in numeric:
        values = []
        for cell in raw[field]:
            if not cell.strip():
                value = 0.0 if field in optional_zero else np.nan
            else:
                try:
                    value = parse_number(cell)
                    if not np.isfinite(value):
                        value = np.nan
                except (ParseError, ValueError, OverflowError):
                    value = np.nan
            values.append(value)
        data[field] = values
        issue(data[field].isna(), field, "missing_or_invalid_number")
    for field in dates:
        # Формат источников 1С/ETM: ISO, время допускается.
        iso = raw[field].str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?")
        data[field] = pd.to_datetime(raw[field].where(iso), format="mixed", errors="coerce")
        issue(data[field].isna(), field, "missing_or_invalid_date")
    required_text = ["folder", "doc"] if kind == "acts" else ["agent", "agreement_id", "txn_id"]
    for field in required_text:
        issue(raw[field].str.strip().eq(""), field, "missing_value")
    if kind == "acts":
        issue(~raw.act_status.str.strip().str.lower().isin(["", "черновик", "переиздан"]),
              "act_status", "unknown_act_status")
        issue(data.period_start.gt(data.period_end), "period_end", "invalid_period")
        issue(data.date.lt(data.period_start) | data.date.gt(data.period_end), "date", "date_outside_period")
    else:
        issue(~raw.kind.str.strip().str.lower().isin(["выкуп", "возврат", "войд", "оплата"]), "kind", "unknown_kind")
        issue(~raw.currency.str.strip().str.upper().isin(["KGS", "USD", "EUR", "RUB", "KZT"]), "currency", "unknown_currency")
        issue(~raw.creator.str.strip().isin(["etm-bot", "субагент"]), "creator", "unknown_creator")
        issue(raw.txn_id.duplicated(keep=False), "txn_id", "duplicate_transaction_id")
        data["currency"] = raw.currency.str.strip().str.upper()
    issues = pd.DataFrame(issues, columns=["source", "source_row", "field", "raw_value", "reason"])
    bad = set(issues.source_row)
    audit = raw.copy()
    audit["disposition"] = np.where(audit.source_row.isin(bad), "rejected", "accepted")
    return data[~data.source_row.isin(bad)].copy(), audit, issues


def key_candidates(audit, field, source):
    out = audit[["source_row", field]].rename(columns={field: "original_name"}).copy()
    out["source"] = source
    out["normalized_key"] = out.original_name.map(subagent_key)
    count = out.groupby("normalized_key").original_name.transform("nunique")
    out["review_required"] = count.gt(1)
    return out


def split_money(frame, tickets, amount):
    """Разделение в копейках: остаток от деления получает последний билет."""
    frame = frame.copy()
    frame.attrs.clear()
    rows = []
    for _, row in frame.iterrows():
        numbers = row[tickets]
        if not numbers:
            continue
        cents = int(round(float(row[amount]) * 100))
        base, remainder = divmod(cents, len(numbers))
        for pos, ticket in enumerate(numbers):
            item = row.to_dict()
            item[tickets] = ticket
            item[amount] = (base + (remainder if pos == len(numbers) - 1 else 0)) / 100
            rows.append(item)
    return pd.DataFrame(rows, columns=frame.columns)
