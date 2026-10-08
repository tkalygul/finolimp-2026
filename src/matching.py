"""Step 2: traceable matching with conservation of every source amount.

All amounts use the debt sign (sale positive, payment/refund negative).
Repeated ticket events are kept together and flagged for review, never
expanded into Cartesian products or silently resolved as one event.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import ValidationError

TOLERANCE = 1.0
PAYMENT_DAYS = 5
SOURCES = ("1c", "etm", "registry")
COMPONENT_COLUMNS = ["source", "source_row_id", "allocation_id", "subagent_id",
                     "ticket10", "op_type", "original_kind", "period", "date",
                     "amount", "document", "payment_document", "creator", "parse_status"]


def _text(value):
    return "" if pd.isna(value) else str(value)


def _tickets(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(x) for x in cell]
    return _text(cell).split()


def prepare_components(acts, etm, registry):
    rows = []
    for source, frame in [("1c", acts), ("etm", etm)]:
        frame = frame.copy()
        frame.attrs.clear()
        for position, row in enumerate(frame.to_dict("records"), 2):
            original = _text(row.get("line_type" if source == "1c" else "kind_en"))
            kind = {"sale": "sale", "service_fee": "sale", "purchase": "sale",
                    "void": "sale", "refund": "refund", "payment_bank": "payment",
                    "payment_cash": "payment", "payment": "payment"}.get(original, "unclassified")
            tickets = _tickets(row.get("tickets10")) if kind != "payment" else []
            if kind not in ("payment", "unclassified") and not tickets:
                kind = "unclassified"
            tickets = tickets or [""]
            row_id = source + ":" + _text(row.get("source_row", position))
            amount = float(row["debt_delta"])
            if not np.isfinite(amount):
                raise ValidationError(f"Non-finite amount: {row_id}")
            for seq, ticket in enumerate(tickets, 1):
                rows.append(dict(source=source, source_row_id=row_id, allocation_id=f"{row_id}:{seq}",
                    subagent_id=_text(row["subagent_key" if source == "1c" else "agent_key"]),
                    ticket10=ticket, op_type=kind, original_kind=original, period=_text(row["period"]),
                    date=pd.Timestamp(row["date"]), amount=amount / len(tickets),
                    document=_text(row.get("doc" if source == "1c" else "txn_id")),
                    payment_document=_text(row.get("pay_doc")),
                    creator=_text(row.get("creator_en")), parse_status="ok"))
    reg = registry.copy()
    reg.attrs.clear()
    reg = reg[reg.party_type.eq("subagent") & reg.subagent_id.notna() & reg.ticket10.notna()]
    for row in reg.to_dict("records"):
        original = _text(row["op_type"])
        kind = {"sale": "sale", "void": "sale", "refund": "refund"}.get(original, "unclassified")
        rid = _text(row["row_id"])
        ts = pd.Timestamp(row["ts"])
        value = row["amount_kgs"]
        rows.append(dict(source="registry", source_row_id=rid,
            allocation_id=rid + ":" + _text(row["ticket_seq"]), subagent_id=str(row["subagent_id"]),
            ticket10=str(row["ticket10"]), op_type=kind, original_kind=original,
            period=str(ts.to_period("M")) if pd.notna(ts) else "", date=ts,
            amount=-float(value) if pd.notna(value) else np.nan,
            document=rid, payment_document="", creator=_text(row["created_by"]),
            parse_status=_text(row["parse_status"])))
    out = pd.DataFrame(rows, columns=COMPONENT_COLUMNS)
    if out.duplicated(["source", "allocation_id"]).any():
        raise ValidationError("Repeated source allocation identifiers")
    return out


def _describe(group, match_id, force_review=""):
    records = group if isinstance(group, list) else group.to_dict("records")
    first = records[0]
    by_source = {source: [r for r in records if r["source"] == source] for source in SOURCES}
    row = dict(match_id=match_id, subagent_id=first["subagent_id"], ticket10=first["ticket10"], op_type=first["op_type"])
    for source, selected in by_source.items():
        values = [r["amount"] for r in selected]
        row["amount_" + source] = sum(values) if values and all(pd.notna(v) for v in values) else np.nan
        row["rows_" + source] = len({r["source_row_id"] for r in selected})
        for field, name in [("source_row_id", "refs"), ("document", "documents"), ("period", "periods")]:
            row[name + "_" + source] = json.dumps(sorted({r[field] for r in selected} - {""}), ensure_ascii=False)
    a, e = row["amount_1c"], row["amount_etm"]
    ha, he, hr = (bool(by_source[source]) for source in SOURCES)
    has_void = any(r["original_kind"] == "void" for r in by_source["etm"])
    reasons = [force_review] if force_review else []
    if row["op_type"] == "unclassified":
        reasons.append("unclassified_operation")
    if any(r["parse_status"] != "ok" for r in records):
        reasons.append("registry_parse_problem")
    if row["op_type"] in ("sale", "refund"):
        for source, primary in [("1c", "sale" if row["op_type"] == "sale" else "refund"),
                                ("etm", "purchase" if row["op_type"] == "sale" else "refund"),
                                ("registry", "sale" if row["op_type"] == "sale" else "refund")]:
            if len({r["source_row_id"] for r in by_source[source] if r["original_kind"] == primary}) > 1:
                reasons.append("repeated_events_" + source)
    row["difference_1c_etm"] = (a if ha else 0) - (e if he else 0)
    row["period_mismatch"] = ha and he and row["periods_1c"] != row["periods_etm"]
    row["amount_mismatch"] = ha and he and abs(a-e) > TOLERANCE
    row["difference_1c_registry"] = a - row["amount_registry"]
    row["difference_etm_registry"] = e - row["amount_registry"]
    if hr and ((ha and abs(row["difference_1c_registry"]) > TOLERANCE) or
               (he and abs(row["difference_etm_registry"]) > TOLERANCE)):
        reasons.append("registry_amount_difference")
    if has_void and abs(e) > TOLERANCE:
        reasons.append("void_not_fully_reversed")
    if reasons:
        status = "ambiguous"
    elif has_void and not ha and abs(e) <= TOLERANCE:
        status = "voided"
    elif ha and he:
        status = "amount_difference" if row["amount_mismatch"] else ("period_mismatch" if row["period_mismatch"] else "matched")
    else:
        status = "only_1c" if ha else ("only_etm" if he else "only_registry")
    row["match_status"] = status
    creators = {r["creator"] for r in by_source["etm"]}
    row["registry_status"] = "not_applicable_payment" if row["op_type"]=="payment" else ("present" if hr else ("missing_for_bot" if "bot" in creators else
                                  ("not_expected_self_service" if he and creators == {"subagent"} else "absent"))
    )
    if row["registry_status"] == "missing_for_bot":
        reasons.append("registry_missing_for_bot")
    row["requires_review"] = bool(reasons) or status not in ("matched", "voided")
    row["review_reason"] = ";".join(reasons)
    return row


def match_components(components):
    c = components.copy()
    c["match_id"] = ""
    results = []
    assignments = {}
    records = dict(zip(c.index, c.to_dict("records")))
    def consume(indices, reason=""):
        mid = f"M{len(results) + 1:07d}"
        group = [records[idx] for idx in indices]
        results.append(_describe(group, mid, reason))
        assignments.update({idx: mid for idx in indices})

    ticket = c[c.op_type.isin(["sale", "refund"])]
    for _, group in ticket.groupby(["subagent_id", "ticket10", "op_type"], sort=True, dropna=False):
        consume(group.index)
    # Payments with reliable document identifiers form one-to-many bundles.
    payment = c[c.op_type.eq("payment")]
    bundles = []
    for _, sub in payment.groupby("subagent_id", sort=True):
        for (source, doc), group in sub[sub.payment_document.ne("")].groupby(["source", "payment_document"], sort=True):
            # Same document reused at distant dates must not be netted.
            ordered = group.sort_values(["date", "allocation_id"])
            clusters = ordered.date.diff().dt.total_seconds().gt(PAYMENT_DAYS * 86400).cumsum()
            for _, part in ordered.groupby(clusters):
                bundles.append(dict(indices=list(part.index), source=source, agent=part.subagent_id.iloc[0], doc=doc,
                                    amount=part.amount.sum(), date=part.date.min(), end=part.date.max()))
        for idx, row in sub[sub.payment_document.eq("")].iterrows():
            bundles.append(dict(indices=[idx], source=row.source, agent=row.subagent_id, doc="",
                                amount=row.amount, date=row.date, end=row.date))
    used = set()
    candidates = {}
    for i, left in enumerate(bundles):
        if left["source"] != "1c":
            continue
        options = []
        for j, right in enumerate(bundles):
            if right["source"] != "etm" or left["agent"] != right["agent"]:
                continue
            distance = max(abs((left["date"]-right["end"]).total_seconds()),
                           abs((left["end"]-right["date"]).total_seconds())) / 86400
            if distance > PAYMENT_DAYS:
                continue
            same_doc = bool(left["doc"]) and left["doc"] == right["doc"]
            if same_doc or abs(left["amount"]-right["amount"]) <= TOLERANCE:
                options.append((j, same_doc))
        # Document evidence beats an amount-only candidate.
        candidates[i] = [j for j, same in options if same] if any(s for _, s in options) else [j for j, _ in options]
    reverse = {}
    for i, options in candidates.items():
        for j in options:
            reverse.setdefault(j, []).append(i)
    for i, options in candidates.items():
        if len(options) == 1 and len(reverse[options[0]]) == 1:
            j = options[0]
            consume(bundles[i]["indices"] + bundles[j]["indices"])
            used.update([i, j])
    for i, bundle in enumerate(bundles):
        if i not in used:
            possible = bool(candidates.get(i)) or bool(reverse.get(i))
            consume(bundle["indices"], "ambiguous_payment_candidates" if possible else "")
    for idx in c.index[~c.index.isin(assignments)]:
        consume([idx], "unclassified_operation")
    c["match_id"] = c.index.map(assignments)
    matches = pd.DataFrame(results)
    if matches.empty:
        raise ValidationError("No operations available for matching")
    checks = []
    for source in SOURCES:
        expected = components.loc[components.source.eq(source), "amount"].sum(min_count=1)
        actual = c.loc[c.source.eq(source), "amount"].sum(min_count=1)
        checks.append(dict(source=source, source_components=int(components.source.eq(source).sum()),
                           matched_components=int(c.source.eq(source).sum()), input_amount=expected,
                           output_amount=actual, difference=actual-expected))
    if c.match_id.eq("").any() or c.duplicated(["source", "allocation_id"]).any() or len(c) != len(components):
        raise ValidationError("Source allocation lost or reused during matching")
    # Verify the summary amounts as well, not just the unchanged components.
    for source in ("1c", "etm"):
        expected = components.loc[components.source.eq(source), "amount"].sum()
        if abs(matches["amount_" + source].sum() - expected) > 1e-6:
            raise ValidationError(f"Matching changed {source} total")
    return matches, c, pd.DataFrame(checks)


def match_operations(acts, etm, registry):
    return match_components(prepare_components(acts, etm, registry))[0]


def build_summary(matches):
    return matches.groupby("subagent_id", as_index=False).agg(
        operations=("match_id", "size"), requires_review=("requires_review", "sum"),
        difference_1c_etm=("difference_1c_etm", "sum"))


def run_matching(clean_dir="interim/clean", registry_path="interim/registry.parquet", out_dir="interim/reconciliation"):
    ready = Path(clean_dir) / "reconciliation_ready"
    acts = pd.read_csv(ready / "acts_clean.csv", dtype={"tickets10": str, "pay_doc": str}, encoding="utf-8-sig")
    etm = pd.read_csv(ready / "etm_clean.csv", dtype={"tickets10": str, "txn_id": str, "pay_doc": str}, encoding="utf-8-sig")
    registry = pd.read_parquet(registry_path)
    matches, components, checks = match_components(prepare_components(acts, etm, registry))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for frame, name in [(matches, "operation_matches.csv"), (components, "match_components.csv"),
                        (checks, "matching_checks.csv"), (build_summary(matches), "matching_summary.csv")]:
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    print(f"[Сопоставление] {len(matches)} groups, {int(matches.requires_review.sum())} require review. Output: {output}")
    return dict(matches=matches, components=components, checks=checks)
