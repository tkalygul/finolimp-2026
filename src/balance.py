"""Step 3: monthly signed balance bridge. Unknown balances stay unknown."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import ValidationError

CAUSES = ["ops_amount_difference", "ops_only_1c", "ops_only_etm", "ops_period_mismatch",
          "payments_only_1c", "payments_only_etm", "payments_period_mismatch",
          "payments_amount_difference", "ops_ambiguous", "payments_ambiguous", "ops_unclassified",
          "ops_matched", "ops_voided", "payments_matched"]


def monthly_contributions(matches, components):
    if matches.match_id.duplicated().any() or components.duplicated(["source", "allocation_id"]).any():
        raise ValidationError("Duplicate match or source allocation")
    c = components[components.source.isin(["1c", "etm"])].copy()
    if not np.isfinite(c.amount).all():
        raise ValidationError("Unknown financial component amount")
    if not c.match_id.isin(matches.match_id).all():
        raise ValidationError("Component references an unknown matching group")
    c["signed_contribution"] = np.where(c.source.eq("1c"), c.amount, -c.amount)
    detail = c.groupby(["match_id", "subagent_id", "period"], as_index=False).signed_contribution.sum()
    meta = matches[["match_id", "subagent_id", "op_type", "match_status", "requires_review", "review_reason"]]
    detail = detail.merge(meta, on=["match_id", "subagent_id"], validate="many_to_one")
    def cause(row):
        prefix = "payments" if row.op_type == "payment" else "ops"
        status = row.match_status
        if row.op_type == "unclassified":
            return "ops_unclassified"
        if status == "ambiguous":
            return prefix + "_ambiguous"
        if status == "voided":
            return "ops_voided"
        if status in ["matched", "amount_difference", "only_1c", "only_etm", "period_mismatch"]:
            return prefix + "_" + status
        return prefix + "_ambiguous"
    detail["cause"] = detail.apply(cause, axis=1)
    return detail


def agreement_months(etm, periods):
    e = etm.copy()
    e.attrs.clear()
    e["date"] = pd.to_datetime(e.date)
    e["_txn_order"] = pd.to_numeric(e.txn_id, errors="coerce")
    rows = []
    for (agent, agreement), group in e.groupby(["agent_key", "agreement_id"], sort=True):
        group = group.sort_values(["date", "_txn_order", "txn_id"], kind="stable")
        first_period = str(group.iloc[0].period)
        opening = float(group.iloc[0].balance_after - group.iloc[0].amount_kgs)
        previous = np.nan
        for period in periods:
            operations = group[group.period.eq(period)]
            known = period >= first_period
            start = opening if period == first_period else previous
            end = float(operations.iloc[-1].balance_after) if not operations.empty else previous
            turnover = float(operations.amount_kgs.sum())
            rows.append(dict(subagent_id=agent, agreement_id=agreement, period=period,
                etm_balance_start=start if known else np.nan, etm_balance_end=end if known else np.nan,
                turnover_etm=turnover, agreement_known=known,
                inactive=known and operations.empty))
            previous = end
    return pd.DataFrame(rows, columns=["subagent_id", "agreement_id", "period", "etm_balance_start",
        "etm_balance_end", "turnover_etm", "agreement_known", "inactive"])


def build_balance(acts, etm, matches, components):
    periods_present = set(acts.period.astype(str)) | set(etm.period.astype(str))
    if not periods_present:
        raise ValidationError("No accounting periods")
    periods = pd.period_range(min(periods_present), max(periods_present), freq="M").astype(str).tolist()
    agents = sorted(set(acts.subagent_key) | set(etm.agent_key))
    grid = pd.MultiIndex.from_product([agents, periods], names=["subagent_id", "period"]).to_frame(index=False)
    a = acts.copy()
    a.attrs.clear()
    act_values = a[["saldo_start","saldo_end","debt_delta"]].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    etm_values = etm[["amount_kgs","balance_after"]].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(act_values).all() or not np.isfinite(etm_values).all():
        raise ValidationError("Unknown source amount: repeat stage 1 validation")
    for _, group in a.groupby(["subagent_key", "period"]):
        if group[["saldo_start", "saldo_end"]].nunique().gt(1).any():
            raise ValidationError("Conflicting act balances")
    headers = a.groupby(["subagent_key", "period"], as_index=False).agg(
        saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first"),
        turnover_1c=("debt_delta", "sum")).rename(columns={"subagent_key": "subagent_id"})
    headers["act_exists"] = True
    bridge = grid.merge(headers, on=["subagent_id", "period"], how="left", validate="one_to_one")
    bridge["act_exists"] = bridge.act_exists.fillna(False).astype(bool)
    agreements = agreement_months(etm, periods)
    totals = []
    for (agent, period), group in agreements.groupby(["subagent_id", "period"]):
        known = bool(group.agreement_known.all())
        totals.append(dict(subagent_id=agent, period=period, etm_known=known,
            agreement_count=len(group), unknown_agreements=int((~group.agreement_known).sum()),
            etm_balance_start=group.etm_balance_start.sum() if known else np.nan,
            etm_balance_end=group.etm_balance_end.sum() if known else np.nan,
            turnover_etm=group.turnover_etm.sum(), inactive_agreements=int(group.inactive.sum())))
    totals = pd.DataFrame(totals, columns=["subagent_id", "period", "etm_known", "agreement_count",
        "unknown_agreements", "etm_balance_start", "etm_balance_end", "turnover_etm", "inactive_agreements"])
    bridge = bridge.merge(totals, on=["subagent_id", "period"], how="left", validate="one_to_one")
    bridge["etm_known"] = bridge.etm_known.fillna(False).astype(bool)
    details = monthly_contributions(matches, components)
    review = details[details.requires_review.astype(bool)].groupby(["subagent_id","period"]).match_id.nunique().rename("cause_review_groups").reset_index()
    bridge = bridge.merge(review,on=["subagent_id","period"],how="left",validate="one_to_one")
    bridge["cause_review_groups"] = bridge.cause_review_groups.fillna(0).astype(int)
    if not details.empty:
        causes = details.pivot_table(index=["subagent_id", "period"], columns="cause",
                                     values="signed_contribution", aggfunc="sum", fill_value=0).reset_index()
        bridge = bridge.merge(causes, on=["subagent_id", "period"], how="left", validate="one_to_one")
    for column in CAUSES:
        bridge[column] = bridge[column].fillna(0) if column in bridge else 0.0
    bridge["explained_turnover_difference"] = bridge[CAUSES].sum(axis=1)
    bridge["check_1c"] = bridge.saldo_start + bridge.turnover_1c - bridge.saldo_end
    bridge["check_etm"] = bridge.etm_balance_start + bridge.turnover_etm - bridge.etm_balance_end
    bridge["opening_difference"] = bridge.saldo_start + bridge.etm_balance_start
    bridge["closing_difference"] = bridge.saldo_end + bridge.etm_balance_end
    bridge["turnover_difference"] = bridge.turnover_1c + bridge.turnover_etm
    bridge["component_gap"] = bridge.turnover_difference - bridge.explained_turnover_difference
    bridge["unexplained"] = bridge.closing_difference - bridge.opening_difference - bridge.explained_turnover_difference
    bridge["bridge_check"] = bridge.opening_difference + bridge.explained_turnover_difference + bridge.unexplained - bridge.closing_difference
    bridge["unresolved_cause_contribution"] = bridge.ops_ambiguous + bridge.payments_ambiguous + bridge.ops_unclassified
    bridge["bridge_status"] = np.select(
        [~bridge.act_exists, ~bridge.etm_known, bridge.unexplained.abs().gt(0.02)],
        ["missing_act", "unknown_etm_balance", "unexplained_residual"], default="reconciled")
    # Continuity differences are separate from each month's turnover bridge.
    bridge = bridge.sort_values(["subagent_id", "period"]).reset_index(drop=True)
    previous_end = bridge.groupby("subagent_id").saldo_end.shift()
    bridge["act_carryover_gap"] = bridge.saldo_start - previous_end
    bridge["cumulative_difference_change"] = bridge.closing_difference - bridge.groupby("subagent_id").closing_difference.shift()
    return bridge, details, agreements


def build_balance_summary(matches, bridge):
    rows = []
    for agent, b in bridge.groupby("subagent_id", sort=True):
        m = matches[matches.subagent_id.eq(agent)]
        ticket = m[m.op_type.ne("payment")]
        payment = m[m.op_type.eq("payment")]
        def count(frame, status):
            return int(frame.match_status.eq(status).sum())
        row = dict(subagent_id=agent, operations=len(ticket), matched=count(ticket,"matched"),
            voided=count(ticket,"voided"), amount_difference=count(ticket,"amount_difference"),
            only_1c=count(ticket,"only_1c"), only_etm=count(ticket,"only_etm"),
            period_mismatch=count(ticket,"period_mismatch"),
            amount_difference_sum=ticket.loc[ticket.match_status.eq("amount_difference"),"difference_1c_etm"].sum(),
            only_1c_amount=ticket.loc[ticket.match_status.eq("only_1c"),"difference_1c_etm"].sum(),
            only_etm_amount=ticket.loc[ticket.match_status.eq("only_etm"),"difference_1c_etm"].sum(),
            payments=len(payment), payments_only_1c=count(payment,"only_1c"), payments_only_etm=count(payment,"only_etm"),
            closing_difference_last=b.iloc[-1].closing_difference,
            unexplained_total=b.unexplained.sum() if b.unexplained.notna().all() else np.nan,
            unexplained_absolute_total=b.unexplained.abs().sum() if b.unexplained.notna().all() else np.nan,
            unknown_months=int(b.unexplained.isna().sum()),
            unresolved_groups=int(m.requires_review.sum()),
            matched_share=(count(ticket,"matched")+count(ticket,"voided"))/len(ticket) if len(ticket) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def run_balance(clean_dir="interim/clean", matching_dir="interim/reconciliation", out_dir=None):
    ready = Path(clean_dir) / "reconciliation_ready"
    path = Path(matching_dir)
    acts = pd.read_csv(ready/"acts_clean.csv", encoding="utf-8-sig", dtype={"subagent_key":str})
    etm = pd.read_csv(ready/"etm_clean.csv", encoding="utf-8-sig", dtype={"agent_key":str,"agreement_id":str,"txn_id":str})
    matches = pd.read_csv(path/"operation_matches.csv", encoding="utf-8-sig", dtype={"subagent_id":str,"ticket10":str})
    components = pd.read_csv(path/"match_components.csv", encoding="utf-8-sig", dtype={"subagent_id":str,"ticket10":str}, low_memory=False)
    bridge, details, agreements = build_balance(acts, etm, matches, components)
    out = Path(out_dir) if out_dir else path
    out.mkdir(parents=True, exist_ok=True)
    for frame, name in [(bridge,"balance_bridge.csv"),(build_balance_summary(matches,bridge),"subagent_summary.csv"),
                        (details,"balance_contributions.csv"),(agreements,"agreement_balances.csv")]:
        frame.to_csv(out/name,index=False,encoding="utf-8-sig")
    print(f"[Balance] {len(bridge)} subagent-months: {bridge.bridge_status.value_counts().to_dict()}")
    return bridge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Мост баланса 1С и ETM")
    parser.add_argument("--clean",default="interim/clean")
    parser.add_argument("--matching",default="interim/reconciliation")
    parser.add_argument("--out")
    args = parser.parse_args()
    run_balance(args.clean,args.matching,args.out)
