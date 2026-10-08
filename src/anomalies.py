"""Stage 5: evidence-linked anomaly signals; signals are not accounting losses."""
import hashlib
import json
import math
import numpy as np
import pandas as pd

REPEAT_DAYS=7
VOID_WAIT_DAYS=5
MIN_EMPLOYEE_ROWS=30
MIN_EMPLOYEE_ERRORS=5
BASE_COLUMNS=["anomaly_id","anomaly_type","error_owner","severity","subagent_id","period","ref",
              "amount_kgs","detail","status","confidence","match_ids","source_refs","transaction_ids",
              "overlap_key","amount_semantics","suggested_owner"]


def _strings(values):
    return sorted({str(v) for v in values if pd.notna(v) and str(v)})


def employee_concentration(classified, components):
    reg=components[components.source.eq("registry")].copy()
    reg=reg[reg.creator.notna() & reg.creator.astype(str).ne("")]
    columns=["created_by","rows","agent_errors","suspected_agent_errors","review_rows","shared_review_rows",
             "error_rate","wilson_lower","reference_rate","frequent_failures","enough_data"]
    if reg.empty: return pd.DataFrame(columns=columns)
    reg=reg.merge(classified[["match_id","error_owner","confidence","is_error"]],on="match_id",validate="many_to_one")
    counts=reg.groupby("match_id").creator.nunique()
    reg["shared"]=reg.match_id.map(counts).gt(1)
    # Count original registry rows, not exploded tickets. Shared groups are not individually blamed.
    reg["observed"] = reg.error_owner.eq("agent") & reg.confidence.eq("high") & ~reg.shared
    reg["suspected"] = reg.error_owner.eq("agent") & ~reg.confidence.eq("high") & ~reg.shared
    reg["review"] = reg.is_error.eq(1)
    reg["shared_review"] = reg.shared & reg.review
    rows=reg.groupby(["creator","source_row_id"],as_index=False)[["observed","suspected","review","shared_review"]].max()
    baseline=float(rows.observed.sum()/len(rows)) if len(rows) else 0
    result=[]
    for employee,g in rows.groupby("creator",sort=True):
        n,k=len(g),int(g.observed.sum())
        p=k/n
        z=1.96
        lower=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        enough=n>=MIN_EMPLOYEE_ROWS
        result.append(dict(created_by=employee,rows=n,agent_errors=k,suspected_agent_errors=int(g.suspected.sum()),
            review_rows=int(g.review.sum()),shared_review_rows=int(g.shared_review.sum()),error_rate=p,
            wilson_lower=max(0,lower),reference_rate=baseline,
            frequent_failures=enough and k>=MIN_EMPLOYEE_ERRORS and lower>baseline,enough_data=enough))
    return pd.DataFrame(result,columns=columns).sort_values(["frequent_failures","wilson_lower"],ascending=False)


def detect_canonical_anomalies(classified, components, extras=None):
    rows=[]
    c=components.copy()
    c["date"]=pd.to_datetime(c.date,errors="coerce")
    groups={key:g for key,g in c.groupby("match_id",sort=False)}
    def emit(kind, evidence, amount, detail, severity="medium", status="suspected", suggested="unclear", ref="",case_context=""):
        mids=_strings(evidence.match_id) if not evidence.empty else []
        refs=_strings(evidence.source_row_id) if not evidence.empty else ([ref] if ref else [])
        txns=_strings(evidence.loc[evidence.source.eq("etm"),"document"]) if not evidence.empty else []
        token=kind+"|"+"|".join(refs)+"|"+ref+"|"+case_context
        identity=hashlib.sha256(token.encode()).hexdigest()[:16]
        agent=str(evidence.subagent_id.iloc[0]) if not evidence.empty else ""
        periods=" | ".join(_strings(evidence.period)) if not evidence.empty else ""
        rows.append(dict(anomaly_id="A:"+identity,anomaly_type=kind,error_owner="unclear",severity=severity,
            subagent_id=agent,period=periods,ref=ref or " | ".join(txns or refs),amount_kgs=amount,detail=detail,
            status=status,confidence="high" if status=="observed_data_error" else "low",
            match_ids=json.dumps(mids,ensure_ascii=False),source_refs=json.dumps(refs,ensure_ascii=False),
            transaction_ids=json.dumps(txns,ensure_ascii=False),overlap_key=" | ".join(mids) or " | ".join(refs),
            amount_semantics="Потенциальная сумма; не корректировка и не независимый убыток",
            suggested_owner=suggested))

    # Reconstruct each ETM row from its allocations; this prevents multi-ticket double counting.
    events=[]
    for rid,g in c[c.source.eq("etm")].groupby("source_row_id",sort=False):
        events.append(dict(rid=rid,agent=g.subagent_id.iloc[0],kind=g.original_kind.iloc[0],
            amount=round(float(g.amount.sum()),2),date=g.date.min(),
            tickets=" ".join(_strings(g.ticket10)),document=str(g.payment_document.iloc[0]) if pd.notna(g.payment_document.iloc[0]) else "",
            creator=str(g.creator.iloc[0])))
    events=pd.DataFrame(events,columns=["rid","agent","kind","amount","date","tickets","document","creator"])
    for (_,kind,amount,tickets),g in events.groupby(["agent","kind","amount","tickets"],sort=True):
        if len(g)<2 or kind not in ("purchase","payment") or amount==0: continue
        ordered=g.dropna(subset=["date"]).sort_values(["date","rid"])
        clusters=[]
        for event in ordered.to_dict("records"):
            if not clusters or (event["date"]-clusters[-1][0]["date"]).total_seconds()>REPEAT_DAYS*86400:
                clusters.append([])
            clusters[-1].append(event)
        for cluster in clusters:
            if len(cluster)<2:continue
            if kind=="payment":
                # Distinct non-empty payment documents provide evidence of separate deposits.
                docs=[r["document"] for r in cluster]
                if all(docs) and len(set(docs))==len(docs):continue
            evidence=c[c.source_row_id.isin([r["rid"] for r in cluster]) & c.source.eq("etm")]
            emit("etm_double_credit_suspected" if kind=="payment" else "etm_double_debit_suspected",evidence,
                 abs(amount)*(len(cluster)-1),
                 "Повторные проводки за 7 дней. Проверить документы и законность повторения; разные ID не доказывают дубль.",
                 suggested="bot" if all(r["creator"]=="bot" for r in cluster) else "unclear")

    cutoff=c.loc[c.source.eq("etm"),"date"].max()
    void_match_ids=set(c.loc[c.original_kind.eq("void") & c.source.isin(["etm","registry"]),"match_id"])
    for record in classified.to_dict("records"):
        if record["op_type"]!="sale" or record["match_id"] not in void_match_ids:continue
        g=groups.get(record["match_id"])
        if g is None:continue
        purchases=g[g.source.eq("etm") & g.original_kind.eq("purchase")]
        voids=g[g.source.eq("etm") & g.original_kind.eq("void")]
        requests=g[g.source.eq("registry") & g.original_kind.eq("void")]
        if voids.empty and requests.empty:continue
        if purchases.empty:
            if not voids.empty:emit("void_without_purchase",g,np.nan,"Войд без исходного выкупа: проверить полноту истории")
            continue
        debt=float(purchases.amount.sum())
        credit=-float(voids.amount.sum()) if not voids.empty else 0.0
        remaining=debt-credit
        if remaining>1:
            dates=requests.date if not requests.empty else voids.date
            pending=pd.notna(cutoff) and not dates.empty and (cutoff-dates.max()).total_seconds()<VOID_WAIT_DAYS*86400
            emit("void_pending_credit" if pending else ("void_without_credit" if credit<=1 else "void_partial_credit"),g,
                 remaining,"Выкуп не погашен войдом. Проверить срок исполнения, сборы и частичную отмену; виновник не подтвержден.",
                 severity="low" if pending else "high",status="pending" if pending else "suspected",suggested="bot")
        elif abs(remaining)<=1 and not voids.empty:
            days=(voids.date.max()-purchases.date.min()).total_seconds()/86400
            if days>VOID_WAIT_DAYS:
                emit("void_delayed_credit",g,0.0,"Войд погасил выкуп позже 5 дней; это информационный сигнал, не потеря денег",
                     severity="low",status="informational")
        else:
            emit("void_excess_credit",g,abs(remaining),"Войд вернул больше выкупа; проверить исходные операции",severity="high")

    employees=employee_concentration(classified,c)
    for employee in employees.loc[employees.frequent_failures.eq(True)].to_dict("records"):
        evidence=c[c.source.eq("registry") & c.creator.eq(employee["created_by"])]
        emit("employee_error_concentration",evidence,np.nan,
             f"Сотрудник {employee['created_by']}: {employee['agent_errors']} наблюдаемых ошибок на {employee['rows']} исходных строк. Нижняя граница Wilson выше общей доли; проверить выборку.",
             ref="employee:"+employee["created_by"])

    # Preserve supplementary quality signals, with row-level deduplication and evidence links.
    if extras is not None and not extras.empty:
        # Index evidence once; avoid scanning every component for every quality signal.
        references={}
        for index,agent,document,rid in zip(c.index,c.subagent_id,c.document,c.source_row_id):
            if pd.isna(agent): continue
            for ref in {str(value) for value in [document,rid] if pd.notna(value)}:
                references.setdefault((str(agent),ref),[]).append(index)
        exclude={"etm_duplicate_op","etm_double_credit"}
        other=extras[~extras.anomaly_type.isin(exclude)].copy()
        for (kind,agent,ref),part in other.groupby(["anomaly_type","subagent_id","ref"],dropna=False,sort=False):
            ref=str(ref) if pd.notna(ref) else ""
            evidence=c.loc[references.get((str(agent),ref),[])]
            amount=part.amount_kgs.sum(min_count=1) if kind.startswith("registry_") else part.amount_kgs.iloc[0]
            observed=kind in {"etm_wrong_sign","registry_parse_problem"}
            emit(kind,evidence,amount," | ".join(_strings(part.detail)),severity=str(part.severity.iloc[0]),
                 status="observed_data_error" if observed else "suspected",ref=ref,case_context=str(agent)+":"+" | ".join(_strings(part.period)))
            rows[-1]["subagent_id"]=str(agent) if pd.notna(agent) else ""
            rows[-1]["period"]=" | ".join(_strings(part.period))
            # A malformed field is observed; the human/system responsible may still be unknown.
    anomalies=pd.DataFrame(rows,columns=BASE_COLUMNS)
    if not anomalies.empty:
        anomalies=anomalies.drop_duplicates("anomaly_id").reset_index(drop=True)
    return anomalies,employees
