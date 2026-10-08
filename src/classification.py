"""Stage 4: hypotheses tied to сверки evidence; confidence is not an ML probability."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import ValidationError

CATALOG = {
    "review_required": ("unclear", "Причина требует выяснения"),
    "registry_missing": ("unclear", "Для операции бота нет записи реестра"),
    "dependent_sources_agree": ("unclear", "ETM и реестр совпадают, но источники зависимы"),
    "period_difference": ("unclear", "Разные периоды: проверить задержку отражения"),
    "duplicate_suspected": ("unclear", "Повторные записи: проверить дубль"),
}
OWNER = {"none":"нет ошибки", "unclear":"требует выяснения", "bot":"бот ETM", "agent":"агент (реестр)", "1c":"1С"}


def classify_groups(matches, components):
    if matches.match_id.duplicated().any() or not components.match_id.isin(matches.match_id).all():
        raise ValidationError("Некорректные связи классификации с сверки")
    grouped = {mid:g.to_dict("records") for mid,g in components.groupby("match_id",sort=False)}
    output=[]
    for record in matches.to_dict("records"):
        mid=record["match_id"]
        items=grouped.get(mid,[])
        sides={s:[r for r in items if r["source"]==s] for s in ["1c","etm","registry"]}
        a,e,r=(record["amount_"+s] for s in ["1c","etm","registry"])
        ha,he,hr=(bool(sides[s]) for s in ["1c","etm","registry"])
        reasons=set(filter(None,str(record.get("review_reason","")).split(";")))
        creators={str(x.get("creator","")) for x in sides["etm"]}
        flags=[]
        if ha and he and abs(a-e)>1: flags.append("amount_difference")
        if record.get("period_mismatch",False): flags.append("period_difference")
        if any("repeated_events" in x for x in reasons): flags.append("duplicate_suspected")
        if record["op_type"]!="payment" and record.get("registry_status")=="missing_for_bot": flags.append("registry_missing")
        parse=[str(x.get("parse_status","")) for x in sides["registry"]]
        if any(x!="ok" for x in parse): flags.append("registry_parse_problem")
        void=any(x.get("original_kind")=="void" for x in sides["etm"])
        alternatives=[]
        code,owner,confidence,reason="review_required","unclear","low","Недостаточно независимых данных для определения источника ошибки"
        equal=lambda x,y: pd.notna(x) and pd.notna(y) and abs(x-y)<=1
        status=record["match_status"]
        if "duplicate_suspected" in flags:
            code,reason="duplicate_suspected","Повторные события сохранены в компонентах; одинаковая итоговая сумма не доказывает дубль"
            alternatives=["повторная законная операция","повторный импорт","ошибка записи"]
        elif any("ticket_in_amount" in x or "bad_amount" in x or "bad_number" in x for x in parse):
            code,owner,confidence,reason="registry_parse_error","agent","high","В реестре непосредственно обнаружено некорректное поле суммы"
        elif void and ha and a>1 and he and abs(e)<=1 and any(x.get("original_kind")=="void" for x in sides["registry"]):
            code,owner,confidence,reason="1c_voided_posted","1c","medium","Отмена подтверждается компонентами ETM и реестра, но в 1С остается продажа; проверить документ 1С"
            alternatives=["ошибочно зарегистрированная отмена","задержка исправления 1С"]
        elif "registry_missing" in flags:
            code,reason="registry_missing","Создатель ETM — бот; отсутствие реестра нельзя считать самостоятельной выпиской"
            alternatives=["неполная выгрузка реестра","ошибка билета или контрагента","расхождение происхождения операции"]
        elif status=="ambiguous" and reasons-{"registry_amount_difference"}:
            reason="Неоднозначное сопоставление: "+"; ".join(sorted(reasons))
            alternatives=["ошибка сопоставления","ошибка одного или нескольких источников"]
        elif "period_difference" in flags:
            code,confidence,reason="period_difference","medium","Операции связаны, но отражены в разные месяцы; источник задержки не подтвержден"
            alternatives=["задержка 1С","задержка ETM","допустимый порядок учета"]
        elif status=="voided":
            code,owner,confidence,reason="ok_voided","none","high","Выкуп и войд обнуляют долг ETM; оба компонента сохранены"
        elif ha and he and equal(a,e) and (not hr or equal(a,r)):
            if not hr and record["op_type"]!="payment" and creators!={"subagent"}:
                reason="Отсутствие реестра не подтверждено самостоятельной выпиской"
            else:
                code="ok_payment" if record["op_type"]=="payment" else ("ok_self_service" if not hr else "ok")
                owner,confidence,reason="none","high","Источники согласованы по сумме и периоду"
        elif ha and he and hr:
            if equal(a,r):
                code,owner,confidence,reason="etm_wrong_amount","bot","medium","ETM отличается от 1С и реестра; проверить исходную проводку бота"
                alternatives=["совпавшие ошибки 1С и реестра"]
            elif equal(a,e):
                code,owner,confidence,reason="registry_wrong_amount","agent","medium","Реестр отличается от 1С и ETM"
                alternatives=["различие правил конвертации или состава сборов"]
            elif equal(e,r):
                code,reason="dependent_sources_agree","ETM мог скопировать ошибку реестра; совпадение этих источников не доказывает ошибку 1С"
                alternatives=["ошибка 1С","ошибка реестра, перенесенная ботом"]
        elif ha and hr and not he and equal(a,r):
            code,owner,confidence,reason="etm_not_executed","bot","medium","Операция подтверждается 1С и реестром, но отсутствует в ETM"
            alternatives=["неполная выгрузка ETM","задержка исполнения"]
        else:
            alternatives=["неполная выгрузка","задержка отражения","ошибка номера билета или контрагента","непроведенная операция"]
        if void and status!="voided" and "void_not_fully_reversed" in reasons:
            flags.append("void_not_fully_reversed")
            code,owner,confidence,reason="review_required","unclear","low","Войд не обнуляет выкуп: проверить частичную отмену, сборы и сроки"
        values=[float(x) for x in [a,e,r] if pd.notna(x)]
        risk=max(values)-min(values) if len(values)>1 else (abs(values[0]) if values else np.nan)
        if hr and pd.isna(r): risk=np.nan
        if owner=="none": risk=0.0
        periods=sorted({str(x["period"]) for x in items if pd.notna(x.get("period"))})
        employees=sorted({str(x["creator"]) for x in sides["registry"] if pd.notna(x.get("creator")) and str(x["creator"])})
        result=dict(record,error_type=code,error_owner=owner,error_owner_ru=OWNER[owner],
            error_type_ru=CATALOG[code][1] if code in CATALOG else reason,confidence=confidence,
            reason=reason,issue="",is_error=int(owner!="none"),amount_at_stake=abs(risk),
            flags=json.dumps(sorted(set(flags)),ensure_ascii=False),
            alternative_causes=json.dumps(alternatives,ensure_ascii=False),
            assignment_status="no_issue" if owner=="none" else ("hypothesis" if confidence!="high" else "observed_data_error"),
            period=" | ".join(periods),grp=record["op_type"],amount_reg=r,
            created_by="+".join(employees),employee_ids=json.dumps(employees,ensure_ascii=False),
            doc_1c=record.get("documents_1c","[]"),txn_ids=record.get("documents_etm","[]"),
            proposed_correction=np.nan)
        for source in ["1c","etm"]:
            dates=[pd.Timestamp(x["date"]) for x in sides[source] if pd.notna(x.get("date"))]
            result["date_"+source]=min(dates) if dates else pd.NaT
        txns=[x.get("document","") for x in sides["etm"]]
        result["txn_id"]=pd.to_numeric(txns[0],errors="coerce") if len(txns)==1 else np.nan
        result["doc"]=record.get("documents_1c","[]")
        output.append(result)
    return pd.DataFrame(output)


def balance_cases(bridge):
    rows=[]
    for b in bridge.to_dict("records"):
        period=b["period"]
        for field,code,description in [("unexplained","unexplained_residual","Необъясненный остаток моста"),
            ("act_carryover_gap","act_carryover_gap","Разрыв переноса сальдо 1С"),
            ("check_1c","internal_1c_gap","Нарушение внутреннего баланса 1С"),
            ("check_etm","internal_etm_gap","Нарушение внутреннего баланса ETM")]:
            value=b.get(field,np.nan)
            if pd.notna(value) and abs(value)>0.02:
                rows.append(dict(case_id=f"B:{b['subagent_id']}:{period}:{code}",subagent_id=b["subagent_id"],
                    period=period,error_type=code,error_owner="unclear",confidence="low",reason=description,
                    signed_balance_value=value,requires_review=True,overlap_note="Диагностические случаи могут описывать один разрыв; суммы не складывать"))
        if b.get("bridge_status") in ["missing_act","unknown_etm_balance"]:
            rows.append(dict(case_id=f"B:{b['subagent_id']}:{period}:missing",subagent_id=b["subagent_id"],period=period,
                error_type=b["bridge_status"],error_owner="unclear",confidence="low",reason="Недостаточно данных для сравнения балансов",
                signed_balance_value=np.nan,requires_review=True,overlap_note="Не является суммой корректировки"))
    # Opening difference is a stock, not a new monthly error.
    for agent,g in bridge.groupby("subagent_id"):
        known=g[g.opening_difference.notna()].sort_values("period")
        if known.empty: continue
        first=known.iloc[0]
        if pd.notna(first.opening_difference) and abs(first.opening_difference)>0.02:
            rows.append(dict(case_id=f"B:{agent}:{first.period}:opening",subagent_id=agent,period=first.period,
                error_type="opening_difference",error_owner="unclear",confidence="low",reason="Начальная разница: нужна история до периода выгрузки",
                signed_balance_value=first.opening_difference,requires_review=True,overlap_note="Остаток, не месячный оборот"))
    return pd.DataFrame(rows,columns=["case_id","subagent_id","period","error_type","error_owner","confidence",
        "reason","signed_balance_value","requires_review","overlap_note"])


def load_reconciliation_classification(matching_dir):
    p=Path(matching_dir)
    m=pd.read_csv(p/"operation_matches.csv",dtype={"ticket10":str},encoding="utf-8-sig")
    c=pd.read_csv(p/"match_components.csv",dtype={"ticket10":str},encoding="utf-8-sig",low_memory=False)
    classified=classify_groups(m,c)
    cases=balance_cases(pd.read_csv(p/"balance_bridge.csv",encoding="utf-8-sig"))
    links=pd.read_csv(p/"balance_contributions.csv",encoding="utf-8-sig")
    links=links.merge(classified[["match_id","error_type","error_owner","confidence"]],on="match_id",validate="many_to_one")
    return classified,cases,links
