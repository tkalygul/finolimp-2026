"""P2: Загрузка, очистка и экспорт актов сверки 1С и транзакций ETM.

Как запустить:
    python cleaning.py <папка_с_выгрузками> <папка_результата>
Пример:
    python cleaning.py data_2_final out/clean
"""
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Импортируем единую функцию нормализации ключа субагента из общего модуля src/normalize.py
from src.normalize import subagent_key
from src.quality import read_checked, prepare, key_candidates, split_money
from src.schema import ValidationError

# ================================================================
# ЧАСТЬ 1. ОБЩИЕ УТИЛИТЫ (помощники для текста и чисел)
# ================================================================

def _num(series) -> pd.Series:
    """Превращает строковые суммы с пробелами и запятыми в нормальные числа (float)."""
    def conv(x):
        if pd.isna(x):
            return np.nan
        x = str(x).replace("\xa0", "").replace(" ", "").strip()
        if x == "":
            return np.nan
        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            x = x.replace(",", ".")
        try:
            return float(x)
        except ValueError:
            return np.nan
    return series.map(conv)


# Шаблоны для поиска номеров авиабилетов (13-значных и 10-значных)
_T13 = re.compile(r"(?<!\d)\d{13}(?!\d)")
_T10 = re.compile(r"(?<!\d)\d{10}(?!\d)")


def tickets13(text) -> list:
    """Вытаскивает все 13-значные билеты из текста."""
    if pd.isna(text):
        return []
    s = re.sub(r"(?<!\d)(\d{3})\s*-\s*(\d{10})(?!\d)", r"\1\2", str(text))
    return _T13.findall(s)


def tickets10(text) -> list:
    """Вытаскивает все 10-значные билеты из текста."""
    if pd.isna(text):
        return []
    return _T10.findall(str(text))


# ================================================================
# ЧАСТЬ 2. ОБРАБОТКА АКТОВ 1С (acts.csv)
# ================================================================

def load_acts(path):
    """Загружает акты 1С, очищает суммы, даты и убирает устаревшие черновики."""
    raw = read_checked(path, "acts")
    log = []
    a, audit, issues = prepare(raw, "acts", ("saldo_start", "saldo_end", "debet", "credit"),
                               ("period_start", "period_end", "date"), ("debet", "credit"))
    
    # Приводим финансовые поля к числам, а даты — к формату дат
    for c in ("saldo_start", "saldo_end", "debet", "credit"):
        a[c] = _num(a[c])
    for c in ("period_start", "period_end", "date"):
        a[c] = pd.to_datetime(a[c], errors="coerce")
        
    a["act_status"] = a["act_status"].fillna("").str.strip().str.lower()
    a["subagent"] = a["folder"].str.strip()
    # Используем subagent_key из normalize.py вместо старой norm_name
    a["subagent_key"] = a["subagent"].map(subagent_key)
    a["period"] = a["period_start"].dt.to_period("M").astype(str)

    # Логика приоритетов: если есть переизданный акт, черновик отбрасываем
    prio = {"переиздан": 0, "": 1, "черновик": 2}
    a["_prio"] = a["act_status"].map(prio)
    best = a.groupby(["subagent", "period"])["_prio"].transform("min")
    dropped = a[a["_prio"] != best]
    
    log.append(("acts: строк в файле", len(raw)))
    log.append(("acts: отброшено строк устаревших версий (черновик при наличии переиздан)", len(dropped)))
    a = a[a["_prio"] == best].drop(columns="_prio").reset_index(drop=True)
    audit.loc[audit.source_row.isin(dropped.source_row), "disposition"] = "superseded"
    # При одинаковом приоритете противоречивые шапки нельзя выбирать через first().
    header_fields = ["period_start", "period_end", "saldo_start", "saldo_end", "act_status"]
    conflict = a.groupby(["subagent", "period"])[header_fields].transform("nunique").gt(1).any(axis=1)
    if conflict.any():
        extra = raw[raw.source_row.isin(a.loc[conflict, "source_row"])]
        issues = pd.concat([issues, pd.DataFrame({"source": "acts", "source_row": extra.source_row,
            "field": "act_status", "raw_value": extra.act_status, "reason": "conflicting_act_headers"})], ignore_index=True)
        audit.loc[audit.source_row.isin(extra.source_row), "disposition"] = "rejected"
        a = a[~conflict].copy()

    # Определяем тип каждой строки акта (продажа, возврат, оплата и т.д.)
    d = a["doc"].fillna("")
    a["line_type"] = np.select(
        [d.str.startswith("Реализация сервисный сбор"),
         d.str.startswith("Реализация"),
         d.str.startswith("Возврат"),
         d.str.startswith("Поступление на р/с"),
         d.str.startswith("Приход в кассу")],
        ["service_fee", "sale", "refund", "payment_bank", "payment_cash"], default="other")
         
    # Достаем коды авиакомпаний, PNR и платежные документы
    m = d.str.extract(r"^(?:Реализация|Возврат)\s+(?!сервисный)([A-Z0-9]{2})\s+([A-Z0-9]{6})$")
    a["airline"] = m[0]
    a["pnr"] = m[1]
    sf = d.str.extract(r"сервисный сбор\s+([A-Z0-9]{6})$")[0]
    a["pnr"] = a["pnr"].fillna(sf)
    a["pay_doc"] = d.str.extract(r"(ЦБ-С\d+)")[0]

    a["debet"] = a["debet"].fillna(0.0)
    a["credit"] = a["credit"].fillna(0.0)
    a["debt_delta"] = (a["debet"] - a["credit"]).round(2)
    a["tickets10"] = a["ticket_cell"].map(tickets10)
    a["n_tickets"] = a["tickets10"].map(len)
    a["is_payment"] = a["line_type"].isin(["payment_bank", "payment_cash"])

    # Проверяем математический баланс внутри актов
    g = a.groupby(["subagent", "period"]).agg(s0=("saldo_start", "first"), s1=("saldo_end", "first"),
                                              dd=("debt_delta", "sum"))
    bad = ((g.s0 + g.dd - g.s1).abs() > 0.02).sum()
    
    log.append(("acts: актов (субагент-месяц) с нарушением s_start+обороты=s_end", int(bad)))
    log.append(("acts: актов в итоге", len(g)))
    log.append(("acts: строк с нераспознанным типом", int((a.line_type == "other").sum())))
    log.append(("acts: продаж/возвратов без билетов", int((~a.is_payment & (a.n_tickets == 0)).sum())))
    
    log += [("acts: отклонено строк", int(audit.disposition.eq("rejected").sum())),
            ("acts: принято строк", len(a))]
    a.attrs.update(audit=audit, issues=issues)
    return a, pd.DataFrame(log, columns=["check", "value"])


def act_headers(acts) -> pd.DataFrame:
    """Собирает сводные шапки актов по субагентам и месяцам."""
    h = (acts.groupby(["subagent", "period"])
         .agg(period_start=("period_start", "first"), period_end=("period_end", "first"),
              saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first"),
              act_status=("act_status", "first"), n_lines=("doc", "size"),
              debet=("debet", "sum"), credit=("credit", "sum"))
         .reset_index())
    return h


def act_continuity(headers) -> pd.DataFrame:
    """Проверяет непрерывность сальдо от месяца к месяцу (нет ли разрывов)."""
    h = headers.sort_values(["subagent", "period_start"]).copy()
    h["prev_period"] = h.groupby("subagent")["period"].shift()
    h["prev_saldo_end"] = h.groupby("subagent")["saldo_end"].shift()
    exp_prev = h["period_start"].dt.to_period("M") - 1
    h["is_consecutive"] = h["prev_period"].eq(exp_prev.astype(str))
    h["gap"] = (h["saldo_start"] - h["prev_saldo_end"]).round(2)
    h["act_missing_before"] = h["prev_period"].notna() & ~h["is_consecutive"]
    out = h[(h["gap"].abs() > 0.01) | h["act_missing_before"]]
    return out[["subagent", "period", "prev_period", "prev_saldo_end", "saldo_start", "gap", "act_missing_before"]]


# ================================================================
# ЧАСТЬ 3. ОБРАБОТКА ТРАНЗАКЦИЙ ETM (etm.csv)
# ================================================================

_KIND = {"выкуп": "purchase", "возврат": "refund", "войд": "void", "оплата": "payment"}


def load_etm(path, subagents=None):
    """Загружает транзакции ETM, приводит типы, проверяет билеты и знаки сумм."""
    raw = read_checked(path, "etm")
    log = [("etm: строк в файле", len(raw))]
    e, audit, issues = prepare(raw, "etm", ("amount", "amount_kgs", "balance_after"), ("date",))
    
    for c in ("amount", "amount_kgs", "balance_after"):
        e[c] = _num(e[c])
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["period"] = e["date"].dt.to_period("M").astype(str)
    e["kind"] = e["kind"].str.strip().str.lower()
    e["kind_en"] = e["kind"].map(_KIND)
    e["creator_en"] = e["creator"].map({"etm-bot": "bot", "субагент": "subagent"})
    # Используем subagent_key из normalize.py здесь тоже
    e["agent_key"] = e["agent"].map(subagent_key)

    if subagents is not None:
        key2name = {subagent_key(s): s for s in subagents}
        mapped = e["agent_key"].map(key2name)
        log.append(("etm: строк без соответствия субагенту 1С", int(mapped.isna().sum())))
        e["subagent"] = mapped.fillna(e["agent"])
    else:
        e["subagent"] = e["agent"]

    e["tickets13"] = e["tickets"].map(tickets13)
    e["n_tickets"] = e["tickets13"].map(len)
    e["tickets10"] = e["tickets13"].map(lambda L: [t[-10:] for t in L])
    e["airline_code3"] = e["tickets13"].map(lambda L: L[0][:3] if L else None)
    
    # Сверяем билеты в колонке и в комментарии
    e["tickets13_comment"] = e["comment"].map(tickets13)
    e["tickets_col_vs_comment_ok"] = [(sorted(a) == sorted(b)) or not a for a, b in zip(e.tickets13, e.tickets13_comment)]
    
    log.append(("etm: строк, где билеты в колонке и в комментарии расходятся", int((~e.tickets_col_vs_comment_ok).sum())))
    log.append(("etm: не-оплат без билетов", int(((e.kind_en != "payment") & (e.n_tickets == 0)).sum())))

    # Извлекаем данные из комментариев (пассажиры, маршруты, авиакомпании, PNR)
    c = e["comment"].fillna("").str.replace(r"\s+", " ", regex=True)
    e["pax"] = c.str.extract(r"([A-Z]{2,}/[A-Z]{2,}(?:\s?\+\d)?)")[0]
    e["route"] = c.str.extract(r"\b([A-Z]{3}-[A-Z]{3})\b")[0]
    no_tk = c.str.replace(r"\d{3}-?\d{10}", " ", regex=True).str.replace(r"\d{13}", " ", regex=True)
    e["airline"] = no_tk.str.extract(r"^([A-Z0-9]{2}) [A-Z0-9]{6} (?:выкуп|возврат|войд)")[0]
    e["pnr"] = no_tk.str.extract(r"(?<![A-Z0-9/-])((?=[A-Z0-9]*[A-Z])[A-Z0-9]{6})(?![A-Z0-9/-])")[0]
    e["pay_doc"] = c.str.extract(r"(ЦБ-С\d+)")[0]

    e["deposit_delta"] = e["amount_kgs"]
    e["debt_delta"] = -e["amount_kgs"]
    
    # Проверяем правильность знаков суммы (выкуп должен уменьшать баланс, возврат — увеличивать)
    e["sign_ok"] = np.select([e.kind_en.eq("purchase"), e.kind_en.eq("void")],
                              [e.amount_kgs.lt(0), e.amount_kgs.ge(0)], default=e.amount_kgs.gt(0))
    log.append(("etm: операций с неверным знаком", int((~e.sign_ok).sum())))

    e["fx_rate"] = np.where(e.currency != "KGS", (e.amount_kgs / e.amount).round(4), 1.0)
    e["is_foreign"] = e.currency != "KGS"
    log.append(("etm: дубликаты txn_id", int(e.txn_id.duplicated().sum())))

    # Проверяем цепочку остатков на счетах ETM
    e = e.sort_values(["agreement_id", "date", "txn_id"]).reset_index(drop=True)
    e["prev_balance"] = e.groupby("agreement_id")["balance_after"].shift()
    e["chain_gap"] = (e["prev_balance"] + e["amount_kgs"] - e["balance_after"]).round(2)
    e["chain_break"] = e["chain_gap"].abs().gt(0.01) & e["prev_balance"].notna()
    
    log.append(("etm: разрывов цепочки остатков", int(e.chain_break.sum())))
    log += [("etm: отклонено строк", int(audit.disposition.eq("rejected").sum())),
            ("etm: принято строк", len(e))]
    e.attrs.update(audit=audit, issues=issues)
    return e, pd.DataFrame(log, columns=["check", "value"])


def etm_month_end_balance(etm) -> pd.DataFrame:
    """Считает конечный баланс ETM на конец каждого месяца."""
    last = (etm.sort_values(["agreement_id", "date", "txn_id"])
            .groupby(["subagent", "agreement_id", "period"]).tail(1))
    last = last[["subagent", "agreement_id", "period", "balance_after"]]
    return last.groupby(["subagent", "period"]).balance_after.sum().rename("etm_balance_end").reset_index()


# ================================================================
# ЧАСТЬ 4. ЗАПУСК И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ================================================================

if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data_2_final")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "out/clean")
    
    # Создаем основную папку и подпапку для сверки
    out.mkdir(parents=True, exist_ok=True)
    reconciliation_folder = out / "reconciliation_ready"
    reconciliation_folder.mkdir(exist_ok=True)

    print("[INFO] Загрузка и очистка данных...")
    acts, la = load_acts(src / "acts.csv")
    etm, le = load_etm(src / "etm.csv", subagents=acts.subagent.unique())
    audit = pd.concat([acts.attrs["audit"].assign(source="acts"),
                       etm.attrs["audit"].assign(source="etm")], ignore_index=True)
    issues = pd.concat([acts.attrs["issues"], etm.attrs["issues"]], ignore_index=True)
    audit.to_csv(out / "source_row_audit.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(out / "data_quality_issues.csv", index=False, encoding="utf-8-sig")
    names = pd.concat([key_candidates(acts.attrs["audit"], "folder", "acts"),
                       key_candidates(etm.attrs["audit"], "agent", "etm")], ignore_index=True)
    names["review_required"] = names.groupby("normalized_key").original_name.transform("nunique").gt(1)
    names.to_csv(out / "subagent_name_map.csv", index=False, encoding="utf-8-sig")
    if audit.disposition.eq("rejected").any():
        raise ValidationError(f"Есть отклоненные финансовые строки. Исправьте данные: {out / 'data_quality_issues.csv'}")
    # Диагностика уже записана; не переносим большие исходные таблицы в attrs экспортов.
    acts.attrs.clear()
    etm.attrs.clear()
    hdr = act_headers(acts)
    cont = act_continuity(hdr)
    be = etm_month_end_balance(etm)

    def j(L): return " ".join(L)
    
    def save(df, name):
        df.to_csv(out / name, index=False, encoding="utf-8-sig")
        
    def save_for_reconciliation(df, name):
        df.to_csv(reconciliation_folder / name, index=False, encoding="utf-8-sig")

    a = acts.copy(); a["tickets10"] = a.tickets10.map(j)
    e = etm.copy()
    for c in ("tickets13", "tickets10", "tickets13_comment"): e[c] = e[c].map(j)
    
    # Самые главные файлы для сверки сохраняем в отдельную подпапку reconciliation_ready
    save_for_reconciliation(a, "acts_clean.csv")
    save_for_reconciliation(e, "etm_clean.csv")
    
    # Остальные вспомогательные отчеты и файлы ошибок сохраняем в общую папку
    save(split_money(acts[acts.n_tickets > 0], "tickets10", "debt_delta").rename(columns={"tickets10": "ticket10"})
         [["subagent", "period", "date", "line_type", "doc", "pnr", "airline", "ticket10", "debt_delta"]],
         "acts_tickets_long.csv")
    save(split_money(etm[etm.n_tickets > 0], "tickets13", "amount_kgs").assign(ticket10=lambda d: d.tickets13.str[-10:])
         [["subagent", "agreement_id", "txn_id", "date", "period", "kind_en", "creator_en", "pnr", "tickets13", "ticket10",
            "n_tickets", "amount_kgs", "currency", "amount", "fx_rate"]].rename(columns={"tickets13": "ticket13"}),
         "etm_tickets_long.csv")
    save(hdr, "act_headers.csv")
    save(cont, "act_continuity_gaps.csv")
    save(be, "etm_month_end_balance.csv")
    save(etm[~etm.sign_ok], "etm_wrong_sign.csv")
    save(etm[etm.chain_break][["subagent", "agreement_id", "txn_id", "date", "kind_en", "creator_en", "amount_kgs",
                               "prev_balance", "balance_after", "chain_gap"]], "etm_chain_breaks.csv")
    
    log = pd.concat([la, le])
    save(log, "cleaning_log.csv")
    print(log.to_string(index=False))
    print(f"\n[ГОТОВО] Основные файлы для сверки сохранены в: {reconciliation_folder.resolve()}")
