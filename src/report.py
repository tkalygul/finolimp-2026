"""P3: Сборка итогового Excel-отчёта для бухгалтера.

Листы:
    Сводка, По субагентам, Ошибки по типам, Расхождения по билетам, Оплаты с ошибкой, Аномалии,
    Сотрудники реестра, Справочник ошибок  — результаты P5 (interim/p5), если P5 запускали
    Мост баланса, Сверка P4 по субагентам  — результаты P4 (interim/p4), только исправленной версии
    Сводка реестра, Проблемные строки, Корпоративные, Дубли реестра  — разбор реестра (P3), всегда

Как запустить:
    python -m src.report <папка_interim> <файл_отчёта.xlsx>
Пример:
    python -m src.report interim report/reconciliation_report.xlsx
"""
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.classify_errors import ISSUE_RU, OWNER_RU

# Результаты P5 в папке interim. P5 пишет их все вместе, поэтому нужны либо все, либо ни одного.
P5_FILES = {
    "tickets": "p5/p5_ticket_classified.csv",
    "payments": "p5/p5_payments.csv",
    "anomalies": "p5/p5_anomalies.csv",
    "subagent": "p5/p5_summary_subagent.csv",
    "type": "p5/p5_summary_type.csv",
    "employee": "p5/p5_summary_employee.csv",
    "error_types": "p5/p5_error_types.csv",
}

# Результаты P4 и колонка, которая есть только в исправленной версии P4.
# Старая версия P4 (слияние «многие ко многим», без моста) в отчёт не попадает.
P4_FILES = {
    "bridge": ("p4/p4_balance_bridge.csv", "unexplained"),
    "summary": ("p4/p4_subagent_summary.csv", "unexplained_total"),
}

# Причины в мосте баланса: колонка P4 -> заголовок
BRIDGE_CAUSES = {
    "ops_amount_difference": "Сумма отличается, сом",
    "ops_only_1c": "Операция только в 1С, сом",
    "ops_only_etm": "Операция только в ETM, сом",
    "ops_period_mismatch": "Операция в другом месяце, сом",
    "payments_only_1c": "Оплата только в 1С, сом",
    "payments_only_etm": "Оплата только в ETM, сом",
    "payments_period_mismatch": "Оплата в другом месяце, сом",
}
# Сошедшиеся операции, войды и оплаты дают только копейки округления
BRIDGE_ROUNDING = ["ops_matched", "ops_voided", "payments_matched"]
BRIDGE_NOTE = ("Разница = сальдо 1С + баланс ETM (в ETM долг субагента со знаком минус). "
               "Разница на начало + причины + округления = разница на конец. «Не объяснено» должно быть 0.")

ANOMALY_RU = {
    "etm_duplicate_op": "Бот повторил операцию (те же билеты, та же сумма)",
    "etm_double_credit": "Оплата зачислена в ETM дважды",
    "etm_wrong_sign": "Операция в ETM с неверным знаком",
    "etm_balance_chain_break": "Остаток ETM после операции не сходится с предыдущим",
    "registry_duplicate_row": "Лишняя копия строки в реестре",
    "registry_parse_problem": "Строка реестра разобрана с ошибкой",
    "registry_pax_mismatch": "В строке реестра пассажиров не столько, сколько билетов",
    "1c_act_missing": "Нет акта 1С за месяц, а в ETM есть операции",
    "1c_saldo_carryover": "Сальдо 1С на начало не равно сальдо на конец прошлого месяца",
    "1c_saldo_transfer_suspect": "Такой же разрыв сальдо у другого субагента — похоже, сальдо перенесли не тому",
    "corporate_as_subagent": "Корпоративный клиент записан как субагент",
}
LEVEL_RU = {"high": "высокая", "medium": "средняя", "low": "низкая"}
LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2}
YES_NO = {True: "да", False: "нет"}

# Понятные описания кодов ошибок для бухгалтера
REASON_TEXT = {
    "ok": "Строка разобрана без ошибок",
    "empty_row": "Пустая строка",
    "bad_date": "Дата не в формате ДД.ММ.ГГГГ",
    "bad_kind": "Неизвестный вид операции",
    "no_tickets": "Нет номеров билетов",
    "bad_ticket_format": "Номер билета в неверном формате",
    "duplicate_ticket_in_cell": "Один билет указан дважды в строке",
    "no_amount": "Нет суммы",
    "no_currency": "В сумме не указана валюта",
    "bad_amount_format": "Сумма в непонятном формате",
    "ambiguous_number": "Неоднозначное число (тысячи или дробь)",
    "bad_number": "Некорректное число в сумме",
    "unknown_currency": "Неизвестная валюта",
    "bad_fee": "Некорректный сервисный сбор",
    "bad_rate": "Нет или неверный курс валюты",
    "sign_mismatch": "Знак суммы не соответствует виду операции",
    "fee_on_non_sale": "Сервисный сбор указан у возврата или войда",
    "penalty_on_sale": "Штраф указан у продажи",
    "ticket_in_amount": "В ячейке суммы стоит номер билета",
    "party_empty": "Не указан контрагент",
    "party_unknown": "Контрагент не найден среди субагентов ETM",
    "party_ambiguous": "Обрезанное название подходит нескольким субагентам",
}

# Заголовки колонок реестра на русском
SOURCE_TITLES = {
    "date": "Дата", "employee": "Сотрудник", "kind": "Операция", "party": "Контрагент",
    "tickets": "Билеты", "pax": "Пассажиры", "pnr": "PNR", "airline": "Авиакомпания",
    "route": "Маршрут", "pay_cell": "Сумма (как в реестре)", "rate_usd": "Курс USD",
    "rate_eur": "Курс EUR", "rate_rub": "Курс RUB", "rate_kzt": "Курс KZT",
}
OP_TITLES = {"sale": "продажа", "refund": "возврат", "void": "войд"}
# Формат колонки задаётся окончанием заголовка: «..., сом» — сумма, «..., %» — доля
MONEY_SUFFIX = ", сом"
PERCENT_SUFFIX = ", %"

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")
TITLE_FONT = Font(bold=True, size=12)
# Строки, на которые бухгалтеру стоит посмотреть в первую очередь
WARN_FILL = PatternFill("solid", fgColor="FCE4D6")
MAX_COLUMN_WIDTH = 50


# ================================================================
# ЧАСТЬ 1. ЧТЕНИЕ РЕЗУЛЬТАТОВ ЗАГРУЗЧИКА
# ================================================================

def read_registry_outputs(interim_dir) -> dict:
    """Читает файлы, которые сохраняет write_outputs в load_registry.py."""
    folder = Path(interim_dir)
    paths = {
        "registry": folder / "registry.parquet",
        "rejects": folder / "registry_rejects.csv",
        "non_subagents": folder / "registry_non_subagents.csv",
        "report": folder / "registry_load_report.txt",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"В {folder} нет файлов реестра: {', '.join(missing)}")
    csv = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return {
        "registry": pd.read_parquet(paths["registry"]),
        "rejects": pd.read_csv(paths["rejects"], **csv),
        "non_subagents": pd.read_csv(paths["non_subagents"], **csv),
        "text": paths["report"].read_text(encoding="utf-8"),
    }


def row_number(row_id) -> int:
    """Берёт номер строки реестра из row_id (registry:5 -> 5), он совпадает с номером в Excel."""
    return int(str(row_id).split(":")[1])


def parse_load_report(text) -> tuple:
    """Разбирает текстовый отчёт загрузки на показатели и счётчики причин."""
    metrics, reasons = [], []
    in_reasons = False
    for line in text.splitlines():
        if line.startswith("Причины:"):
            in_reasons = True
            continue
        key, _, value = line.strip().partition(": ")
        if not key:
            continue
        if value.isdigit():
            value = int(value)
        (reasons if in_reasons else metrics).append((key, value))
    return metrics, reasons


def read_p5_outputs(interim_dir) -> dict:
    """Читает результаты P5. Если P5 ещё не запускали, возвращает пустой словарь."""
    folder = Path(interim_dir)
    paths = {k: folder / f for k, f in P5_FILES.items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if len(missing) == len(paths):
        return {}
    if missing:
        raise FileNotFoundError(f"В {folder / 'p5'} не хватает файлов P5: {', '.join(missing)}")
    dtypes = {"ticket10": str, "txn_ids": str, "created_by": str, "ref": str}
    return {k: pd.read_csv(p, dtype=dtypes, encoding="utf-8-sig", low_memory=False) for k, p in paths.items()}


def read_p4_outputs(interim_dir) -> dict:
    """Читает результаты P4, но только исправленной версии (с мостом баланса), иначе пустой словарь."""
    out = {}
    for key, (file, required) in P4_FILES.items():
        path = Path(interim_dir) / file
        if not path.exists():
            return {}
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if required not in df.columns:
            return {}
        out[key] = df
    return out


def subagent_names(interim_dir) -> dict:
    """Ключ субагента -> название: из актов 1С, а если там нет — из ETM."""
    ready = Path(interim_dir) / "clean" / "p4_ready"
    names = {}
    for file, key, name in (("etm_clean.csv", "agent_key", "agent"),
                            ("acts_clean.csv", "subagent_key", "subagent")):
        path = ready / file
        if path.exists():
            df = pd.read_csv(path, usecols=[key, name], dtype=str, encoding="utf-8-sig").dropna()
            names.update(zip(df[key], df[name]))
    return names


def _names(ids, names) -> pd.Series:
    """Названия субагентов вместо ключей; если названия нет, остаётся ключ."""
    return ids.map(names).fillna(ids)


def _date(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%d.%m.%Y")


# ================================================================
# ЧАСТЬ 2. ТАБЛИЦЫ ДЛЯ ЛИСТОВ
# ================================================================

def summary_tables(text) -> tuple:
    """Собирает таблицы показателей и причин ошибок для листа сводки."""
    metrics, reasons = parse_load_report(text)
    metrics_df = pd.DataFrame(metrics, columns=["Показатель", "Значение"])
    reasons_df = pd.DataFrame(
        [(code, REASON_TEXT.get(code, code), count) for code, count in reasons],
        columns=["Код", "Что это значит", "Строк"])
    # Сначала успешные строки, потом ошибки по убыванию
    reasons_df["_ok"] = reasons_df["Код"] != "ok"
    reasons_df = (reasons_df.sort_values(["_ok", "Строк"], ascending=[True, False])
                  .drop(columns="_ok").reset_index(drop=True))
    return metrics_df, reasons_df


def rejects_table(rejects) -> pd.DataFrame:
    """Готовит лист проблемных строк с описанием причин."""
    df = pd.DataFrame({
        "Строка в реестре": rejects["row_id"].map(row_number),
        "Код ошибки": rejects["parse_status"],
        "Что не так": rejects["parse_status"].map(
            lambda s: "; ".join(REASON_TEXT.get(c, c) for c in s.split(";"))),
        "Подробности": rejects["parse_detail"],
    })
    for col, title in SOURCE_TITLES.items():
        df[title] = rejects[col]
    return df.sort_values("Строка в реестре").reset_index(drop=True)


def corporate_table(non_subagents) -> pd.DataFrame:
    """Готовит лист корпоративных клиентов, они исключены из сверки субагентов."""
    df = pd.DataFrame({
        "Строка в реестре": non_subagents["row_id"].map(row_number),
        "Сумма строки, сом": pd.to_numeric(non_subagents["amount_row_kgs"], errors="coerce"),
    })
    for col, title in SOURCE_TITLES.items():
        df[title] = non_subagents[col]
    return df.sort_values("Строка в реестре").reset_index(drop=True)


def duplicates_table(registry) -> pd.DataFrame:
    """Готовит лист полных дублей строк реестра, по одной строке на запись."""
    dup = registry[registry["dup_group"].notna()]
    rows = dup.groupby("row_id", sort=False).agg(
        original=("dup_group", "first"), extra=("is_dup_extra", "first"),
        ts=("ts", "first"), op=("op_type", "first"), party=("party_raw", "first"),
        tickets=("ticket10", " ".join), amount=("amount_row_kgs", "first")).reset_index()
    df = pd.DataFrame({
        "Строка в реестре": rows["row_id"].map(row_number),
        "Оригинал (строка)": rows["original"].map(row_number),
        "Лишняя копия": rows["extra"].map({True: "да", False: "нет"}),
        "Дата": rows["ts"].dt.strftime("%d.%m.%Y"),
        "Операция": rows["op"].map(OP_TITLES),
        "Контрагент": rows["party"],
        "Билеты": rows["tickets"],
        "Сумма строки, сом": rows["amount"],
    })
    return df.sort_values(["Оригинал (строка)", "Строка в реестре"]).reset_index(drop=True)


# ================================================================
# ЧАСТЬ 3. ТАБЛИЦЫ P5: ОШИБКИ И АНОМАЛИИ
# ================================================================

def overview_blocks(p5, p4, names) -> list:
    """Таблицы листа «Сводка»: показатели, ошибки по виновникам, топ субагентов, частые сбои."""
    t, p, sub, emp = p5["tickets"], p5["payments"], p5["subagent"], p5["employee"]
    metrics = [
        ("Билетных операций (субагент + билет + продажа/возврат)", len(t)),
        ("из них с ошибкой", int(t["is_error"].sum())),
        ("Оплат", len(p)),
        ("из них с ошибкой", int(p["is_error"].sum())),
        ("Аномалий", len(p5["anomalies"])),
        ("Субагентов с частыми сбоями", int(sub["frequent_failures"].sum())),
        ("Сотрудников реестра с частыми сбоями", int(emp["frequent_failures"].sum())),
    ]
    if p4:
        b = p4["bridge"]
        metrics += [
            ("Субагенто-месяцев в мосте баланса", len(b)),
            ("из них с разницей 1С и ETM на конец месяца", int(b["closing_difference"].abs().gt(1).sum())),
            ("из них с необъяснённой разницей", int(b["unexplained"].abs().gt(1).sum())),
        ]
    blocks = [("Показатели", pd.DataFrame(metrics, columns=["Показатель", "Количество"]))]

    te, pe = t[t["is_error"] == 1], p[p["is_error"] == 1]
    counts = pd.DataFrame({"Ошибок в билетах": te.groupby("error_owner_ru").size(),
                           "Ошибок в оплатах": pe.groupby("error_owner_ru").size()}).fillna(0).astype(int)
    counts["Всего ошибок"] = counts.sum(axis=1)
    counts = counts.sort_values("Всего ошибок", ascending=False)
    counts.loc["Итого"] = counts.sum()
    amounts = pd.concat([te, pe]).groupby("error_owner_ru")["amount_at_stake"].apply(lambda s: s.abs().sum())
    amounts["Итого"] = amounts.sum()
    counts["Сумма под риском, сом"] = amounts.round(2)
    blocks.append(("Ошибки по виновникам", counts.rename_axis("Виновник").reset_index()))

    s = sub.assign(amount=sub[["amount_bot", "amount_agent", "amount_1c", "amount_unclear"]].sum(axis=1))
    top = s.nlargest(10, "amount")
    blocks.append(("Топ-10 субагентов по сумме под риском", pd.DataFrame({
        "Субагент": _names(top["subagent_id"], names),
        "Ошибок": top["errors"].astype(int),
        "Доля ошибок, %": top["error_rate"],
        "Сумма под риском, сом": top["amount"].round(2),
        "Частые сбои": top["frequent_failures"].map(YES_NO),
    })))

    freq = sub[sub["frequent_failures"]]
    if not freq.empty:
        blocks.append(("Субагенты с частыми сбоями (доля ошибок выше средней на 2σ, ошибок не меньше 5)",
                       pd.DataFrame({"Субагент": _names(freq["subagent_id"], names),
                                     "Операций": freq["operations"].astype(int),
                                     "Ошибок": freq["errors"].astype(int),
                                     "Доля ошибок, %": freq["error_rate"]})))
    freq = emp[emp["frequent_failures"]]
    if not freq.empty:
        blocks.append(("Сотрудники реестра с частыми сбоями", employee_table(freq)))
    return blocks


def subagent_table(sub, names) -> pd.DataFrame:
    """Ошибки по субагентам: сколько и на какую сумму у каждого виновника."""
    df = pd.DataFrame({
        "Субагент": _names(sub["subagent_id"], names),
        "Операций": sub["operations"].astype(int),
        "Ошибок": sub["errors"].astype(int),
        "Доля ошибок, %": sub["error_rate"],
        "Частые сбои": sub["frequent_failures"].map(YES_NO),
    })
    for owner, title in (("bot", "бот ETM"), ("agent", "агент"), ("1c", "1С"), ("unclear", "неясно")):
        df[f"Ошибок: {title}"] = sub[f"errors_{owner}"].astype(int)
        df[f"Сумма: {title}, сом"] = sub[f"amount_{owner}"].round(2)
    df["Аномалий"] = sub["anomalies"].astype(int) if "anomalies" in sub else 0
    return df.reset_index(drop=True)


def type_table(types) -> pd.DataFrame:
    return pd.DataFrame({
        "Виновник": types["error_owner_ru"],
        "Что случилось": types["error_type_ru"],
        "Ошибок": types["count"].astype(int),
        "Сумма, сом": types["amount"].round(2),
        "Субагентов": types["subagents"].astype(int),
        "Код ошибки": types["error_type"],
    }).reset_index(drop=True)


def ticket_errors_table(tickets, names) -> pd.DataFrame:
    """Билетные операции с ошибкой: суммы в трёх источниках, кто виноват и почему."""
    t = tickets[tickets["is_error"] == 1]
    df = pd.DataFrame({
        "Субагент": _names(t["subagent_id"], names),
        "Месяц": t["period"],
        "Билет": t["ticket10"],
        "Операция": t["grp"].map(OP_TITLES),
        "Сумма 1С, сом": t["amount_1c"].round(2),
        "Сумма ETM, сом": t["amount_etm"].round(2),
        "Сумма реестра, сом": t["amount_reg"].round(2),
        "Виновник": t["error_owner_ru"],
        "Что случилось": t["error_type_ru"],
        "Уточнение": t["issue"].map(ISSUE_RU),
        "Подробности": t["reason"],
        "Уверенность": t["confidence"].map(LEVEL_RU),
        "Сумма под риском, сом": t["amount_at_stake"].round(2),
        "Документы 1С": t["doc_1c"],
        "Транзакции ETM": t["txn_ids"],
        "Кто вносил в реестр": t["created_by"],
        "Код ошибки": t["error_type"],
    })
    return df.sort_values(["Субагент", "Месяц", "Билет"]).reset_index(drop=True)


def payment_errors_table(payments, names) -> pd.DataFrame:
    """Оплаты с ошибкой: не зачислены, зачислены дважды, не проведены в 1С."""
    p = payments[payments["is_error"] == 1]
    df = pd.DataFrame({
        "Субагент": _names(p["subagent_id"], names),
        "Месяц": p["period"],
        "Дата в 1С": _date(p["date_1c"]),
        "Документ 1С": p["doc"],
        "Сумма в 1С, сом": p["amount_1c"].round(2),
        "Дата в ETM": _date(p["date_etm"]),
        "Транзакция ETM": p["txn_id"].astype("Int64"),
        "Сумма в ETM, сом": p["amount_etm"].round(2),
        "Виновник": p["error_owner_ru"],
        "Что случилось": p["error_type_ru"],
        "Подробности": p["reason"],
        "Уверенность": p["confidence"].map(LEVEL_RU),
        "Сумма под риском, сом": p["amount_at_stake"].round(2),
        "Код ошибки": p["error_type"],
    })
    return df.sort_values(["Субагент", "Месяц"]).reset_index(drop=True)


def anomalies_table(anomalies, names) -> pd.DataFrame:
    """Аномалии: сначала высокая важность."""
    a = anomalies.assign(_order=anomalies["severity"].map(LEVEL_ORDER))
    a = a.sort_values(["_order", "anomaly_type", "subagent_id", "period"])
    return pd.DataFrame({
        "Важность": a["severity"].map(LEVEL_RU),
        "Что найдено": a["anomaly_type"].map(ANOMALY_RU).fillna(a["anomaly_type"]),
        "Виновник": a["error_owner"].map(OWNER_RU),
        "Субагент": _names(a["subagent_id"], names),
        "Месяц": a["period"],
        "Ссылка": a["ref"],
        "Сумма, сом": a["amount_kgs"].round(2),
        "Подробности": a["detail"],
        "Код": a["anomaly_type"],
    }).reset_index(drop=True)


def employee_table(emp) -> pd.DataFrame:
    return pd.DataFrame({
        "Сотрудник": emp["created_by"],
        "Строк в реестре": emp["rows"].astype(int),
        "Ошибок агента": emp["agent_errors"].astype(int),
        "Доля ошибок, %": emp["error_rate"],
        "Частые сбои": emp["frequent_failures"].map(YES_NO),
    }).reset_index(drop=True)


def error_catalog_table(types) -> pd.DataFrame:
    return pd.DataFrame({
        "Код": types["error_type"],
        "Виновник": types["error_owner_ru"],
        "Что это значит": types["description"],
    })


# ================================================================
# ЧАСТЬ 4. ТАБЛИЦЫ P4: МОСТ БАЛАНСА
# ================================================================

def bridge_table(bridge, names) -> pd.DataFrame:
    """Мост по субагенту и месяцу: разница на начало + причины = разница на конец."""
    b = bridge
    df = pd.DataFrame({
        "Субагент": _names(b["subagent_id"], names),
        "Месяц": b["period"],
        "Акт 1С есть": b["act_exists"].map(YES_NO),
        "Сальдо 1С на начало, сом": b["saldo_start"],
        "Баланс ETM на начало, сом": b["etm_balance_start"],
        "Разница на начало, сом": b["opening_difference"],
    })
    for col, title in BRIDGE_CAUSES.items():
        df[title] = b[col]
    df["Округления (до 1 сом), сом"] = b[BRIDGE_ROUNDING].sum(axis=1).round(2)
    df["Разница на конец, сом"] = b["closing_difference"]
    df["Не объяснено, сом"] = b["unexplained"]
    df["Сальдо 1С на конец, сом"] = b["saldo_end"]
    df["Баланс ETM на конец, сом"] = b["etm_balance_end"]
    return df.sort_values(["Субагент", "Месяц"]).reset_index(drop=True)


def p4_summary_table(summary, names) -> pd.DataFrame:
    """Сверка 1С и ETM по субагентам: сколько сошлось, чего не хватает, сколько не объяснено."""
    s = summary
    df = pd.DataFrame({
        "Субагент": _names(s["subagent_id"], names),
        "Операций": s["operations"],
        "Сошлось": s["matched"],
        "Войды": s["voided"],
        "Сумма отличается": s["amount_difference"],
        "Только в 1С": s["only_1c"],
        "Только в ETM": s["only_etm"],
        "В другом месяце": s["period_mismatch"],
        "Разница сумм, сом": s["amount_difference_sum"],
        "Только в 1С, сом": s["only_1c_amount"],
        "Только в ETM, сом": s["only_etm_amount"],
        "Оплат": s["payments"],
        "Оплата только в 1С": s["payments_only_1c"],
        "Оплата только в ETM": s["payments_only_etm"],
        "Разница на конец периода, сом": s["closing_difference_last"],
        "Не объяснено, сом": s["unexplained_total"],
        "Доля сошедшихся, %": s["matched_share"],
    })
    return df.sort_values("Субагент").reset_index(drop=True)


# ================================================================
# ЧАСТЬ 5. ОФОРМЛЕНИЕ И СБОРКА ФАЙЛА
# ================================================================

def write_sheet(writer, name, df, startrow=0, table=True, highlight=None):
    """Пишет таблицу на лист и оформляет шапку, ширину колонок и формат сумм.

    highlight — флаги по строкам таблицы: отмеченные строки заливаются цветом.
    """
    df.to_excel(writer, sheet_name=name, index=False, startrow=startrow)
    ws = writer.sheets[name]
    for cell in ws[startrow + 1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for col_idx, title in enumerate(df.columns, start=1):
        letter = get_column_letter(col_idx)
        values = [len(str(title))] + [len(str(v)) for v in df[title] if v is not None]
        width = min(max(values) + 2, MAX_COLUMN_WIDTH)
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, width)
        for cell in ws[letter][startrow + 1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if str(title).endswith(MONEY_SUFFIX):
                cell.number_format = "#,##0.00"
            elif str(title).endswith(PERCENT_SUFFIX):
                cell.number_format = "0.00%"
    if highlight is not None:
        for i, flag in enumerate(highlight):
            if flag:
                for cell in ws[startrow + 2 + i][:len(df.columns)]:
                    cell.fill = WARN_FILL
    if table:
        ws.freeze_panes = ws.cell(row=startrow + 2, column=1)
        ws.auto_filter.ref = f"A{startrow + 1}:{get_column_letter(len(df.columns))}{startrow + 1 + len(df)}"
    return ws


def write_blocks(writer, name, blocks):
    """Пишет на один лист несколько таблиц друг под другом, у каждой — заголовок."""
    row = 0
    for title, df in blocks:
        write_sheet(writer, name, df, startrow=row + 1, table=False)
        writer.sheets[name].cell(row=row + 1, column=1, value=title).font = TITLE_FONT
        row += len(df) + 4


def build_report(interim_dir, out_path) -> Path:
    """Собирает Excel-отчёт: разбор реестра, а также листы P5 и P4, если они уже посчитаны."""
    data = read_registry_outputs(interim_dir)
    metrics_df, reasons_df = summary_tables(data["text"])
    p5 = read_p5_outputs(interim_dir)
    p4 = read_p4_outputs(interim_dir)
    names = subagent_names(interim_dir)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        if p5:
            write_blocks(writer, "Сводка", overview_blocks(p5, p4, names))
            sub = p5["subagent"]
            write_sheet(writer, "По субагентам", subagent_table(sub, names),
                        highlight=sub["frequent_failures"].tolist())
            write_sheet(writer, "Ошибки по типам", type_table(p5["type"]))
            write_sheet(writer, "Расхождения по билетам", ticket_errors_table(p5["tickets"], names))
            write_sheet(writer, "Оплаты с ошибкой", payment_errors_table(p5["payments"], names))
            write_sheet(writer, "Аномалии", anomalies_table(p5["anomalies"], names))
        if p4:
            bridge = bridge_table(p4["bridge"], names)
            write_sheet(writer, "Мост баланса", bridge, startrow=2,
                        highlight=bridge["Не объяснено, сом"].abs().gt(1).tolist())
            writer.sheets["Мост баланса"]["A1"] = BRIDGE_NOTE
            write_sheet(writer, "Сверка P4 по субагентам", p4_summary_table(p4["summary"], names))
        if p5:
            emp = p5["employee"]
            write_sheet(writer, "Сотрудники реестра", employee_table(emp),
                        highlight=emp["frequent_failures"].tolist())
        # На листе сводки реестра две таблицы: показатели и причины ошибок
        write_sheet(writer, "Сводка реестра", metrics_df, table=False)
        write_sheet(writer, "Сводка реестра", reasons_df, startrow=len(metrics_df) + 3, table=False)
        write_sheet(writer, "Проблемные строки", rejects_table(data["rejects"]))
        write_sheet(writer, "Корпоративные", corporate_table(data["non_subagents"]))
        write_sheet(writer, "Дубли реестра", duplicates_table(data["registry"]))
        if p5:
            write_sheet(writer, "Справочник ошибок", error_catalog_table(p5["error_types"]))
    return out


# ================================================================
# ЧАСТЬ 6. ЗАПУСК
# ================================================================

if __name__ == "__main__":
    interim = sys.argv[1] if len(sys.argv) > 1 else "interim"
    target = sys.argv[2] if len(sys.argv) > 2 else "report/reconciliation_report.xlsx"
    print(f"[Report] Отчёт сохранён: {build_report(interim, target)}")
