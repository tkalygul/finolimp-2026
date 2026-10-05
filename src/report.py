"""P3: Сборка итогового Excel-отчёта для бухгалтера.

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

# Листы от P4 и P5: название листа -> файл в папке interim.
# Лист попадёт в отчёт, только если файл уже существует.
EXTRA_SHEETS = {}

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
MONEY_TITLES = ("Сумма строки, сом",)

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")
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
# ЧАСТЬ 3. ОФОРМЛЕНИЕ И СБОРКА ФАЙЛА
# ================================================================

def write_sheet(writer, name, df, startrow=0, table=True):
    """Пишет таблицу на лист и оформляет шапку, ширину колонок и формат сумм."""
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
            if title in MONEY_TITLES:
                cell.number_format = "#,##0.00"
    if table:
        ws.freeze_panes = ws.cell(row=startrow + 2, column=1)
        ws.auto_filter.ref = f"A{startrow + 1}:{get_column_letter(len(df.columns))}{startrow + 1 + len(df)}"
    return ws


def add_extra_sheets(writer, interim_dir):
    """Добавляет листы P4 и P5, если их файлы уже лежат в interim."""
    for sheet, file in EXTRA_SHEETS.items():
        path = Path(interim_dir) / file
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")
        write_sheet(writer, sheet, df)


def build_report(interim_dir, out_path) -> Path:
    """Собирает Excel-отчёт из результатов загрузки реестра и листов P4 и P5."""
    data = read_registry_outputs(interim_dir)
    metrics_df, reasons_df = summary_tables(data["text"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # На листе сводки две таблицы: показатели и причины ошибок
        write_sheet(writer, "Сводка реестра", metrics_df, table=False)
        write_sheet(writer, "Сводка реестра", reasons_df, startrow=len(metrics_df) + 3, table=False)
        write_sheet(writer, "Проблемные строки", rejects_table(data["rejects"]))
        write_sheet(writer, "Корпоративные", corporate_table(data["non_subagents"]))
        write_sheet(writer, "Дубли реестра", duplicates_table(data["registry"]))
        add_extra_sheets(writer, interim_dir)
    return out


# ================================================================
# ЧАСТЬ 4. ЗАПУСК
# ================================================================

if __name__ == "__main__":
    interim = sys.argv[1] if len(sys.argv) > 1 else "interim"
    target = sys.argv[2] if len(sys.argv) > 2 else "report/reconciliation_report.xlsx"
    print(f"[Report] Отчёт сохранён: {build_report(interim, target)}")
