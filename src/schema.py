import pandas as pd

# Обязательные колонки единой схемы данных
REQUIRED_COLUMNS = [
    "source",          # acts / etm / registry
    "row_id",          # исходный файл + номер строки
    "subagent_id",     # единый ID субагента
    "period",          # ГГГГ-ММ
    "ts",              # дата или дата со временем
    "op_type",         # продажа / возврат / войд / оплата / обмен
    "ticket10",        # 10-значный номер билета
    "pnr",             # код брони
    "passenger",       # нормализованное имя пассажира
    "amount_orig",     # сумма в исходной валюте
    "currency",        # валюта
    "amount_kgs",      # сумма в сомах со знаком (плюс = депозит)
    "created_by",      # автор / агент
    "balance_after"    # остаток ETM после операции (только для etm)
]

def validate(df: pd.DataFrame) -> bool:
    """Проверяет соответствие датафрейма единой схеме данных."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"В датафрейме отсутствуют обязательные колонки: {missing_cols}")
    return True
