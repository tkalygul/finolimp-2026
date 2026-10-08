import pandas as pd


class ValidationError(Exception):
    """Кастомное исключение для ошибок валидации данных."""
    pass


# Эталонные схемы обязательных колонок для каждого файла проекта
REQUIRED_COLUMNS = {
    "acts": [
        "folder", "period_start", "period_end", "act_status", "date", "doc",
        "invoice", "ticket_cell", "pax", "debet", "credit", "saldo_start", "saldo_end"
    ],
    "etm": [
        "date", "txn_id", "amount", "amount_kgs", "balance_after", "agent",
        "agreement_id", "kind", "tickets", "currency", "creator", "comment"
    ],
    "registry": [
        "date", "employee", "kind", "party", "tickets", "pax", "pnr", "airline",
        "route", "pay_cell", "rate_usd", "rate_eur", "rate_rub", "rate_kzt"
    ]
}


def validate(df: pd.DataFrame, dataset_type: str = "acts") -> pd.DataFrame:
    """
    Проверяет DataFrame на соответствие схеме:
    1. Проверяет, что таблица не пустая.
    2. Проверяет наличие обязательных колонок.
    
    :param df: pandas DataFrame для проверки
    :param dataset_type: тип датасета ('acts', 'etm', 'registry')
    :return: исходный DataFrame, если всё прошло успешно
    """
    if df is None or df.empty:
        raise ValidationError(f"[Schema] Ошибка: Датасет '{dataset_type}' пустой или не был загружен!")

    if dataset_type not in REQUIRED_COLUMNS:
        raise ValidationError(f"[Schema] Ошибка: Неизвестный тип датасета '{dataset_type}'.")

    expected_cols = REQUIRED_COLUMNS[dataset_type]
    if df.columns.duplicated().any():
        raise ValidationError(f"[Schema] Повторяющиеся колонки в '{dataset_type}'.")
    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        raise ValidationError(
            f"[Schema] Ошибка в '{dataset_type}': отсутствуют обязательные колонки -> {missing_cols}. "
            f"Текущие колонки в файле: {list(df.columns)}"
        )

    print(f"[Schema] Датасет '{dataset_type}' успешно прошел валидацию. Строк: {len(df)}")
    return df
