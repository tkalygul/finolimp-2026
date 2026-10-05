"""Тесты Excel-отчёта (P3).

Как запустить:
    pytest tests/test_report.py -q
"""
import pandas as pd
import pytest
from openpyxl import load_workbook

import src.report as report
from src.load_registry import load_registry, write_outputs
from src.report import build_report, parse_load_report, row_number
from tests.test_registry_parser import ETM_KEYS, FIXTURE, TICKET_AGENTS, _write_registry


@pytest.fixture(scope="module")
def interim(tmp_path_factory):
    """Папка interim, собранная загрузчиком из тестовых строк реестра."""
    cases = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    data = tmp_path_factory.mktemp("report_data")
    _write_registry(data, cases)
    result = load_registry(data, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)
    folder = tmp_path_factory.mktemp("interim")
    write_outputs(result, folder)
    return folder, result


@pytest.fixture(scope="module")
def workbook(interim, tmp_path_factory):
    folder, _ = interim
    path = build_report(folder, tmp_path_factory.mktemp("out") / "report.xlsx")
    return load_workbook(path)


def test_row_number():
    assert row_number("registry:5") == 5


def test_parse_load_report():
    text = "Файл: registry.csv\nИсходных строк: 12\nБез ошибок: 10 (83.33%)\nПричины:\n  ok: 10\n  bad_date: 2\n"
    metrics, reasons = parse_load_report(text)
    assert metrics == [("Файл", "registry.csv"), ("Исходных строк", 12), ("Без ошибок", "10 (83.33%)")]
    assert reasons == [("ok", 10), ("bad_date", 2)]


def test_sheets_in_order(workbook):
    assert workbook.sheetnames == ["Сводка реестра", "Проблемные строки", "Корпоративные", "Дубли реестра"]


def test_summary_has_metrics_and_reasons(workbook, interim):
    _, result = interim
    ws = workbook["Сводка реестра"]
    rows = [[c.value for c in row] for row in ws.iter_rows()]
    assert ["Исходных строк", result.stats["source_rows"]] in [r[:2] for r in rows]
    assert any(r[0] == "ok" and r[1] == "Строка разобрана без ошибок" for r in rows)


def test_rejects_sheet_matches_loader(workbook, interim):
    _, result = interim
    ws = workbook["Проблемные строки"]
    numbers = [row[0].value for row in ws.iter_rows(min_row=2)]
    assert numbers == sorted(row_number(r) for r in result.rejects["row_id"])
    header = [c.value for c in ws[1]]
    assert header[:4] == ["Строка в реестре", "Код ошибки", "Что не так", "Подробности"]


def test_rejects_have_readable_reason(workbook):
    ws = workbook["Проблемные строки"]
    for row in ws.iter_rows(min_row=2):
        assert row[2].value and row[2].value != row[1].value


def test_corporate_sheet_matches_loader(workbook, interim):
    _, result = interim
    ws = workbook["Корпоративные"]
    assert ws.max_row - 1 == len(result.non_subagents)


def test_duplicates_sheet_matches_loader(workbook, interim):
    _, result = interim
    ws = workbook["Дубли реестра"]
    extra = [row[2].value for row in ws.iter_rows(min_row=2)].count("да")
    assert extra == result.stats["dup_extra_rows"]
    assert ws.max_row - 1 == result.stats["dup_rows"]


def test_header_style_and_filter(workbook):
    ws = workbook["Проблемные строки"]
    assert ws["A1"].font.bold
    assert ws.freeze_panes == "A2"
    assert ws.auto_filter.ref.startswith("A1:")


def test_missing_files_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="registry.parquet"):
        build_report(tmp_path, tmp_path / "report.xlsx")


def test_extra_sheet_added_only_if_file_exists(interim, tmp_path, monkeypatch):
    folder, _ = interim
    monkeypatch.setattr(report, "EXTRA_SHEETS", {"Баланс": "balance.csv", "Аномалии": "anomalies.csv"})
    (folder / "balance.csv").write_text("subagent_id,diff\nманастур,100\n", encoding="utf-8-sig")
    try:
        wb = load_workbook(build_report(folder, tmp_path / "report.xlsx"))
    finally:
        (folder / "balance.csv").unlink()
    assert "Баланс" in wb.sheetnames
    assert "Аномалии" not in wb.sheetnames
    assert [c.value for c in wb["Баланс"][2]] == ["манастур", 100]
