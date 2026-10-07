"""Тесты загрузчика реестра (P3).

Как запустить:
    pytest tests/test_registry_parser.py -q
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from src.load_registry import (
    SOURCE_COLUMNS, InputFileError, find_input_file, load_registry, resolve_party,
)
from src.normalize import ParseError, normalize_pax, parse_number, parse_pay, subagent_key, tickets10

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "registry_cases.csv"
REAL_DATA = REPO / "data_2_final"

# Названия субагентов как в etm.csv, включая похожие друг на друга
ETM_AGENT_NAMES = [
    "Blue Bird Travel", "Jailoo Trip", "Karakol Express", "SkyWay Travel", "Steppe Travel",
    "Абдылда уулу Б.", "Абдыраманова Д. А.", "Ак-Куу Авиа", "ОсОО Ала-Арча Авиа",
    "Ала-Арча Авиакасса", "Ала-Арча Тревел", "ОсОО Ала-Арча Трип", "Асанов Кубат Чынаревич",
    "Байтур Трип", "Джолдошова Салтанат Сатаревна", "ИП Джолдошов Д. У.", "Джолдош уулу Руслан",
    "Джолдош уулу Эмир", "ИССЫК ТРИП", "Кадырова Айнура Жумабекевна", "ИП Мамбетова Ч. Э.",
    "Мамбет уулу Чынгыз", "Манас Вояж", "Манас Тревел", "Манас Тур", "Манас Туризм",
    "МУРАС ТРИП", "САЛАМ ЭЙР", "Сатарова А. О.", "Сатарова Жылдыз Жумабековна",
    "Сатар уулу Санжар", "Сыдыков С. У.", "ТАЛАС ЛАЙН", "ИП Турсунов Т. А.",
    "Урматова Айпери Нурбековна", "Урматов Эрлан Сагынович", "ОсОО Чолпон Авиакасса",
    "ЧОЛПОН ЭЙР", "Ыйык Тур", "ЫЙЫК ЭКСПРЕСС", "Эсенбеков Кубат Элдияревич",
]
ETM_KEYS = {subagent_key(n) for n in ETM_AGENT_NAMES}
# Билеты «Джолдош уулу» в ETM, билетов строки T21 там нет
TICKET_AGENTS = {
    "2559099203": {"джолдошуулуэмир"},
    "2460735407": {"джолдошуулуруслан"},
    "2461883333": {"джолдошуулуэмир"},
    "2461883334": {"джолдошуулуэмир"},
}


# --- Функции нормализации ---

@pytest.mark.parametrize("name, key", [
    ("ОсОО «Ак-Жол Логистик»", "ак-жоллогистик"),
    ('ОсОО "Ыйык Тур"', "ыйыктур"),
    ("Ыйык Тур  ", "ыйыктур"),
    ("ИП Турсунов Т. А.", "турсуновт.а."),
    ("Турсунов Т.А.", "турсуновт.а."),
    ("SkyWayTravel", "skywaytravel"),
    ("SKYWAY TRAVEL", "skywaytravel"),
    ("салам эйр", "саламэйр"),
    ("Сёмин Пётр", "семинпетр"),
    ("ЗАО «Кыргыз Телеком Сервис»", "кыргызтелекомсервис"),
    ("Ипак Жолу", "ипакжолу"),  # ИП без пробела это часть имени
    ("", ""),
])
def test_subagent_key(name, key):
    assert subagent_key(name) == key


def test_p2_uses_same_subagent_key():
    """cleaning.py (P2) должен строить ключ той же функцией, иначе таблицы не склеятся."""
    import cleaning
    assert cleaning.subagent_key is subagent_key


@pytest.mark.parametrize("empty", [None, float("nan"), pd.NA, ""])
def test_missing_values(empty):
    """None, NaN и pd.NA ведут себя как пустая ячейка: ни выдуманных пассажиров, ни неверной причины."""
    assert normalize_pax(empty) == []
    with pytest.raises(ParseError) as t:
        tickets10(empty)
    assert t.value.reason == "no_tickets"
    with pytest.raises(ParseError) as p:
        parse_pay(empty)
    assert p.value.reason == "no_amount"


@pytest.mark.parametrize("cell, expected", [
    ("1412416941125", ["2416941125"]),
    ("141-2416941125", ["2416941125"]),
    ("2416941125", ["2416941125"]),
    ("376-2459531798 3762459531800", ["2459531798", "2459531800"]),
    ("733-2552671079; 2552671082", ["2552671079", "2552671082"]),
    ("958-2460735248 / 9582460735251", ["2460735248", "2460735251"]),
    ("2420846744\n8452420846747\n2420846750", ["2420846744", "2420846747", "2420846750"]),
    ("376-2459531798|3762459531800", ["2459531798", "2459531800"]),
])
def test_tickets10(cell, expected):
    assert tickets10(cell) == expected


@pytest.mark.parametrize("cell, reason", [
    ("", "no_tickets"),
    ("  \n ", "no_tickets"),
    ("14124169411", "bad_ticket_format"),       # 11 цифр
    ("14124169411255", "bad_ticket_format"),    # 14 цифр
    ("1412416941125 abc", "bad_ticket_format"),
    ("1412416941125 2416941125", "duplicate_ticket_in_cell"),
])
def test_tickets10_errors(cell, reason):
    with pytest.raises(ParseError) as err:
        tickets10(cell)
    assert err.value.reason == reason


@pytest.mark.parametrize("text, value", [
    ("7.889,70", 7889.70), ("115 877,00", 115877.0), ("15 666.00", 15666.0),
    ("-24.782,00", -24782.0), ("- 279.760,00", -279760.0), ("20062", 20062.0),
    ("625.5", 625.5), ("1,234.56", 1234.56), ("1.234.567", 1234567.0), ("0,5", 0.5),
    ("1 234,50", 1234.5),
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text, reason", [
    ("12.345", "ambiguous_number"),   # тысячи или дробь
    ("12,345", "ambiguous_number"),
    ("1,2,3", "bad_number"),
    ("12 34,00", "bad_number"),       # группа не из 3 цифр
    ("1.23,45", "bad_number"),
    ("abc", "bad_number"),
    ("", "bad_number"),
])
def test_parse_number_errors(text, reason):
    with pytest.raises(ParseError) as err:
        parse_number(text)
    assert err.value.reason == reason


def test_parse_pay_full():
    p = parse_pay("7.889,70 USD + 35usd sf")
    assert (p.amount, p.currency, p.fee, p.fee_currency, p.penalty_pct) == (7889.70, "USD", 35.0, "USD", None)


def test_parse_pay_refund_with_penalty():
    p = parse_pay("-340,75 EUR (штраф 15%)")
    assert (p.amount, p.currency, p.fee, p.penalty_pct) == (-340.75, "EUR", None, 15.0)


@pytest.mark.parametrize("cell, reason", [
    ("3762459527703", "ticket_in_amount"),
    ("376-2459527703", "ticket_in_amount"),
    ("2459527703", "ticket_in_amount"),
    ("12345", "no_currency"),
    ("12.345 kgs", "ambiguous_number"),
    ("100 GBP", "unknown_currency"),
    ("100 usdt", "bad_amount_format"),
    ("100 kgs + 5 kgs", "bad_fee"),          # без sf
    ("100 kgs + 5 sf", "bad_fee"),           # без валюты сбора
    ("100 kgs + 5 kgs sf + 1 kgs sf", "bad_amount_format"),
    ("100 kgs (скидка 5%)", "bad_amount_format"),
    ("", "no_amount"),
    ("kgs", "bad_amount_format"),
])
def test_parse_pay_errors(cell, reason):
    with pytest.raises(ParseError) as err:
        parse_pay(cell)
    assert err.value.reason == reason


def test_normalize_pax():
    assert normalize_pax("taalayov/bakyt,  OROZBEKOV/SANZHAR ") == ["TAALAYOV/BAKYT", "OROZBEKOV/SANZHAR"]
    assert normalize_pax("") == []


# --- Контрагенты ---

CORP = {"фондмурас"}


@pytest.mark.parametrize("party, tickets, expected", [
    ("Манас Тур", [], ("subagent", "манастур", "exact", None)),
    ("Сатар уулу", [], ("subagent", "сатаруулусанжар", "prefix", None)),
    ("Фонд «Мурас»", [], ("corporate", None, "corporate", None)),
    ("Мурас", [], ("unresolved", None, "none", "party_unknown")),          # короче порога
    ("Манас", [], ("unresolved", None, "none", "party_unknown")),
    ("Манас Т", ["0000000000"], ("unresolved", None, "none", "party_ambiguous")),
    ("Джолдош уулу", ["2559099203"], ("subagent", "джолдошуулуэмир", "prefix_ticket", None)),
    ("Джолдош уулу", ["0000000000"], ("unresolved", None, "none", "party_ambiguous")),
    ("Совсем Новый Агент", [], ("unresolved", None, "none", "party_unknown")),
])
def test_resolve_party(party, tickets, expected):
    assert resolve_party(subagent_key(party), tickets, ETM_KEYS, CORP, TICKET_AGENTS) == expected


# --- Реальные строки из реестра ---

def _write_registry(folder: Path, rows: pd.DataFrame, name="registry.csv") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    rows[SOURCE_COLUMNS].to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def cases():
    return pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def loaded(cases, tmp_path_factory):
    folder = tmp_path_factory.mktemp("fixture_data")
    _write_registry(folder, cases)
    return load_registry(folder, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)


def test_fixture_has_50_plus_real_rows(cases):
    assert len(cases) >= 50
    assert cases["src_row"].is_unique


def _num(text):
    return float(text) if text != "" else math.nan


def _assert_num(actual, expected_text, what, case_id):
    expected = _num(expected_text)
    if math.isnan(expected):
        assert actual is None or pd.isna(actual), f"{case_id}: {what} должно быть пусто, а {actual}"
    else:
        assert actual == pytest.approx(expected, abs=1e-6), f"{case_id}: {what}"


def test_each_real_row(cases, loaded):
    reg = loaded.registry
    for i, case in cases.iterrows():
        cid = case["case_id"]
        out = reg[reg["row_id"] == f"registry:{i + 2}"].sort_values("ticket_seq")
        expected_tickets = case["exp_tickets"].split()
        assert out["ticket10"].tolist() == expected_tickets, cid
        assert (out["n_tickets_in_row"] == len(expected_tickets)).all(), cid
        first = out.iloc[0]
        assert first["parse_status"] == case["exp_status"], cid
        assert first["op_type"] == case["exp_op_type"], cid
        _assert_num(first["amount_orig"], case["exp_amount_orig"], "amount_orig", cid)
        assert (first["currency"] if pd.notna(first["currency"]) else "") == case["exp_currency"], cid
        _assert_num(first["fee_orig"], case["exp_fee"], "fee_orig", cid)
        assert (first["fee_currency"] if pd.notna(first["fee_currency"]) else "") == case["exp_fee_currency"], cid
        _assert_num(first["penalty_pct"], case["exp_penalty_pct"], "penalty_pct", cid)
        for amount in out["amount_kgs"]:
            _assert_num(amount, case["exp_amount_kgs"], "amount_kgs", cid)
        assert first["party_type"] == case["exp_party_type"], cid
        assert (first["subagent_id"] if pd.notna(first["subagent_id"]) else "") == case["exp_subagent_id"], cid
        assert first["party_match"] == case["exp_party_match"], cid
        assert bool(first["is_dup_extra"]) == (case["exp_is_dup_extra"] == "True"), cid
        assert bool(first["pax_count_mismatch"]) == (case["exp_pax_count_mismatch"] == "True"), cid


def test_every_problem_row_is_listed(cases, loaded):
    bad_ids = {f"registry:{i + 2}" for i, c in cases.iterrows() if c["exp_status"] != "ok"}
    assert set(loaded.rejects["row_id"]) == bad_ids
    assert (loaded.rejects["parse_status"] != "ok").all()


def test_corporate_rows_listed_separately(cases, loaded):
    corp_ids = {f"registry:{i + 2}" for i, c in cases.iterrows() if c["exp_party_type"] == "corporate"}
    assert set(loaded.non_subagents["row_id"]) == corp_ids
    assert loaded.registry.loc[loaded.registry.party_type == "corporate", "subagent_id"].isna().all()


def test_duplicates_are_flagged_not_dropped(cases, loaded):
    reg = loaded.registry
    extra = reg[reg.is_dup_extra]
    assert len(extra) > 0
    # У каждой копии должен остаться оригинал
    assert set(extra["dup_group"]) <= set(reg["row_id"])


def test_one_row_per_ticket(cases, loaded):
    total = sum(len(t.split()) for t in cases["exp_tickets"])
    assert len(loaded.registry) == total
    assert not loaded.registry.duplicated(["row_id", "ticket_seq"]).any()


def test_signs_follow_etm_logic(loaded):
    reg = loaded.registry.dropna(subset=["amount_kgs"])
    assert (reg.loc[reg.op_type == "sale", "amount_kgs"] < 0).all()
    assert (reg.loc[reg.op_type.isin(["refund", "void"]), "amount_kgs"] > 0).all()


# --- Плохие строки и поиск файлов ---

def _row(**over):
    base = dict(date="01.01.2026", employee="r.sydykov", kind="продажа", party="Манас Тур",
                tickets="1412416941125", pax="ISAKOV/EMIR", pnr="ABC123", airline="FX",
                route="FRU-IST", pay_cell="1000 kgs", rate_usd="88.17", rate_eur="94.59",
                rate_rub="1.0", rate_kzt="0.171")
    base.update(over)
    return base


@pytest.mark.parametrize("over, reason", [
    (dict(date="2026-01-01"), "bad_date"),
    (dict(date="31.02.2026"), "bad_date"),
    (dict(kind="обмен"), "bad_kind"),
    (dict(tickets=""), "no_tickets"),
    (dict(pay_cell="-1000 kgs"), "sign_mismatch"),
    (dict(kind="возврат", pay_cell="1000 kgs"), "sign_mismatch"),
    (dict(kind="возврат", pay_cell="-1000 kgs + 100 kgs sf"), "fee_on_non_sale"),
    (dict(pay_cell="1000 kgs (штраф 10%)"), "penalty_on_sale"),
    (dict(pay_cell="10 usd", rate_usd="н/д"), "bad_rate"),
    (dict(pay_cell="10 usd", rate_usd="0"), "bad_rate"),
    (dict(party="Совсем Новый Агент"), "party_unknown"),
    (dict(pay_cell="12.345 kgs"), "ambiguous_number"),
])
def test_bad_rows_are_listed_with_reason(tmp_path, over, reason):
    _write_registry(tmp_path, pd.DataFrame([_row(), _row(**over)]))
    res = load_registry(tmp_path, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)
    assert res.stats["source_rows"] == 2
    assert res.rejects["row_id"].tolist() == ["registry:3"]
    assert reason in res.rejects["parse_status"].iloc[0].split(";")


def test_rate_is_taken_from_same_row(tmp_path):
    _write_registry(tmp_path, pd.DataFrame([_row(pay_cell="10 usd + 1 usd sf", rate_usd="90")]))
    res = load_registry(tmp_path, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)
    assert res.registry["amount_kgs"].iloc[0] == pytest.approx(-990.0)


def test_amount_is_split_between_tickets(tmp_path):
    _write_registry(tmp_path, pd.DataFrame([_row(tickets="1412416941125 1412416941126 1412416941127",
                                                 pay_cell="900 kgs + 300 kgs sf")]))
    res = load_registry(tmp_path, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)
    assert res.registry["amount_kgs"].tolist() == pytest.approx([-400.0] * 3)
    assert res.registry["amount_row_kgs"].tolist() == pytest.approx([-1200.0] * 3)


def test_find_input_file_by_mask(tmp_path):
    (tmp_path / "registry (2).csv").write_text("x", encoding="utf-8")
    (tmp_path / "etm (2).csv").write_text("x", encoding="utf-8")
    assert find_input_file(tmp_path, "registry*.csv").name == "registry (2).csv"
    (tmp_path / "Registry_old.CSV").write_text("x", encoding="utf-8")
    with pytest.raises(InputFileError, match="несколько"):
        find_input_file(tmp_path, "registry*.csv")
    with pytest.raises(InputFileError, match="нет файла"):
        find_input_file(tmp_path, "acts*.csv")


def test_cp1251_file_is_read(tmp_path):
    pd.DataFrame([_row()])[SOURCE_COLUMNS].to_csv(tmp_path / "registry.csv", index=False, encoding="cp1251")
    res = load_registry(tmp_path, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)
    assert res.registry["subagent_id"].iloc[0] == "манастур"


def test_missing_column_fails_loudly(tmp_path):
    _write_registry(tmp_path, pd.DataFrame([_row()]))
    df = pd.read_csv(tmp_path / "registry.csv", dtype=str).drop(columns=["rate_kzt"])
    df.to_csv(tmp_path / "registry.csv", index=False)
    with pytest.raises(Exception, match="rate_kzt"):
        load_registry(tmp_path, etm_keys=ETM_KEYS, ticket_agents=TICKET_AGENTS)


# --- Полный выданный набор (если он есть локально) ---

@pytest.mark.skipif(not (REAL_DATA / "registry.csv").exists(), reason="нет data_2_final")
def test_real_dataset_numbers():
    res = load_registry(REAL_DATA)
    s = res.stats
    assert s["source_rows"] == 12146
    assert s["output_rows"] == 19023
    assert s["ok_share"] >= 0.95
    assert s["ok_rows"] + s["reject_rows"] == s["source_rows"]
    assert s["corporate_rows"] == 423
    assert len(s["prefix_keys"]) == 19
    assert s["reasons"].get("ticket_in_amount") == 65
    joldosh = res.registry[res.registry.party_key == "джолдошуулу"]
    assert joldosh["row_id"].nunique() == 58
