"""Запуск без дополнительных зависимостей: python -m unittest discover -s tests -p test_stage1_quality.py"""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.quality import prepare, split_money
from src.schema import validate, ValidationError
from cleaning import load_acts, load_etm


class QualityTests(unittest.TestCase):
    def act(self, **updates):
        row = dict(folder="ИП Тест", doc="Реализация", source_row=2,
                   period_start="2026-01-01", period_end="2026-01-31", date="2026-01-15",
                   act_status="", saldo_start="100", saldo_end="100", debet="", credit="")
        row.update(updates)
        return pd.DataFrame([row])

    def prepare_act(self, **updates):
        return prepare(self.act(**updates), "acts", ["saldo_start", "saldo_end", "debet", "credit"],
                       ["period_start", "period_end", "date"], ["debet", "credit"])

    def test_blank_turnover_is_zero(self):
        data, audit, issues = self.prepare_act()
        self.assertEqual(data.debet.iloc[0], 0)
        self.assertEqual(len(issues), 0)
        self.assertEqual(audit.disposition.iloc[0], "accepted")

    def test_invalid_amount_is_rejected_with_original_value(self):
        data, audit, issues = self.prepare_act(debet="ошибка")
        self.assertTrue(data.empty)
        self.assertEqual(audit.disposition.iloc[0], "rejected")
        self.assertEqual(issues.raw_value.iloc[0], "ошибка")

    def test_required_balance_cannot_be_blank(self):
        self.assertTrue(self.prepare_act(saldo_start="")[0].empty)

    def test_invalid_and_outside_dates(self):
        for date in ["2026-02-30", "2026-02-01", "wrong"]:
            with self.subTest(date=date):
                self.assertTrue(self.prepare_act(date=date)[0].empty)

    def test_unknown_status_is_rejected(self):
        self.assertTrue(self.prepare_act(act_status="новый")[0].empty)

    def test_money_split_preserves_positive_and_negative_cents(self):
        for amount in [100.01, -100.01, 0.01, -0.01]:
            frame = pd.DataFrame({"tickets": [["1", "2", "3"]], "amount": [amount]})
            split = split_money(frame, "tickets", "amount")
            self.assertAlmostEqual(split.amount.sum(), amount)
            self.assertTrue(all(abs(x * 100 - round(x * 100)) < 1e-8 for x in split.amount))

    def test_schema_requires_fields_used_by_cleaning(self):
        with self.assertRaises(ValidationError):
            validate(pd.DataFrame({"date": ["2026-01-01"]}), "acts")

    def test_conflicting_act_headers_are_not_selected_arbitrarily(self):
        rows = pd.concat([self.act(saldo_end="100"), self.act(saldo_end="200")], ignore_index=True)
        rows["invoice"] = "СЧ-1"
        rows["ticket_cell"] = "1234567890"
        rows["pax"] = "TEST/NAME"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "acts.csv"
            rows.drop(columns="source_row").to_csv(path, index=False)
            data, _ = load_acts(path)
            self.assertTrue(data.empty)
            self.assertTrue(data.attrs["audit"].disposition.eq("rejected").all())

    def test_unknown_etm_agent_is_preserved_and_zero_void_is_valid(self):
        row = dict(agent="Новый агент", agreement_id="AGR-1", txn_id="123", date="2026-01-01 12:00:00",
                   kind="войд", tickets="1231234567890", amount="0", currency="KGS",
                   amount_kgs="0", balance_after="100", creator="etm-bot", comment="")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "etm.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            data, log = load_etm(path, ["Другой агент"])
            self.assertEqual(data.subagent.iloc[0], "Новый агент")
            self.assertTrue(data.sign_ok.iloc[0])
            self.assertEqual(int(log.loc[log.check.eq("etm: строк без соответствия субагенту 1С"), "value"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
