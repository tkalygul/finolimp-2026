import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.matching import prepare_components, match_components, run_matching


class MatchingTests(unittest.TestCase):
    def act(self, amount=100, **kw):
        row = dict(source_row=2, subagent_key="agent", tickets10="1234567890", line_type="sale",
                   period="2026-01", date="2026-01-15", debt_delta=amount, doc="sale", pay_doc="")
        row.update(kw)
        return row

    def etm(self, amount=100, **kw):
        row = dict(source_row=2, agent_key="agent", tickets10="1234567890", kind_en="purchase",
                   period="2026-01", date="2026-01-15", debt_delta=amount, txn_id="100", pay_doc="",
                   creator_en="subagent")
        row.update(kw)
        return row

    def run_match(self, acts, etm, reg=None):
        a = pd.DataFrame(acts) if acts else pd.DataFrame(columns=self.act().keys())
        e = pd.DataFrame(etm) if etm else pd.DataFrame(columns=self.etm().keys())
        r = pd.DataFrame(reg) if reg else pd.DataFrame(columns=["party_type", "subagent_id", "ticket10"])
        return match_components(prepare_components(a, e, r))

    def test_fee_and_sale_match_one_charge(self):
        m,c,_ = self.run_match([self.act(100), self.act(10,source_row=3,line_type="service_fee")], [self.etm(110)])
        self.assertEqual(len(m), 1)
        self.assertEqual(m.match_status.iloc[0], "matched")
        self.assertEqual(m.amount_1c.sum(), 110)
        self.assertEqual(len(c), 3)

    def test_duplicate_keys_never_multiply_amounts(self):
        m,c,_ = self.run_match([self.act(100),self.act(100,source_row=3)],
                             [self.etm(100),self.etm(100,source_row=3,txn_id="101")])
        self.assertEqual(m.amount_1c.sum(),200)
        self.assertEqual(m.amount_etm.sum(),200)
        self.assertEqual(m.match_status.iloc[0],"ambiguous")
        self.assertEqual(len(c),4)

    def test_month_boundary_preserves_both_periods(self):
        m,c,_ = self.run_match([self.act(date="2026-02-01",period="2026-02")],
                             [self.etm(date="2026-01-31")])
        self.assertEqual(m.match_status.iloc[0],"period_mismatch")
        self.assertEqual(set(c.period),{"2026-01","2026-02"})

    def test_void_links_to_purchase_without_losing_components(self):
        m,c,_ = self.run_match([], [self.etm(100),self.etm(-100,source_row=3,kind_en="void",txn_id="101")])
        self.assertEqual(m.match_status.iloc[0],"voided")
        self.assertEqual(m.amount_etm.sum(),0)
        self.assertEqual(len(c),2)

    def test_partial_void_requires_review(self):
        m,_,_=self.run_match([], [self.etm(100),self.etm(-90,source_row=3,kind_en="void",txn_id="101")])
        self.assertTrue(m.requires_review.iloc[0])
        self.assertIn("void_not_fully_reversed",m.review_reason.iloc[0])

    def test_refund_is_separate_from_sale(self):
        m,_,_=self.run_match([self.act(),self.act(-30,source_row=3,line_type="refund")],
                            [self.etm(),self.etm(-30,source_row=3,kind_en="refund",txn_id="101")])
        self.assertEqual(len(m),2)
        self.assertEqual(set(m.op_type),{"sale","refund"})

    def test_bot_missing_registry_is_flagged(self):
        m,_,_=self.run_match([self.act()],[self.etm(creator_en="bot")])
        self.assertEqual(m.registry_status.iloc[0],"missing_for_bot")
        self.assertTrue(m.requires_review.iloc[0])

    def test_payments_one_to_many_by_document(self):
        m,c,_=self.run_match([self.act(-100,line_type="payment_bank",tickets10="",pay_doc="PAY-1")],
            [self.etm(-60,kind_en="payment",tickets10="",pay_doc="PAY-1"),
             self.etm(-40,source_row=3,txn_id="101",kind_en="payment",tickets10="",pay_doc="PAY-1")])
        self.assertEqual(len(m),1)
        self.assertEqual(m.match_status.iloc[0],"matched")
        self.assertEqual(len(c),3)

    def test_competing_payments_are_not_guessed(self):
        m,_,_=self.run_match([self.act(-100,line_type="payment_bank",tickets10="")],
            [self.etm(-100,kind_en="payment",tickets10=""),
             self.etm(-100,source_row=3,txn_id="101",kind_en="payment",tickets10="")])
        self.assertEqual(len(m),3)
        self.assertTrue(m.match_status.eq("ambiguous").all())

    def test_unknown_operations_are_accounted_for(self):
        m,c,_=self.run_match([self.act(15,line_type="other",tickets10="")],[])
        self.assertEqual(m.amount_1c.sum(),15)
        self.assertTrue(m.requires_review.iloc[0])
        self.assertEqual(len(c),1)

    def test_multiticket_allocations_preserve_totals(self):
        m,c,checks=self.run_match([self.act(100.01,tickets10="1234567890 1234567891")],
                                 [self.etm(100.01,tickets10="1234567890 1234567891")])
        self.assertAlmostEqual(m.amount_1c.sum(),100.01)
        self.assertAlmostEqual(m.amount_etm.sum(),100.01)
        self.assertFalse(c.duplicated(["source","allocation_id"]).any())
        self.assertTrue(checks.difference.dropna().abs().lt(1e-6).all())

    def test_registry_only_is_retained(self):
        reg=dict(party_type="subagent",subagent_id="agent",ticket10="1234567890",op_type="sale",
                 row_id="registry:2",ticket_seq=1,ts="2026-01-15",amount_kgs=-100,created_by="user",parse_status="ok")
        m,c,_=self.run_match([],[],[reg])
        self.assertEqual(m.match_status.iloc[0],"only_registry")
        self.assertEqual(m.amount_registry.iloc[0],100)
        self.assertEqual(len(c),1)

    def test_file_entry_point_preserves_ticket_strings_and_outputs_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            ready=root/"clean"/"reconciliation_ready"
            ready.mkdir(parents=True)
            pd.DataFrame([self.act(tickets10="0123456789")]).to_csv(ready/"acts_clean.csv",index=False)
            pd.DataFrame([self.etm(tickets10="0123456789")]).to_csv(ready/"etm_clean.csv",index=False)
            empty=pd.DataFrame(columns=["party_type","subagent_id","ticket10"])
            with patch("src.matching.pd.read_parquet",return_value=empty):
                result=run_matching(root/"clean",root/"registry.parquet",root/"out")
            self.assertEqual(result["matches"].ticket10.iloc[0],"0123456789")
            self.assertTrue((root/"out"/"matching_checks.csv").exists())
            self.assertEqual(result["matches"].match_status.iloc[0],"matched")


if __name__ == "__main__":
    unittest.main()
