import unittest
import pandas as pd
from src.balance import build_balance, build_balance_summary


class BalanceTests(unittest.TestCase):
    def act(self, period="2026-01", start=-500, end=-400, amount=100):
        return dict(subagent_key="a", period=period, saldo_start=start, saldo_end=end, debt_delta=amount)

    def etm(self, period="2026-01", amount=-100, balance=400, agreement="AGR-1", txn="1"):
        return dict(agent_key="a",agreement_id=agreement,txn_id=txn,period=period,
                    date=period+"-15",amount_kgs=amount,balance_after=balance)

    def calculate(self, acts, etm, status="matched", months=None, missing_component=False):
        a=pd.DataFrame(acts)
        e=pd.DataFrame(etm)
        m=pd.DataFrame([dict(match_id="M1",subagent_id="a",op_type="sale",match_status=status,
            requires_review=status!="matched",review_reason="",difference_1c_etm=0)])
        components=[]
        for source, rows in [("1c",acts),("etm",etm)]:
            for i,row in enumerate(rows):
                components.append(dict(match_id="M1",subagent_id="a",source=source,
                    allocation_id=f"{source}:{i}",period=row["period"],
                    amount=row["debt_delta"] if source=="1c" else -row["amount_kgs"]))
        c=pd.DataFrame(components)
        if missing_component:
            c=c[c.source.eq("1c")]
        b,d,g=build_balance(a,e,m,c)
        return b,d,g,m

    def test_nonzero_opening_is_inferred(self):
        b,_,_,_=self.calculate([self.act()],[self.etm()])
        self.assertEqual(b.etm_balance_start.iloc[0],500)
        self.assertAlmostEqual(b.unexplained.iloc[0],0)

    def test_multiple_agreements_are_summed(self):
        b,_,g,_=self.calculate([self.act(start=-700,end=-600)],
            [self.etm(),self.etm(amount=0,balance=200,agreement="AGR-2",txn="2")])
        self.assertEqual(b.etm_balance_start.iloc[0],700)
        self.assertEqual(b.etm_balance_end.iloc[0],600)
        self.assertEqual(len(g),2)

    def test_inactive_month_carries_balance(self):
        b,_,_,_=self.calculate([self.act(),self.act("2026-02",-400,-400,0)],[self.etm()])
        self.assertEqual(b.etm_balance_end.tolist(),[400,400])
        self.assertAlmostEqual(b.unexplained.iloc[1],0)

    def test_missing_act_stays_unknown(self):
        b,_,_,m=self.calculate([self.act()],[self.etm(),self.etm("2026-02",amount=0,balance=400,txn="2")])
        self.assertEqual(b.bridge_status.iloc[1],"missing_act")
        self.assertTrue(pd.isna(b.unexplained.iloc[1]))
        s=build_balance_summary(m,b)
        self.assertTrue(pd.isna(s.unexplained_total.iloc[0]))

    def test_later_first_agreement_is_unknown_before_observation(self):
        b,_,_,_=self.calculate([self.act(),self.act("2026-02",-500,-400,100)],
                             [self.etm("2026-02")])
        self.assertFalse(b.etm_known.iloc[0])
        self.assertTrue(pd.isna(b.etm_balance_start.iloc[0]))

    def test_cross_month_contributions_reverse(self):
        b,d,_,_=self.calculate([self.act(amount=0,end=-500),self.act("2026-02",-500,-400,100)],
                             [self.etm()],status="period_mismatch")
        self.assertEqual(b.ops_period_mismatch.tolist(),[-100,100])
        self.assertTrue(b.unexplained.abs().lt(1e-6).all())
        self.assertEqual(b.closing_difference.tolist(),[-100,0])

    def test_missing_component_is_explicit_residual(self):
        b,_,_,_=self.calculate([self.act()],[self.etm()],missing_component=True)
        self.assertEqual(b.component_gap.iloc[0],-100)
        self.assertEqual(b.unexplained.iloc[0],-100)
        self.assertEqual(b.bridge_status.iloc[0],"unexplained_residual")

    def test_broken_balance_chain_not_hidden(self):
        b,_,_,_=self.calculate([self.act(amount=200,end=-300)],
            [self.etm(),self.etm(amount=-100,balance=350,txn="2")])
        self.assertEqual(b.check_etm.iloc[0],-50)
        self.assertEqual(b.unexplained.iloc[0],50)

    def test_ambiguous_cause_can_be_numerically_accounted_for(self):
        b,_,_,_=self.calculate([self.act()],[self.etm()],status="ambiguous")
        self.assertEqual(b.ops_ambiguous.iloc[0],0)
        self.assertEqual(b.bridge_status.iloc[0],"reconciled")

    def test_registry_does_not_add_to_balance(self):
        a=pd.DataFrame([self.act()])
        e=pd.DataFrame([self.etm()])
        m=pd.DataFrame([dict(match_id="M1",subagent_id="a",op_type="sale",match_status="ambiguous",requires_review=True,review_reason="registry_amount_difference")])
        c=pd.DataFrame([dict(match_id="M1",subagent_id="a",source=source,allocation_id=source+":1",period="2026-01",amount=value)
                        for source,value in [("1c",100),("etm",100),("registry",1000000000)]])
        b,_,_=build_balance(a,e,m,c)
        self.assertEqual(b.explained_turnover_difference.iloc[0],0)

    def test_void_across_months_preserves_cancellation(self):
        b,_,_,_=self.calculate([self.act(amount=0,end=-500),self.act("2026-02",-500,-500,0)],
            [self.etm(),self.etm("2026-02",amount=100,balance=500,txn="2")],status="voided")
        self.assertEqual(b.ops_voided.tolist(),[-100,100])
        self.assertTrue(b.unexplained.abs().lt(1e-6).all())

    def test_act_carryover_gap_is_separate(self):
        b,_,_,_=self.calculate([self.act(),self.act("2026-02",-300,-300,0)],[self.etm()])
        self.assertEqual(b.act_carryover_gap.iloc[1],100)
        self.assertEqual(b.unexplained.iloc[1],0)

    def test_missing_etm_is_not_zero(self):
        a=pd.DataFrame([self.act()])
        e=pd.DataFrame(columns=self.etm().keys())
        m=pd.DataFrame([dict(match_id="M1",subagent_id="a",op_type="sale",match_status="only_1c",requires_review=True,review_reason="")])
        c=pd.DataFrame([dict(match_id="M1",subagent_id="a",source="1c",allocation_id="1c:1",period="2026-01",amount=100)])
        b,_,_=build_balance(a,e,m,c)
        self.assertTrue(pd.isna(b.etm_balance_end.iloc[0]))
        self.assertEqual(b.bridge_status.iloc[0],"unknown_etm_balance")


if __name__=="__main__":
    unittest.main()
