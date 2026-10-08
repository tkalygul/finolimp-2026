import unittest
import pandas as pd
import json
from src.matching import prepare_components,match_components
from src.classification import classify_groups,balance_cases


class ClassificationTests(unittest.TestCase):
    def test_matched_bot_payment_does_not_require_registry(self):
        from tests.test_stage2_matching import MatchingTests
        helper=MatchingTests()
        a=pd.DataFrame([helper.act(-100,line_type='payment',tickets10='',pay_doc='P1')])
        e=pd.DataFrame([helper.etm(-100,kind_en='payment',tickets10='',pay_doc='P1',creator_en='bot')])
        r=pd.DataFrame(columns=['party_type','subagent_id','ticket10'])
        m,c,_=match_components(prepare_components(a,e,r))
        self.assertEqual(m.registry_status.iloc[0],'not_applicable_payment')
        self.assertFalse(m.requires_review.iloc[0])
        result=classify_groups(m,c).iloc[0]
        self.assertEqual(result.error_type,'ok_payment')
        self.assertEqual(result.error_owner,'none')

    def classify(self,a=100,e=100,r=100,creator="bot",status=None,repeat=False):
        m=pd.DataFrame([dict(match_id="M1",subagent_id="a",ticket10="123",op_type="sale",
            amount_1c=a,amount_etm=e,amount_registry=r,match_status=status or "amount_difference",
            period_mismatch=False,registry_status="present" if pd.notna(r) else ("missing_for_bot" if creator=="bot" else "not_expected_self_service"),
            review_reason="repeated_events_etm" if repeat else "",documents_1c='["DOC"]',documents_etm='["100"]')])
        rows=[]
        for source,amount in [("1c",a),("etm",e),("registry",r)]:
            if pd.notna(amount):
                rows.append(dict(match_id="M1",source=source,amount=amount,period="2026-01",date="2026-01-15",
                    creator=creator if source=="etm" else "employee",parse_status="ok",
                    original_kind="purchase" if source=="etm" else "sale",document="100"))
        return classify_groups(m,pd.DataFrame(rows)).iloc[0]

    def test_dependent_sources_do_not_prove_1c_error(self):
        r=self.classify(a=120,e=100,r=100)
        self.assertEqual(r.error_owner,"unclear")
        self.assertEqual(r.error_type,"dependent_sources_agree")

    def test_bot_amount_is_hypothesis(self):
        r=self.classify(a=100,e=120,r=100)
        self.assertEqual(r.error_owner,"bot")
        self.assertEqual(r.confidence,"medium")
        self.assertEqual(r.assignment_status,"hypothesis")
        self.assertTrue(pd.isna(r.proposed_correction))

    def test_agent_amount_is_hypothesis(self):
        self.assertEqual(self.classify(a=100,e=100,r=120).error_owner,"agent")

    def test_bot_missing_registry_is_not_self_service(self):
        r=self.classify(r=float("nan"),status="matched")
        self.assertEqual(r.error_type,"registry_missing")
        self.assertEqual(r.error_owner,"unclear")

    def test_real_self_service_can_be_ok(self):
        r=self.classify(r=float("nan"),creator="subagent",status="matched")
        self.assertEqual(r.error_type,"ok_self_service")
        self.assertEqual(r.is_error,0)

    def test_registry_only_does_not_blame_bot(self):
        r=self.classify(a=float("nan"),e=float("nan"),status="only_registry")
        self.assertEqual(r.error_owner,"unclear")

    def test_repeated_events_are_not_confirmed_duplicate(self):
        r=self.classify(e=200,repeat=True,status="ambiguous")
        self.assertEqual(r.error_type,"duplicate_suspected")
        self.assertEqual(r.error_owner,"unclear")
        self.assertIn("amount_difference",json.loads(r["flags"]))

    def test_balance_missing_data_has_unknown_amount(self):
        b=pd.DataFrame([dict(subagent_id="a",period="2026-01",bridge_status="missing_act",
            opening_difference=float("nan"),unexplained=float("nan"),check_1c=float("nan"),check_etm=0,act_carryover_gap=float("nan"))])
        cases=balance_cases(b)
        self.assertEqual(len(cases),1)
        self.assertTrue(pd.isna(cases.signed_balance_value.iloc[0]))


if __name__=="__main__": unittest.main()
