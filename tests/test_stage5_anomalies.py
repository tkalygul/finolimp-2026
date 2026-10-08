import unittest
import pandas as pd
from src.anomalies import detect_canonical_anomalies, employee_concentration


class AnomalyTests(unittest.TestCase):
    def component(self, rid, **changes):
        row=dict(source='etm', source_row_id=rid, match_id='m1', date='2026-01-01',
                 period='2026-01', document=rid, payment_document='', creator='bot',
                 ticket10='1234567890', amount=100., original_kind='purchase', subagent_id='a')
        row.update(changes)
        return row

    def classified(self, **changes):
        row=dict(match_id='m1', op_type='sale', error_owner='unclear', confidence='low', is_error=1)
        row.update(changes)
        return pd.DataFrame([row])

    def detect(self, rows):
        return detect_canonical_anomalies(self.classified(), pd.DataFrame(rows))[0]

    def test_repeat_is_suspected_and_preserves_amount(self):
        result=self.detect([self.component('e1', amount=50), self.component('e1', amount=50,ticket10='2345678901'),
                            self.component('e2',amount=50),self.component('e2',amount=50,ticket10='2345678901')])
        self.assertEqual(len(result),1)
        self.assertEqual(result.amount_kgs.iloc[0],100)
        self.assertEqual(result.error_owner.iloc[0],'unclear')
        self.assertEqual(result.status.iloc[0],'suspected')
        self.assertIn('e1',result.source_refs.iloc[0])

    def test_distinct_payment_documents(self):
        result=self.detect([self.component('e1',original_kind='payment',payment_document='p1'),
                            self.component('e2',original_kind='payment',payment_document='p2')])
        self.assertTrue(result.empty)

    def test_repeated_payment_document(self):
        result=self.detect([self.component('e1',original_kind='payment',payment_document='p1'),
                            self.component('e2',original_kind='payment',payment_document='p1')])
        self.assertEqual(result.anomaly_type.iloc[0],'etm_double_credit_suspected')

    def test_separate_agents_and_dates(self):
        self.assertTrue(self.detect([self.component('e1'),self.component('e2',subagent_id='b')]).empty)
        self.assertTrue(self.detect([self.component('e1'),self.component('e2',date='2026-01-20')]).empty)

    def test_unknown_dates_do_not_prove_repeat(self):
        self.assertTrue(self.detect([self.component('e1',date=None),self.component('e2',date=None)]).empty)

    def void_rows(self, credit=None, recent=False):
        rows=[self.component('e1'),self.component('r1',source='registry',original_kind='void',amount=-100,
              date='2026-01-19' if recent else '2026-01-02',creator='employee'),
              self.component('end',match_id='end',date='2026-01-20',amount=1,ticket10='9999999999')]
        if credit is not None: rows.append(self.component('v1',original_kind='void',amount=-credit,date='2026-01-10'))
        return rows

    def test_void_without_credit(self):
        result=self.detect(self.void_rows())
        self.assertEqual(result.anomaly_type.iloc[0],'void_without_credit')
        self.assertEqual(result.amount_kgs.iloc[0],100)

    def test_recent_void_pending(self):
        self.assertEqual(self.detect(self.void_rows(recent=True)).status.iloc[0],'pending')

    def test_partial_void(self):
        result=self.detect(self.void_rows(60))
        self.assertEqual(result.anomaly_type.iloc[0],'void_partial_credit')
        self.assertEqual(result.amount_kgs.iloc[0],40)

    def test_delayed_full_void_not_loss(self):
        result=self.detect(self.void_rows(100))
        self.assertEqual(result.status.iloc[0],'informational')
        self.assertEqual(result.amount_kgs.iloc[0],0)

    def test_original_rows_counted_once(self):
        rows=[self.component('r1',source='registry',creator='alice'),
              self.component('r1',source='registry',creator='alice',ticket10='2345678901')]
        result=employee_concentration(self.classified(error_owner='agent',confidence='high'),pd.DataFrame(rows))
        self.assertEqual(result.rows.iloc[0],1)
        self.assertEqual(result.agent_errors.iloc[0],1)
        self.assertFalse(result.frequent_failures.iloc[0])

    def test_shared_case_not_assigned_to_employee(self):
        rows=[self.component('r1',source='registry',creator='alice'),self.component('r2',source='registry',creator='bob')]
        result=employee_concentration(self.classified(error_owner='agent',confidence='high'),pd.DataFrame(rows))
        self.assertEqual(result.agent_errors.sum(),0)
        self.assertEqual(result.shared_review_rows.sum(),2)

    def test_supplementary_registry_signal_deduplicates_tickets(self):
        rows=pd.DataFrame([self.component('r1',source='registry',creator='alice',amount=50),
                           self.component('r1',source='registry',creator='alice',amount=50,ticket10='2345678901')])
        extras=pd.DataFrame([dict(anomaly_type='registry_parse_problem',subagent_id='a',ref='r1',
                            period='2026-01',amount_kgs=50,detail='parse error',severity='medium')]*2)
        result,_=detect_canonical_anomalies(self.classified(),rows,extras)
        self.assertEqual(len(result),1)
        self.assertEqual(result.amount_kgs.iloc[0],100)
        self.assertEqual(result.status.iloc[0],'observed_data_error')
        self.assertEqual(result.error_owner.iloc[0],'unclear')

    def test_unmatched_reference_does_not_link_whole_month(self):
        extras=pd.DataFrame([dict(anomaly_type='1c_act_missing',subagent_id='a',ref='',
                            period='2026-01',amount_kgs=100,detail='missing',severity='high')])
        result,_=detect_canonical_anomalies(self.classified(),pd.DataFrame([self.component('e1')]),extras)
        self.assertEqual(result.match_ids.iloc[0],'[]')


if __name__ == '__main__': unittest.main()
