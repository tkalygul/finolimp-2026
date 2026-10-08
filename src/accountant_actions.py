"""Stage 6: reviewable accounting proposals, never automatic postings."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import ValidationError

TOLERANCE = 0.02


def _ids(value):
    if pd.isna(value): return []
    try: return json.loads(value)
    except (TypeError, ValueError): return []


def build_accountant_actions(classified, bridge, anomalies=None):
    if classified.match_id.duplicated().any():
        raise ValidationError('Повторные ID групп: корректировки могут задвоиться')
    if bridge.duplicated(['subagent_id', 'period']).any():
        raise ValidationError('Повторные строки моста баланса')
    anomaly_links={}
    if anomalies is not None:
        for row in anomalies.to_dict('records'):
            for mid in _ids(row.get('match_ids', '[]')):
                anomaly_links.setdefault(mid, []).append(row)
    details=[]
    for row in classified.to_dict('records'):
        if not row.get('is_error', 0) and row['match_id'] not in anomaly_links: continue
        linked=anomaly_links.get(row['match_id'], [])
        flags=set(_ids(row.get('flags', '[]')))
        blocked=bool(flags & {'duplicate_suspected','period_difference','registry_parse_problem','void_not_fully_reversed'})
        blocked |= any(a.get('status') != 'informational' for a in linked)
        target='выяснить'; delta=np.nan
        action='Проверить полноту выгрузок, документы, билет, сумму, сборы и период; установить источник ошибки'
        code=row['error_type']
        a,e=row.get('amount_1c',np.nan),row.get('amount_etm',np.nan)
        if not blocked and code in {'etm_wrong_amount','etm_not_executed'}:
            # Missing ETM is zero movement, not an unknown balance.
            if code=='etm_not_executed': e=0.0
            if pd.notna(a) and pd.notna(e):
                target='ETM'; delta=float(e-a)
                action='Подтвердить документ 1С и запись реестра, полноту ETM; затем исправить сумму или провести пропущенную операцию'
        elif not blocked and code=='1c_voided_posted' and pd.notna(a) and pd.notna(e):
            target='1С'; delta=float(e-a)
            action='Подтвердить отмену и состав сборов; затем сторнировать или исправить документ продажи в 1С'
        elif row.get('error_owner')=='agent':
            action='Исправить исходную запись реестра после проверки билета, суммы, валюты и курса; проверить, перенесена ли ошибка в ETM'
        elif code=='period_difference':
            action='Сверить даты документов и правила закрытия месяца в 1С/ETM; исправлять период только после подтверждения, не создавать повторную денежную проводку'
        elif code=='dependent_sources_agree':
            action='Сверить первичный документ с 1С и реестром; проверить счет и состав суммы в 1С, затем установить, какой источник исправлять'
        elif code=='registry_missing':
            action='Найти исходную запись реестра для операции бота или подтвердить неполную выгрузку; проверить происхождение операции'
        direction=('увеличить' if delta>0 else 'уменьшить') if pd.notna(delta) and abs(delta)>TOLERANCE else ''
        details.append(dict(match_id=row['match_id'],subagent_id=row['subagent_id'],period=row.get('period',''),
            ticket10=row.get('ticket10',''),error_type=code,error_owner=row.get('error_owner','unclear'),
            confidence=row.get('confidence','low'),target=target,proposed_delta=delta,direction=direction,
            action=action,status='после подтверждения' if target!='выяснить' else 'сначала выяснить',
            amount_1c=a,amount_etm=e,documents_1c=row.get('doc_1c','[]'),transactions_etm=row.get('txn_ids','[]'),
            reason=row.get('reason',''),anomaly_ids=json.dumps([x['anomaly_id'] for x in linked],ensure_ascii=False)))
    columns=['match_id','subagent_id','period','ticket10','error_type','error_owner','confidence','target',
             'proposed_delta','direction','action','status','amount_1c','amount_etm','documents_1c','transactions_etm','reason','anomaly_ids']
    detail=pd.DataFrame(details,columns=columns)
    summary=[]
    for agent,b in bridge.groupby('subagent_id',sort=True):
        b=b.sort_values('period'); last=b.iloc[-1]
        d=detail[detail.subagent_id.eq(agent)]
        etm=d[d.target.eq('ETM')]; onec=d[d.target.eq('1С')]; review=d[d.target.eq('выяснить')]
        linked_anomalies=[] if anomalies is None else anomalies[anomalies.subagent_id.eq(agent)].to_dict('records')
        # Period is a joined list; proposals must fall entirely within the covered interval.
        inside=lambda value: all(b.period.min()<=p<=last.period for p in str(value).split(' | '))
        outside=int((~etm.period.map(inside).eq(True)).sum()+(~onec.period.map(inside).eq(True)).sum())
        etm=etm[etm.period.map(inside).eq(True)]; onec=onec[onec.period.map(inside).eq(True)]
        de=float(etm.proposed_delta.sum()); dc=float(onec.proposed_delta.sum())
        before=last.closing_difference
        after=before+de+dc if pd.notna(before) else np.nan
        scenario_change = abs(after) - abs(before) if pd.notna(before) and pd.notna(after) else np.nan
        scenario_warning = ''
        if pd.isna(scenario_change):
            scenario_assessment = 'Неизвестно'
        elif scenario_change > TOLERANCE:
            scenario_assessment = 'Увеличивается'
            scenario_warning = ('После предложенных исправлений общая разница увеличивается. '
                'Необходима проверка остальных расхождений и истории проводок. '
                'Исправление отдельной операции может убрать взаимную компенсацию ошибок.')
        elif scenario_change < -TOLERANCE:
            scenario_assessment = 'Уменьшается'
        else:
            scenario_assessment = 'Без изменения'
        unknown=int(b.closing_difference.isna().sum())
        residual=float(b.unexplained.abs().sum()) if b.unexplained.notna().all() else np.nan
        questions=[]
        if scenario_warning: questions.append(scenario_warning)
        if len(review): questions.append(f'Установить причину {len(review)} групп расхождений')
        if outside: questions.append(f'Проверить {outside} предложений вне полного периода моста; они не включены в сценарий')
        if linked_anomalies: questions.append(f'Проверить {len(linked_anomalies)} сигналов аномалий; суммы не складывать')
        if unknown: questions.append(f'Получить недостающие балансы/акты за {unknown} месяцев')
        if pd.isna(residual) or residual>TOLERANCE: questions.append('Объяснить остатки моста баланса')
        for field,text in [('check_1c','Проверить внутренний баланс акта 1С'),
                           ('check_etm','Проверить внутренний баланс ETM'),
                           ('act_carryover_gap','Проверить перенос сальдо между актами 1С')]:
            if field in b and b[field].abs().gt(TOLERANCE).any(): questions.append(text)
        opening=b.iloc[0].opening_difference
        if pd.isna(opening) or abs(opening)>TOLERANCE: questions.append('Подтвердить начальное сальдо по предыдущей истории')
        if pd.notna(after) and abs(after)>TOLERANCE: questions.append('Оставшаяся разница требует проверки; не списывать её автоматически')
        summary.append(dict(subagent_id=agent,period_from=b.period.min(),period_to=last.period,
            closing_difference=before,etm_delta_after_confirmation=de,onec_debt_delta_after_confirmation=dc,
            approved_etm_delta=0.0,approved_onec_delta=0.0,
            expected_difference_after_proposals=after,remaining_difference_without_approval=before,
            scenario_absolute_change=scenario_change,scenario_assessment=scenario_assessment,
            scenario_warning=scenario_warning,
            etm_action='; '.join(f"{r.direction} депозит на {abs(r.proposed_delta):.2f} сом ({r.match_id})" for r in etm.itertuples()) or 'Нет обоснованного предложения',
            onec_action='; '.join(f"{r.direction} долг на {abs(r.proposed_delta):.2f} сом ({r.match_id})" for r in onec.itertuples()) or 'Нет обоснованного предложения',
            clarify='; '.join(questions) or 'Расхождений, требующих действий, не обнаружено',
            review_groups=len(review),anomaly_count=len(linked_anomalies),unknown_months=unknown,
            unexplained_absolute_total=residual,match_ids=json.dumps(d.match_id.tolist(),ensure_ascii=False),
            status='требуется подтверждение/выяснение' if len(d) or questions else 'согласовано',
            note='Знак ETM: плюс увеличивает депозит; знак 1С: плюс увеличивает долг. Сценарий после подтверждения, проводки не выполнены.'))
    return pd.DataFrame(summary),detail


def run_accountant_actions(interim_dir):
    root=Path(interim_dir)
    classification=root/'p5/p5_group_classification.csv'
    bridge=root/'reconciliation/balance_bridge.csv'
    if not classification.exists() or not bridge.exists(): return None
    c=pd.read_csv(classification,encoding='utf-8-sig',dtype={'ticket10':str},low_memory=False)
    b=pd.read_csv(bridge,encoding='utf-8-sig')
    path=root/'p5/p5_anomalies.csv'
    a=pd.read_csv(path,encoding='utf-8-sig',low_memory=False) if path.exists() else None
    summary,detail=build_accountant_actions(c,b,a)
    for frame,name in [(summary,'accountant_actions.csv'),(detail,'accountant_action_details.csv')]:
        frame.to_csv(root/'p5'/name,index=False,encoding='utf-8-sig')
    return summary,detail


SUMMARY_RU={'subagent_id':'Субагент','period_from':'Период с','period_to':'Период по',
 'closing_difference':'Разница на конец, сом','etm_delta_after_confirmation':'Предложение ETM со знаком, сом',
 'onec_debt_delta_after_confirmation':'Предложение 1С со знаком, сом','approved_etm_delta':'Подтверждено ETM, сом',
 'approved_onec_delta':'Подтверждено 1С, сом','expected_difference_after_proposals':'Разница в сценарии после подтверждения, сом',
 'scenario_absolute_change':'Изменение модуля общей разницы, сом',
 'scenario_assessment':'Общая разница после предложений',
 'scenario_warning':'Предупреждение по сценарию',
 'remaining_difference_without_approval':'Разница до подтверждения, сом','etm_action':'Что изменить в ETM',
 'onec_action':'Что исправить в 1С','clarify':'Что сначала выяснить','review_groups':'Групп для выяснения',
 'anomaly_count':'Сигналов аномалий','unknown_months':'Месяцев с неизвестным балансом',
 'unexplained_absolute_total':'Сумма модулей месячных необъясненных остатков, сом','match_ids':'Связанные группы','status':'Статус','note':'Пояснение знаков'}
DETAIL_RU={'match_id':'ID группы сопоставления','subagent_id':'Субагент','period':'Период','ticket10':'Билет',
 'error_type':'Код ошибки','error_owner':'Предполагаемый источник','confidence':'Уверенность правила','target':'Где исправить',
 'proposed_delta':'Изменение со знаком, сом','direction':'Направление','action':'Действие бухгалтера','status':'Статус',
 'amount_1c':'Сумма 1С, сом','amount_etm':'Сумма ETM, сом','documents_1c':'Документы 1С',
 'transactions_etm':'Транзакции ETM','reason':'Объяснение','anomaly_ids':'Связанные аномалии'}
