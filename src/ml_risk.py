"""Reproducible regularized logistic models and chronological validation.

Default labels are reconciliation rules, not independently verified ground truth.
Only source measurements are features; labels, identifiers and rule results are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import ValidationError

OWNERS=['bot','agent','1c']
SEED=42


class MLUnavailableError(ValidationError):
    """Valid reconciliation data, insufficient history/labels for honest ML validation."""


class LogisticModel:
    """Multiclass softmax regression with train-only scaling and L2 penalty."""
    def fit(self,x,y,epochs=250):
        x=np.asarray(x,dtype=float); y=np.asarray(y)
        if len(x)==0: raise ValidationError('Нет строк для обучения модели')
        self.classes=np.unique(y)
        self.mean=x.mean(axis=0); self.scale=x.std(axis=0)
        self.scale[self.scale<1e-8]=1
        z=np.column_stack([np.ones(len(x)),(x-self.mean)/self.scale])
        self.weights=np.zeros((z.shape[1],len(self.classes)))
        targets=(y[:,None]==self.classes[None,:]).astype(float)
        # Full-batch gradient: deterministic and no class weights distorting probabilities.
        for _ in range(epochs):
            scores=z@self.weights; scores-=scores.max(axis=1,keepdims=True)
            p=np.exp(scores); p/=p.sum(axis=1,keepdims=True)
            grad=z.T@(p-targets)/len(z)
            grad[1:]+=0.01*self.weights[1:]
            self.weights-=0.15*grad
        return self

    def predict_proba(self,x):
        z=np.column_stack([np.ones(len(x)),(np.asarray(x,dtype=float)-self.mean)/self.scale])
        scores=z@self.weights; scores-=scores.max(axis=1,keepdims=True)
        p=np.exp(scores); return p/p.sum(axis=1,keepdims=True)

    def as_dict(self):
        return {k:getattr(self,k).tolist() for k in ['classes','mean','scale','weights']}

    @classmethod
    def from_dict(cls,values):
        model=cls()
        for key,value in values.items(): setattr(model,key,np.asarray(value))
        return model


def feature_table(components):
    c=components.copy(); c['date']=pd.to_datetime(c.date,errors='coerce')
    ids=pd.Index(sorted(c.match_id.unique()),name='match_id')
    f=pd.DataFrame(index=ids)
    # Monetary totals use normalized debt signs, not inferred owners or correction amounts.
    for source in ['1c','etm','registry']:
        g=c[c.source.eq(source)].groupby('match_id')
        amount=g.amount.sum(min_count=1).reindex(ids)
        f[source+'_present']=g.size().reindex(ids,fill_value=0).gt(0).astype(float)
        f[source+'_log_abs_amount']=np.log1p(amount.abs().fillna(0))
        f[source+'_amount_sign']=np.sign(amount.fillna(0))
        f[source+'_amount_unknown']=amount.isna().astype(float)
        f[source+'_rows']=np.log1p(g.source_row_id.nunique().reindex(ids,fill_value=0))
    for kind in ['purchase','payment','void','refund','sale','service_fee']:
        f['kind_'+kind]=np.log1p(c[c.original_kind.eq(kind)].groupby('match_id').source_row_id.nunique().reindex(ids,fill_value=0))
    for creator in ['bot','subagent']:
        f['creator_'+creator]=np.log1p(c[c.source.eq('etm') & c.creator.eq(creator)].groupby('match_id').source_row_id.nunique().reindex(ids,fill_value=0))
    f['ticket_count']=np.log1p(c.groupby('match_id').ticket10.nunique().reindex(ids,fill_value=0))
    # Absolute identifiers, employee names and rule output fields are deliberately absent.
    dates=c.groupby('match_id').date.agg(['min','max']).reindex(ids)
    f['day_of_week']=dates['max'].dt.dayofweek.fillna(-1)
    f['day_of_month']=dates['max'].dt.day.fillna(0)
    return f.astype(float),dates


def make_labels(classified,verified=None):
    if classified.match_id.duplicated().any(): raise ValidationError('Повторные ID групп ML')
    labels=classified[['match_id','error_owner','confidence','assignment_status']].copy()
    labels['is_error']=labels.error_owner.map({'none':0,'bot':1,'agent':1,'1c':1})
    labels['label_basis']='rule_proxy'
    if verified is not None:
        v=verified.copy()
        required={'match_id','is_error','error_owner','is_verified'}
        if not required.issubset(v): raise ValidationError('Разметке нужны match_id,is_error,error_owner,is_verified')
        if v.match_id.duplicated().any() or not v.match_id.isin(labels.match_id).all():
            raise ValidationError('Неизвестные/повторные группы ручной разметки')
        truth=v.is_verified.astype(str).str.lower().isin(['1','true','yes'])
        v=v[truth]
        if not v.is_error.isin([0,1]).all(): raise ValidationError('is_error должен быть 0 или 1')
        if not ((v.is_error.eq(0)&v.error_owner.eq('none')) | (v.is_error.eq(1)&v.error_owner.isin(OWNERS+['unclear']))).all():
            raise ValidationError('Несогласованные метки ошибки/источника')
        # A verified dataset is used alone, without silently mixing hypotheses.
        labels=labels.merge(v[['match_id','is_error','error_owner']],on='match_id',how='inner',suffixes=('_rule',''))
        labels['label_basis']='verified'
    return labels.set_index('match_id')


def chronological_split(dates):
    periods=dates['max'].dt.to_period('M')
    valid=periods.dropna()
    if valid.nunique()<2: raise MLUnavailableError('Для временной проверки нужны минимум два месяца')
    cutoff=valid.max().start_time
    train=dates.index[dates['max'].lt(cutoff)]
    test=dates.index[dates['min'].ge(cutoff)]
    excluded=dates.index.difference(train.union(test))
    # Cross-boundary groups are excluded, not split into transactions/tickets.
    return train,test,excluded,cutoff


def classification_metrics(y,pred,classes):
    y=np.asarray(y); pred=np.asarray(pred)
    matrix=[[int(((y==a)&(pred==b)).sum()) for b in classes] for a in classes]
    per={}
    for label in classes:
        tp=int(((y==label)&(pred==label)).sum()); support=int((y==label).sum())
        precision=tp/max(1,int((pred==label).sum())); recall=tp/max(1,support)
        per[str(label)]={'support':support,'precision':precision,'recall':recall,
                         'f1':2*precision*recall/max(1e-15,precision+recall)}
    supported=[r['f1'] for r in per.values() if r['support']]
    return {'n':len(y),'accuracy':float((y==pred).mean()) if len(y) else None,
            'macro_f1':float(np.mean(supported)) if supported else None,
            'classes':list(classes),'confusion_matrix':matrix,'per_class':per}


def binary_metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p)
    result=classification_metrics(y,(p>=.5).astype(int),[0,1])
    result['brier_score']=float(np.mean((y-p)**2)) if len(y) else None
    result['roc_auc']=None; result['average_precision']=None
    if len(np.unique(y))==2:
        ranks=pd.Series(p).rank(method='average').to_numpy()
        pos=int(y.sum()); neg=len(y)-pos
        result['roc_auc']=float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))
        # Ties share a threshold, avoiding order-dependent optimistic AP.
        thresholds=pd.DataFrame({'p':p,'y':y}).groupby('p').y.agg(['sum','size']).sort_index(ascending=False)
        tp=thresholds['sum'].cumsum(); total=thresholds['size'].cumsum()
        result['average_precision']=float(((tp/total)*thresholds['sum']/pos).sum())
    return result


def _probability(model,x,label):
    if model is None or label not in model.classes: return np.zeros(len(x))
    return model.predict_proba(x)[:,list(model.classes).index(label)]


def predict_saved_model(model_path,components):
    artifact=json.loads(Path(model_path).read_text(encoding='utf-8'))
    if artifact.get('format_version')!=1: raise ValidationError('Неизвестная версия модели')
    features,_=feature_table(components)
    if features.columns.tolist()!=artifact['feature_names']: raise ValidationError('Признаки не совпадают с моделью')
    error=LogisticModel.from_dict(artifact['error_model'])
    source=LogisticModel.from_dict(artifact['source_model']) if artifact['source_model'] else None
    x=features.to_numpy()
    result=pd.DataFrame({'match_id':features.index,'p_error':_probability(error,x,1)})
    for owner in OWNERS: result['p_source_'+owner]=_probability(source,x,owner)
    return result


def aggregate_risks(groups,components):
    fields=['match_id','p_error']+['p_source_'+x for x in OWNERS]
    joined=components.merge(groups[fields],on='match_id',validate='many_to_one')
    # Original transactions are not inflated by exploded tickets. Max group score is a priority score.
    # Keep all source probabilities from the same highest-risk group, so they sum to one.
    transactions=joined.sort_values(['p_error','match_id'],ascending=[False,True]).drop_duplicates(
        ['source','source_row_id','subagent_id'])[['source','source_row_id','subagent_id']+fields[1:]].reset_index(drop=True)
    transactions['predicted_owner']=transactions[['p_source_'+x for x in OWNERS]].idxmax(axis=1).str.replace('p_source_','',regex=False)
    transactions.loc[transactions.p_error.lt(.5),'predicted_owner']='none'
    transactions.loc[transactions.p_error.ge(.5) & transactions[['p_source_'+x for x in OWNERS]].max(axis=1).lt(.6),'predicted_owner']='unclear'
    def rank(frame,key):
        r=frame.groupby(key).agg(rows=('p_error','size'),mean_risk=('p_error','mean'),max_risk=('p_error','max'),
                                high_risk_rows=('p_error',lambda v:int(v.ge(.5).sum()))).reset_index()
        r['enough_data']=r.rows.ge(30)
        r=r.sort_values(['mean_risk','rows',key],ascending=[False,False,True]).reset_index(drop=True)
        r['priority']=np.arange(1,len(r)+1)
        return r
    sub=rank(transactions[transactions.source.eq('etm')],'subagent_id')
    reg=joined[joined.source.eq('registry') & joined.creator.notna() & joined.creator.ne('')]
    reg=reg.groupby(['creator','source_row_id'],as_index=False).p_error.max()
    employee=rank(reg,'creator')
    return transactions,sub,employee


def run_ml(interim_dir='interim',verified_labels=None):
    root=Path(interim_dir); out=root/'ml'; out.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(root/'reconciliation/match_components.csv',dtype={'ticket10':str},encoding='utf-8-sig',low_memory=False)
    classified=pd.read_csv(root/'p5/p5_group_classification.csv',encoding='utf-8-sig',low_memory=False)
    verified=pd.read_csv(verified_labels) if verified_labels else None
    features,dates=feature_table(c); labels=make_labels(classified,verified)
    train,test,excluded,cutoff=chronological_split(dates)
    train=train.intersection(labels.index[labels.is_error.notna()]); test=test.intersection(labels.index[labels.is_error.notna()])
    if len(train)<20 or len(test)<5: raise MLUnavailableError('Недостаточно размеченных групп в обучении/проверке ML')
    y=labels.loc[train,'is_error'].astype(int).to_numpy()
    if len(np.unique(y))<2: raise MLUnavailableError('Для обучения нужны примеры ошибки и отсутствия ошибки')
    error_model=LogisticModel().fit(features.loc[train].to_numpy(),y)
    source_train=train.intersection(labels.index[labels.is_error.eq(1)&labels.error_owner.isin(OWNERS)])
    source_test=test.intersection(labels.index[labels.is_error.eq(1)&labels.error_owner.isin(OWNERS)])
    source_model=LogisticModel().fit(features.loc[source_train].to_numpy(),labels.loc[source_train,'error_owner'].to_numpy()) if len(source_train) else None
    mode='verified' if verified_labels else 'rule_proxy'
    supported=[] if source_model is None else source_model.classes.tolist()
    notes=['Вероятности не откалиброваны независимо; используют доступные после сверки сведения, не прогноз до операции.',
           'Оценка на последнем месяце; группы, пересекающие границу, исключены.',
           'Максимум по группам исходной транзакции — оценка приоритета, не вероятность объединения событий.']
    if mode=='rule_proxy': notes.append('Метки получены правилами. Метрики измеряют воспроизведение правил, не точность реальных ошибок или виновника. unclear исключен из бинарной разметки.')
    missing=sorted(set(OWNERS)-set(supported))
    if missing: notes.append('Нет обучающих примеров источников: '+', '.join(missing)+'. Нулевая вероятность означает неподдерживаемый класс.')
    xt=features.loc[test].to_numpy(); yt=labels.loc[test,'is_error'].astype(int).to_numpy()
    p=_probability(error_model,xt,1)
    baseline=np.full(len(test),y.mean())
    source_pred=source_model.classes[np.argmax(source_model.predict_proba(features.loc[source_test].to_numpy()),axis=1)] if source_model is not None and len(source_test) else np.array([])
    metrics={'label_basis':mode,'cutoff':str(cutoff),'train_groups':len(train),'test_groups':len(test),
             'excluded_boundary_or_unknown_date':len(excluded),'unlabeled_groups':int(labels.is_error.isna().sum()),
             'binary':binary_metrics(yt,p),'binary_baseline_train_prevalence':binary_metrics(yt,baseline),
             'source':classification_metrics(labels.loc[source_test,'error_owner'].to_numpy(),source_pred,OWNERS) if len(source_pred) else None,
             'source_supported_classes':supported,'source_unsupported_classes':missing,'notes':notes}
    metrics['train_error_counts']={str(int(k)):int(v) for k,v in labels.loc[train,'is_error'].value_counts().items()}
    metrics['train_source_counts']={str(k):int(v) for k,v in labels.loc[source_train,'error_owner'].value_counts().items()}
    if len(source_pred):
        majority=labels.loc[source_train,'error_owner'].value_counts().index[0]
        metrics['source_baseline_majority']=classification_metrics(labels.loc[source_test,'error_owner'].to_numpy(),np.repeat(majority,len(source_test)),OWNERS)
    # Keep evaluated models: no test-set refit hidden behind reported metrics.
    x=features.to_numpy(); groups=pd.DataFrame({'match_id':features.index,'p_error':_probability(error_model,x,1)})
    for owner in OWNERS: groups['p_source_'+owner]=_probability(source_model,x,owner)
    groups['evaluation_role']=np.where(groups.match_id.isin(train),'train',np.where(groups.match_id.isin(test),'test','unlabeled_or_excluded'))
    groups['label_basis']=mode
    groups['unsupported_sources']=' | '.join(missing)
    transactions,sub,employee=aggregate_risks(groups,c)
    for frame in [transactions,sub,employee]:
        frame['label_basis']=mode; frame['score_semantics']='приоритет проверки; некалиброванная оценка'
        frame['unsupported_sources']=' | '.join(missing)
    test_predictions=groups[groups.match_id.isin(test)].merge(labels[['is_error','error_owner']],left_on='match_id',right_index=True,validate='one_to_one')
    for frame,name in [(groups,'group_predictions.csv'),(transactions,'transaction_risks.csv'),(sub,'subagent_risks.csv'),
                       (employee,'employee_risks.csv'),(test_predictions,'holdout_predictions.csv')]:
        frame.to_csv(out/name,index=False,encoding='utf-8-sig')
    artifact={'format_version':1,'algorithm':'L2 softmax regression','seed':SEED,'feature_names':features.columns.tolist(),
              'label_basis':mode,'error_model':error_model.as_dict(),'source_model':source_model.as_dict() if source_model else None,
              'train_cutoff':str(cutoff),'supported_sources':supported,'notes':notes,
              'input_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [root/'reconciliation/match_components.csv',root/'p5/p5_group_classification.csv']}}
    for data,name in [(metrics,'metrics.json'),(artifact,'model.json'),(features.columns.tolist(),'features.json')]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[ML] {len(train)} train / {len(test)} test; labels={mode}; F1={metrics["binary"]["macro_f1"]:.3f}; unsupported sources={missing}')
    return metrics


def report_frames(interim_dir):
    root=Path(interim_dir)/'ml'
    if not (root/'metrics.json').exists():
        if not (root/'status.json').exists(): return {}
        status=json.loads((root/'status.json').read_text(encoding='utf-8'))
        return {'Качество модели':pd.DataFrame([{'Статус':status['status'],'Пояснение':status.get('reason','')}])}
    metrics=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    rows=[{'Показатель':'Основа разметки','Значение':metrics['label_basis']},
          {'Показатель':'Начало отложенного периода','Значение':metrics['cutoff']},
          {'Показатель':'Обучающих групп','Значение':metrics['train_groups']},
          {'Показатель':'Проверочных групп','Значение':metrics['test_groups']}]
    for task in ['train_error_counts','train_source_counts','binary','binary_baseline_train_prevalence','source','source_baseline_majority']:
        for name,value in (metrics.get(task) or {}).items():
            rows.append({'Показатель':task+': '+name,'Значение':json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value})
    rows.extend({'Показатель':'Ограничение','Значение':note} for note in metrics['notes'])
    frames={'Качество модели':pd.DataFrame(rows)}
    translations={'source':'Источник','source_row_id':'Исходная транзакция/строка','subagent_id':'Субагент','creator':'Сотрудник',
                  'p_error':'Оценка риска ошибки','p_source_bot':'Источник бот, условная оценка',
                  'p_source_agent':'Источник агент, условная оценка','p_source_1c':'Источник 1С, условная оценка',
                  'predicted_owner':'Предполагаемый источник','rows':'Исходных строк','mean_risk':'Средняя оценка риска',
                  'max_risk':'Максимальная оценка риска','high_risk_rows':'Строк с оценкой от 0.5','enough_data':'Не менее 30 строк',
                  'priority':'Очередность','label_basis':'Основа разметки','score_semantics':'Как понимать оценку',
                  'unsupported_sources':'Источники без обучающих примеров'}
    for file,sheet in [('transaction_risks.csv','Риски транзакций'),('subagent_risks.csv','Приоритет субагентов'),('employee_risks.csv','Приоритет сотрудников')]:
        frames[sheet]=pd.read_csv(root/file,encoding='utf-8-sig').rename(columns=translations)
    return frames


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Этап 7: модель риска с временной проверкой')
    parser.add_argument('--interim',default='interim'); parser.add_argument('--verified-labels')
    args=parser.parse_args(); run_ml(args.interim,args.verified_labels)
