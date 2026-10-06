import json, hashlib, platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from .data import ROOT, read, features, labels

HORIZON = 30

def classification(y,p,t):
    z = p >= t
    return dict(pr_auc=float(average_precision_score(y,p)), roc_auc=float(roc_auc_score(y,p)), precision=float(precision_score(y,z,zero_division=0)), recall=float(recall_score(y,z,zero_division=0)), f1=float(f1_score(y,z,zero_division=0)), brier=float(brier_score_loss(y,p)), false_positives=int(((y==0)&z).sum()), missed_failures=int(((y==1)&~z).sum()), observations=len(y))

def regression(y,p):
    return dict(mae=float(mean_absolute_error(y,p)), rmse=float(np.sqrt(mean_squared_error(y,p))))

def bootstrap_units(units,y,p,t):
    rng=np.random.default_rng(73); scores=[]
    for _ in range(300):
        sample=rng.choice(np.unique(units),len(np.unique(units)),replace=True)
        ix=np.concatenate([np.flatnonzero(units==u) for u in sample])
        if len(np.unique(y[ix]))==2: scores.append(average_precision_score(y[ix],p[ix]))
    return list(np.quantile(scores,[.025,.975]).astype(float))

def main():
    train=read('train_FD001.txt'); test=read('test_FD001.txt')
    offsets=pd.read_csv(ROOT/'data/RUL_FD001.txt',sep=r'\s+',header=None)[0]
    offsets.index=np.arange(1,len(offsets)+1)
    x=features(train); rul=labels(train); y=(rul<=HORIZON).astype(int)
    ids=np.random.default_rng(42).permutation(train.unit.unique())
    fit=train.unit.isin(ids[:60]); val=train.unit.isin(ids[60:80]); audit=train.unit.isin(ids[80:])
    # Exclude failure-cycle observations: prediction must precede actual failure.
    fit &= rul>0; val &= rul>0; audit &= rul>0
    models={'prevalence':DummyClassifier(strategy='prior'), 'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)), 'gradient_boosting':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=5,random_state=42)}
    validation={}
    for name,m in models.items():
        m.fit(x[fit],y[fit]); validation[name]=float(average_precision_score(y[val],m.predict_proba(x[val])[:,1]))
    selected=max(validation,key=validation.get); model=models[selected]
    pv=model.predict_proba(x[val])[:,1]
    # Operational starting assumption: a missed positive costs 5x a false alert.
    thresholds=np.linspace(.05,.95,91)
    costs=[int(((y[val]==1)&(pv<t)).sum())*5+int(((y[val]==0)&(pv>=t)).sum()) for t in thresholds]
    threshold=float(thresholds[np.argmin(costs)])
    xt=features(test); truth=labels(test,offsets); yt=(truth<=HORIZON).astype(int)
    last=test.groupby('unit').tail(1).index
    results={}
    for name,m in models.items():
        p=m.predict_proba(xt.loc[last])[:,1]
        results[name]=classification(yt.loc[last].to_numpy(),p,threshold if name==selected else .5)
    reg_models={'median':DummyRegressor(strategy='median'),'gradient_boosting':HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=5,random_state=42)}
    capped=rul.clip(upper=125); vr={}
    for name,m in reg_models.items():
        m.fit(x[fit],capped[fit]); vr[name]=regression(capped[val],m.predict(x[val]))['mae']
    rname=min(vr,key=vr.get); reg=reg_models[rname]
    regression_results={name:regression(truth.loc[last].clip(upper=125),m.predict(xt.loc[last])) for name,m in reg_models.items()}
    pa=model.predict_proba(x[audit])[:,1]
    importance=permutation_importance(model,x[val],y[val],scoring='average_precision',n_repeats=3,random_state=42)
    top=sorted(zip(x.columns,importance.importances_mean),key=lambda z:z[1],reverse=True)
    predictions=test.loc[last,['unit','cycle']].copy()
    predictions['risk']=model.predict_proba(xt.loc[last])[:,1]
    predictions['predicted_rul']=np.clip(reg.predict(xt.loc[last]),0,125)
    predictions['actual_rul']=truth.loc[last]
    art=ROOT/'artifacts'; art.mkdir(exist_ok=True)
    predictions.to_csv(art/'predictions.csv',index=False)
    # Validation residual quantile gives a heuristic interval; no coverage guarantee.
    radius=float(np.quantile(np.abs(capped[val]-reg.predict(x[val])),.9))
    bounds={c:[float(x.loc[fit,c].min()),float(x.loc[fit,c].max())] for c in x}
    joblib.dump(dict(model=model,regressor=reg,threshold=threshold,columns=list(x),radius=radius,bounds=bounds),art/'model.joblib')
    report=dict(dataset='NASA C-MAPSS FD001 (simulated turbofan engines)',horizon=HORIZON,selected=selected,regressor=rname,threshold=threshold,validation_pr_auc=validation,validation_rul_mae=vr,official_test=results,official_test_rul=regression_results,internal_audit=classification(y[audit].to_numpy(),pa,threshold),audit_pr_auc_95_interval=bootstrap_units(train.loc[audit,'unit'].to_numpy(),y[audit].to_numpy(),pa,threshold),splits={'train':ids[:60].tolist(),'validation':ids[60:80].tolist(),'audit':ids[80:].tolist()},features=list(x),global_importance=[{'feature':c,'pr_auc_drop':float(v)} for c,v in top],training_rows=int(fit.sum()),timestamp=datetime.now(timezone.utc).isoformat(),sklearn=sklearn.__version__,python=platform.python_version(),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.txt')})
    (art/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['selected','threshold','official_test','official_test_rul']},indent=2))

if __name__=='__main__': main()
