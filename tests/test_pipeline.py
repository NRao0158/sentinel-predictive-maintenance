import numpy as np
import pytest
from sentinel.data import read,features,validate,labels
from sentinel.alerts import update

def test_features_are_causal():
    df=read('train_FD001.txt').query('unit == 1')
    prefix=df.iloc[:40].copy()
    np.testing.assert_allclose(features(df).iloc[:40],features(prefix))
    altered=df.copy(); altered.loc[40:,'s2']=1e6
    np.testing.assert_allclose(features(df).iloc[:40],features(altered).iloc[:40])

def test_machine_boundaries():
    df=read('train_FD001.txt').query('unit <= 2')
    both=features(df)
    one=features(df[df.unit==2])
    np.testing.assert_allclose(both.iloc[-len(one):],one)

def test_input_rejection():
    df=read('test_FD001.txt').iloc[:3]
    with pytest.raises(ValueError): validate(df.drop(columns='s2'))
    with pytest.raises(ValueError): validate(df.assign(s2=np.nan))
    with pytest.raises(ValueError): validate(df.assign(cycle=1))

def test_labels_and_alerts():
    df=read('train_FD001.txt').query('unit == 1')
    assert labels(df).iloc[-1]==0
    s={}
    assert update(s,1,.9,.5,50)=='created'
    s['1']['status']='acknowledged'
    assert update(s,1,.9,.5,51)=='unchanged'
    assert update(s,1,.2,.5,52)=='resolved'
    assert update(s,1,.9,.5,53)=='created'

def test_artifact_and_splits():
    import json,joblib
    from sentinel.data import ROOT
    r=json.loads((ROOT/'artifacts/metrics.json').read_text())
    a,b,c=[set(r['splits'][k]) for k in ['train','validation','audit']]
    assert not(a&b or b&c or a&c)
    model=joblib.load(ROOT/'artifacts/model.joblib')
    x=features(read('test_FD001.txt')).tail(5)
    p=model['model'].predict_proba(x)[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
