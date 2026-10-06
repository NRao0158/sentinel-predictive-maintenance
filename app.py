import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sentinel.data import ROOT, COLS, features, validate
from sentinel.alerts import update

st.set_page_config(page_title='Sentinel | Failure Prediction',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp {background:#0b1220;color:#e7edf6} h1 {letter-spacing:-2px} [data-testid="stMetric"] {background:#142237;padding:20px;border-radius:14px;border:1px solid #243752} .stButton button {border-radius:10px} </style>''',unsafe_allow_html=True)
st.caption('SENTINEL / PREDICTIVE MAINTENANCE LAB')
st.title('See failure before it happens.')
st.write('Predict 30-cycle failure risk, explore sensor history, and rehearse a maintenance decision.')
st.info('Public NASA FD001 simulation benchmark. Engines are not printers or scanners; cycles are not hours. This demo does not send notifications or validate logistics performance.')

@st.cache_resource
def load():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    return joblib.load(ROOT/'artifacts/model.joblib')

@st.cache_data
def demo():
    return pd.read_csv(ROOT/'artifacts/demo_history.csv')

b=load(); report=json.loads((ROOT/'artifacts/metrics.json').read_text())
tab1,tab2,tab3=st.tabs(['Prediction studio','Model evidence','Batch scoring'])

with tab1:
    history=demo()
    unit=st.selectbox('Replay an unseen benchmark engine',sorted(history.unit.unique()))
    full=history[history.unit==unit]
    cycle=st.slider('Operating cycle (only history up to this point is used)',int(full.cycle.min()),int(full.cycle.max()),int(full.cycle.max()))
    visible=full[full.cycle<=cycle]
    x=features(visible); row=x.tail(1)
    risk=float(b['model'].predict_proba(row)[0,1]); rul=float(np.clip(b['regressor'].predict(row)[0],0,125))
    a,c,d=st.columns(3)
    a.metric('Failure within 30 cycles',f'{risk:.1%}')
    c.metric('Estimated remaining life',f'{rul:.0f} cycles')
    d.metric('Recommended action','Inspect' if risk>=b['threshold'] else 'Continue monitoring')
    st.caption(f"Validation-selected alert threshold: {b['threshold']:.2f}. RUL is capped at 125 cycles. Risk scores are uncalibrated model estimates.")
    st.write(f"Heuristic RUL range: {max(0,rul-b['radius']):.0f}–{min(125,rul+b['radius']):.0f} cycles. Based on validation residuals; coverage is not guaranteed.")
    sensor=st.selectbox('Sensor history',['s2','s3','s4','s7','s9','s11','s12','s14','s15'])
    chart=alt.Chart(visible).mark_line(color='#42d9b5').encode(x=alt.X('cycle:Q',title='Operating cycle'),y=alt.Y(f'{sensor}:Q',scale=alt.Scale(zero=False)),tooltip=['cycle',sensor])
    st.altair_chart(chart,width='stretch')
    # Local sensitivity: perturb one feature to the training midpoint; not causal attribution.
    changes=[]
    for feature in row.columns:
        altered=row.copy(); low,high=b['bounds'][feature]; altered[feature]=(low+high)/2
        changed=float(b['model'].predict_proba(altered)[0,1])
        changes.append({'feature':feature,'risk_change_to_training_midpoint':changed-risk})
    with st.expander('Prediction sensitivity (not causal explanations)'):
        st.write('Features are perturbed independently, so combinations may be unrealistic. A change is model sensitivity, not a maintenance diagnosis.')
        st.dataframe(pd.DataFrame(changes).assign(magnitude=lambda z:z.risk_change_to_training_midpoint.abs()).sort_values('magnitude',ascending=False).head(8),hide_index=True)
    st.session_state.setdefault('alerts',{})
    st.session_state.setdefault('events',[])
    if st.button('Evaluate simulated alert'):
        event=update(st.session_state.alerts,int(unit),risk,b['threshold'],cycle)
        if event!='unchanged': st.session_state.events.append({'unit':int(unit),'cycle':cycle,'event':event})
        st.success(f'Alert {event}. No external message sent.')
    if str(unit) in st.session_state.alerts and st.session_state.alerts[str(unit)]['status']=='open':
        if st.button('Acknowledge inspection'):
            st.session_state.alerts[str(unit)]['status']='acknowledged'
            st.session_state.events.append({'unit':int(unit),'cycle':cycle,'event':'acknowledged'})
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts.values()),hide_index=True)
        st.download_button('Download event history',pd.DataFrame(st.session_state.events).to_csv(index=False),'alert_events.csv','text/csv')
    st.caption('Alerts persist only in this browser session. Lower replay risk resolves an alert; it is not evidence of repaired equipment.')

with tab2:
    st.subheader('Evidence from machines excluded from training')
    st.write('60 training engines / 20 validation engines / 20 internal audit engines. Official test: last observed cycle of 100 separate engines. Model and threshold chosen on validation only.')
    st.dataframe(pd.DataFrame(report['official_test']).T, width='stretch')
    st.write('Remaining useful life: evaluated against true RUL capped at 125 cycles.')
    st.dataframe(pd.DataFrame(report['official_test_rul']).T,width='stretch')
    st.write('Internal audit PR-AUC bootstrap interval (machines resampled):',report['audit_pr_auc_95_interval'])
    st.bar_chart(pd.DataFrame(report['global_importance']).head(10).set_index('feature'))
    st.caption('Importance measured on validation; correlated features share importance. No measured downtime savings or printer generalization claims.')
    st.download_button('Download complete experiment report',json.dumps(report,indent=2),'metrics.json','application/json')

with tab3:
    st.write('Upload a CSV of FD001-compatible engine history. One row per unit/cycle; required numeric columns:')
    st.code(','.join(COLS))
    st.download_button('Download sample history',history[COLS].to_csv(index=False),'sample_history.csv','text/csv')
    upload=st.file_uploader('Engine telemetry CSV',type=['csv'])
    if upload is not None:
        try:
            df=validate(pd.read_csv(upload))
            if len(df)>100000: raise ValueError('Limit uploads to 100,000 observations.')
            xf=features(df); idx=df.groupby('unit').tail(1).index
            result=df.loc[idx,['unit','cycle']].copy()
            result['risk']=b['model'].predict_proba(xf.loc[idx])[:,1]
            result['predicted_rul']=np.clip(b['regressor'].predict(xf.loc[idx]),0,125)
            result['out_of_training_range']=[any(r[c]<b['bounds'][c][0] or r[c]>b['bounds'][c][1] for c in xf) for _,r in xf.loc[idx].iterrows()]
            result['inspect']=result.risk>=b['threshold']
            st.dataframe(result.sort_values('risk',ascending=False),hide_index=True)
            st.download_button('Download predictions',result.to_csv(index=False),'predictions.csv','text/csv')
            if result.out_of_training_range.any(): st.warning('Some features are outside training ranges. Treat these predictions as unreliable pending domain validation.')
        except (ValueError,TypeError,pd.errors.ParserError) as exc:
            st.error(str(exc))
