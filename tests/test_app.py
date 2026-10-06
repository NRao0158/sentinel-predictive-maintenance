from streamlit.testing.v1 import AppTest
from sentinel.data import ROOT

def test_demo_and_alert_interaction():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert len(app.metric)==3
    app.button[0].click().run()
    assert not app.exception
    assert 'Alert' in app.success[0].value
    app.slider[0].set_value(20).run()
    assert not app.exception
    assert len(app.metric)==3
