"""Session-local alert lifecycle. No external notifications."""
def update(state,unit,risk,threshold,cycle):
    key=str(unit); current=state.get(key)
    if risk>=threshold and (current is None or current['status']=='resolved'):
        state[key]={'unit':unit,'cycle':cycle,'risk':float(risk),'status':'open'}
        return 'created'
    if risk<threshold*.75 and current and current['status']!='resolved':
        current['status']='resolved'
        return 'resolved'
    return 'unchanged'
