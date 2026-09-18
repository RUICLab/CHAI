"""Chronos under the SAME honest rules as Phase A: raw (unlooped) solar,
non-overlapping targets, held-out selection half vs evaluation half."""
import numpy as np
from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (forecast_ewma, forecast_ewma_slow,
                               forecast_mean_window, forecast_persistence,
                               forecast_chronos, forecast_chronos_q10)
from adaptive.forecast_study import to_bins, windows

P = PowerParams()
solar = to_bins(load_trace("traces/Solar_Indoor_Moving.csv"), P)
half = len(solar)//2

METHODS = {
  "persistence":          lambda h,b: forecast_persistence(h,None,b),
  "ewma a=0.3 (current)": lambda h,b: forecast_ewma(h,None,b),
  "ewma a=0.02":          lambda h,b: forecast_ewma_slow(h,None,b),
  "mean 60s":             lambda h,b: forecast_mean_window(60)(h,None,b),
  "CHRONOS (median)":     lambda h,b: forecast_chronos(h,None,b),
  "CHRONOS q10":          lambda h,b: forecast_chronos_q10(h,None,b),
}

print("Chronos vs cheap methods, unlooped solar, non-overlapping, held-out\n")
for H in (8,15,25,40):
    res={}
    for name,fn in METHODS.items():
        e_sel,e_ev=[],[]
        for start,stop,acc in ((0,half,e_sel),(half,len(solar),e_ev)):
            for i,target in windows(solar,H,start,stop):
                pred=fn(solar[:i],H)
                acc.append(abs(pred-target)/target)
        res[name]=(np.mean(e_sel) if e_sel else np.nan,
                   np.mean(e_ev) if e_ev else np.nan, len(e_sel), len(e_ev))
    n_sel,n_ev = res["persistence"][2], res["persistence"][3]
    print(f"H={H}s  select n={n_sel}  evaluate n={n_ev}")
    print(f"  {'method':<24}{'select':>10}{'evaluate':>11}")
    for name,(s,e,_,_) in res.items():
        print(f"  {name:<24}{s*100:9.1f}%{e*100:10.1f}%")
    print()
