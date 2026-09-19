"""Does dropping the exact level and densifying the cheap end beat the current ladder?

Current  L0 [1,1] 62cy err 0.000 | L1 [5,2] 13cy 0.019 | L2 [4,4] 8cy 0.280
Proposed (no exact level, denser cheap end), all MEASURED points:
         M0 [5,2] 13cy 0.019 | M1 [4,4] 8cy 0.280 | M2 [4,8] 6cy 0.344
"""
import numpy as np
from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import build_levels, policy_fixed, forecast_persistence, simulate

P=PowerParams()
CURRENT=[{"name":"L0","stride":"1,1","error":0.000,"cycles":62},
         {"name":"L1","stride":"5,2","error":0.019,"cycles":13},
         {"name":"L2","stride":"4,4","error":0.280,"cycles":8}]
LADDER =[{"name":"M0","stride":"5,2","error":0.019,"cycles":13},
         {"name":"M1","stride":"4,4","error":0.280,"cycles":8},
         {"name":"M2","stride":"4,8","error":0.344,"cycles":6}]

def ev(res, levels):
    e=np.array(res["errors"],dtype=float); p=res["periods"]
    off=res["energy_harvested_mJ"]+res["energy_spilled_mJ"]
    q=float(np.sum(1.0-e)) if e.size else 0.0
    return res["completed"]/p if p else 0, (e.mean() if e.size else float('nan')), q, q/off if off else 0, res["level_mix"]

for tname in ("Solar_Indoor_Moving","RF_2","RF_6"):
    tr=load_trace(f"traces/{tname}.csv")
    print(f"\n### {tname}")
    print(f"  {'deadline':>9}{'ladder':>10}{'policy':>14}{'compl':>7}{'mean err':>10}{'quality':>9}{'q/mJ':>8}   mix")
    for d in (8.0,15.0,25.0,40.0):
        for lname, spec in (("current",CURRENT),("proposed",LADDER)):
            L=build_levels(P, spec)
            best=None
            for l in L:
                r=simulate(tr,L,P,295.0,d,policy_fixed(l.name),forecast_persistence)
                c,me,q,qm,mix=ev(r,L)
                if best is None or q>best[2]: best=(l.name,c,q,qm,me,mix)
            r=simulate(tr,L,P,295.0,d,policy_fixed(L[0].name),forecast_persistence,hindsight=True)
            c,me,q,qm,mix=ev(r,L)
            print(f"  {str(int(d))+'s':>9}{lname:>10}{'best fixed='+best[0]:>14}{best[1]*100:6.0f}%{best[4]:>10.3f}{best[2]:>9.1f}{best[3]:>8.3f}   {best[5]}")
            print(f"  {'':>9}{'':>10}{'ORACLE tier':>14}{c*100:6.0f}%{me:>10.3f}{q:>9.1f}{qm:>8.3f}   {mix}")
