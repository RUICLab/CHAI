"""Read the campaign JSON and say what it actually supports.

Two rules this script exists to enforce, both of which an earlier analysis broke:

1. The comparator is the best FIXED level chosen independently for each
   (benchmark, scenario, deadline, radio cost). Comparing an adaptive policy
   against one level picked in advance is how "reactive beats the best fixed
   level" got published when fixed-L0 in fact dominated it.

2. Better means: completion at least as good, and error strictly lower. A policy
   that trades deadlines for output quality has not won, it has changed the
   operating point. Ties are reported as ties.

Invariants are checked from the recorded per-cell fields, so the campaign is
auditable without re-running it.
"""

from __future__ import annotations

import json
import sys
from math import isnan
from collections import defaultdict

TOL = 1e-9

FIXED = "fixed-"
HINDSIGHT = "hindsight (boundary oracle)"
REACTIVE = "reactive (on-device)"


def load(paths):
    rows, levels = [], {}
    for path in paths:
        with open(path) as handle:
            blob = json.load(handle)
        rows.extend(blob["results"])
        levels[blob["results"][0]["benchmark"]] = blob["levels"]
    return rows, levels


def check_invariants(rows):
    """Every violation, with enough context to find the cell again."""
    bad = []
    for r in rows:
        where = (f"{r['benchmark']}/{r['scenario']}/d={r['deadline_s']:.0f}s/"
                 f"c={r['comms_uJ']:.0f}uJ/{r['policy']}")
        if r["mid_task_switches"] != 0:
            bad.append(f"{where}: {r['mid_task_switches']} mid-task switches")
        if r["decisions"] != r["periods"]:
            bad.append(f"{where}: {r['decisions']} decisions != "
                       f"{r['periods']} periods")
        if r["completed"] > r["periods"]:
            bad.append(f"{where}: completed {r['completed']} > "
                       f"periods {r['periods']}")
        if not (0.0 - TOL <= r["completion"] <= 1.0 + TOL):
            bad.append(f"{where}: completion {r['completion']:.3f} out of range")
        if r["energy_to_load_mJ"] > r["energy_harvested_mJ"] + TOL:
            bad.append(f"{where}: load {r['energy_to_load_mJ']:.2f} mJ > "
                       f"harvested {r['energy_harvested_mJ']:.2f} mJ")
    return bad


def hindsight_notes(cells):
    """Cells where a fixed level beat the boundary oracle.

    This is NOT an invariant violation, though an earlier version of this file
    treated it as one -- contradicting the oracle's own docstring. The boundary
    oracle maximises each period in isolation, so it can spend charge that a
    later period needed and finish fewer tasks overall. Sobel never triggered
    it; fft/mixed at an 8 s deadline does, with cheapest-fixed at 91.1% against
    the oracle's 88.9%.

    It is reported because it is informative -- it marks cells where per-period
    greed is actively the wrong strategy -- not because anything is wrong.
    """
    bad = []
    for key, byname in cells.items():
        hind = byname.get(HINDSIGHT)
        if hind is None:
            continue
        for name, row in byname.items():
            if not name.startswith(FIXED):
                continue
            if row["completion"] > hind["completion"] + TOL:
                bad.append(f"{fmt_key(key)}: {name} completion "
                           f"{row['completion']:.3f} > hindsight "
                           f"{hind['completion']:.3f}")
            elif (abs(row["completion"] - hind["completion"]) <= TOL
                  and row["mean_error"] < hind["mean_error"] - TOL):
                bad.append(f"{fmt_key(key)}: {name} error "
                           f"{row['mean_error']:.4f} < hindsight "
                           f"{hind['mean_error']:.4f} at equal completion")
    return bad


def fmt_key(key):
    bench, scen, deadline, comms = key
    return f"{bench}/{scen}/d={deadline:.0f}s/c={comms:.0f}uJ"


def group(rows):
    """(benchmark, scenario, deadline, comms) -> {policy: row}.

    Non-server policies were only simulated at comms=0, so they are copied into
    the paid-radio cells too: their cost does not change, and without them the
    paid cells would have no fixed comparator at all.
    """
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["benchmark"], r["scenario"], r["deadline_s"],
               r["comms_uJ"])][r["policy"]] = r
    free = {k: v for k, v in cells.items() if k[3] == 0.0}
    for key, byname in cells.items():
        if key[3] == 0.0:
            continue
        for name, row in free.get((key[0], key[1], key[2], 0.0), {}).items():
            byname.setdefault(name, row)
    return cells


def best_fixed(byname):
    """Most completions; ties broken by lower error. Chosen per cell."""
    fixed = [(n, r) for n, r in byname.items() if n.startswith(FIXED)]
    return max(fixed, key=lambda nr: (nr[1]["completion"], -nr[1]["mean_error"]))


def verdict(row, ref):
    """How `row` compares to the best fixed level in its own cell.

    A policy that completed nothing has no mean error -- it is NaN, not zero.
    NaN fails every comparison silently, so it is handled explicitly: without
    this, beating a reference that completed nothing was reported as "more
    tasks, worse error", and two policies that both completed nothing compared
    as a tie on an error neither of them has.
    """
    dc = row["completion"] - ref["completion"]
    row_err, ref_err = row["mean_error"], ref["mean_error"]
    if dc < -TOL:
        return "loses completion"
    if dc > TOL:
        # No error to lose against when the reference produced no output at all.
        if isnan(ref_err):
            return "wins completion"
        return "wins completion" if row_err <= ref_err + TOL \
            else "more tasks, worse error"
    if isnan(row_err) or isnan(ref_err):
        return "ties"          # neither completed anything; nothing to compare
    if row_err < ref_err - TOL:
        return "wins error"
    if row_err > ref_err + TOL:
        return "loses error"
    return "ties"


def in_bound_tasks(row, errors, bound):
    """Completed tasks whose level satisfies the error bound.

    Tasks above the bound score nothing: under "results per joule subject to
    error <= bound" an out-of-bound result is not a cheaper result, it is not a
    result. level_mix counts only completed tasks, so this needs no correction
    for deadline misses.
    """
    return sum(c for c, e in zip(row["level_mix"], errors)
               if e <= bound + 1e-12)


def efficiency(row, errors, bound):
    """The same throughput under three denominators, because they disagree.

    A policy that declines to start tasks keeps the capacitor fuller, so more
    incoming harvest is rejected while the supply is clamped. That rejected
    energy leaves `energy_harvested` entirely. Dividing by harvest alone
    therefore rewards a policy for capturing LESS energy, which looks identical
    to doing work more efficiently.

      per_harvest  tasks / energy accepted into the capacitor -- the original
                   metric, and the one open to that artifact.
      per_load     tasks / energy the MCU actually consumed. This is the direct
                   test of the defer mechanism: did declining doomed tasks stop
                   energy being burned for nothing?
      per_offered  tasks / (harvested + spilled), roughly all the energy the
                   environment made available. Rejected energy cannot vanish,
                   so a policy that spills more is charged for it.

    If defer wins on per_load it avoided wasted work. If it wins only on
    per_harvest the gain is accounting.
    """
    n = in_bound_tasks(row, errors, bound)
    harvested = row["energy_harvested_mJ"]
    spilled = row.get("spill_mJ", 0.0)
    load = row["energy_to_load_mJ"]
    return {
        "tasks": n,
        "per_harvest": n / harvested if harvested else 0.0,
        "per_load": n / load if load else 0.0,
        "per_offered": n / (harvested + spilled) if (harvested + spilled) else 0.0,
    }


def report_efficiency(cells, levels_by_bench, bound):
    """Defer and its control, side by side, under all three denominators."""
    interesting = [n for n in
                   {p for b in cells.values() for p in b}
                   if n.startswith("bound-aware")]
    if not interesting:
        return
    print(f"\nEFFICIENCY under three denominators   (error bound {bound})")
    print("  per_harvest can reward a policy for capturing less energy;")
    print("  per_load tests the mechanism; per_offered charges for spill.\n")
    print(f"  {'cell':<30}{'policy':<26}{'compl':>7}{'def':>5}"
          f"{'/harvest':>10}{'/load':>9}{'/offered':>10}")
    for key in sorted(cells):
        if key[3] != 0.0:
            continue
        byname = cells[key]
        errors = [l["error"] for l in levels_by_bench[key[0]]]
        ref = byname.get("fixed-L1")
        shown = [("fixed-L1", ref)] if ref else []
        shown += [(n, byname[n]) for n in sorted(byname) if n.startswith("bound-aware")]
        base = efficiency(ref, errors, bound) if ref else None
        for name, row in shown:
            if row is None:
                continue
            e = efficiency(row, errors, bound)
            tail = ""
            if base and name != "fixed-L1":
                def pct(a, b):
                    return f"{(a / b - 1) * 100:+.1f}%" if b else "   n/a"
                tail = (f"   [{pct(e['per_harvest'], base['per_harvest'])}"
                        f" {pct(e['per_load'], base['per_load'])}"
                        f" {pct(e['per_offered'], base['per_offered'])}]")
            print(f"  {fmt_key(key):<30}{name:<26}"
                  f"{row['completion'] * 100:6.0f}%{row.get('deferred', 0):>5}"
                  f"{e['per_harvest']:>10.3f}{e['per_load']:>9.3f}"
                  f"{e['per_offered']:>10.3f}{tail}")


def main():
    # The error bound is the single most consequential knob in this analysis and
    # it must be passed explicitly. A hardcoded 0.05 silently drove the headline:
    # at 0.05 one level dominates almost everywhere and adaptation has nothing to
    # decide, while at the project's own CHAI_ERROR_BOUND (0.30) the best fixed
    # level varies by scenario and deadline and real forecasters win some cells.
    args = [a for a in sys.argv[1:] if not a.startswith("--bound")]
    bound = next((float(a.split("=", 1)[1]) for a in sys.argv[1:]
                  if a.startswith("--bound=")), None)
    if bound is None:
        from config.config import ERROR_BOUND as bound
    paths = args or ["reports/adaptive/results_sobel.json",
                     "reports/adaptive/results_fft.json"]
    rows, levels = load(paths)
    cells = group(rows)

    print(f"{len(rows)} cells from {len(paths)} file(s)\n")

    bad = check_invariants(rows)
    print("INVARIANTS")
    if bad:
        for line in bad[:40]:
            print(f"  FAIL  {line}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        print(f"\n  {len(bad)} VIOLATIONS -- results are not trustworthy\n")
    else:
        print("  all pass: no mid-task switches, decisions == periods, "
              "completion in range,\n  energy conserved\n")

    notes = hindsight_notes(cells)
    if notes:
        print("NOTE  per-period greed loses to a fixed level in these cells.")
        print("      Expected: the boundary oracle optimises each period alone,")
        print("      so it can spend charge a later period needed.")
        for line in notes:
            print(f"        {line}")
        print()

    # What the oracle says is available, cell by cell. This is the headroom any
    # realisable policy is competing for; where it is zero there is nothing to win.
    print("HEADROOM (boundary oracle vs best fixed level, per cell)")
    print(f"  {'cell':<34}{'best fixed':<12}{'fixed':>14}"
          f"{'hindsight':>14}   headroom")
    headroom = {}
    for key in sorted(cells):
        byname = cells[key]
        if HINDSIGHT not in byname:
            continue
        name, ref = best_fixed(byname)
        hind = byname[HINDSIGHT]
        gap_c = hind["completion"] - ref["completion"]
        gap_e = ref["mean_error"] - hind["mean_error"]
        headroom[key] = (name, ref, hind, gap_c, gap_e)
        note = ("none" if gap_c <= TOL and gap_e <= TOL
                else f"+{gap_c * 100:.0f}pp / -{gap_e:.3f} err")
        print(f"  {fmt_key(key):<34}{name:<12}"
              f"{ref['completion'] * 100:6.0f}% {ref['mean_error']:6.3f}"
              f"{hind['completion'] * 100:6.0f}% {hind['mean_error']:6.3f}"
              f"   {note}")

    print("\nREALISABLE POLICIES vs the best fixed level in the same cell")
    tally = defaultdict(lambda: defaultdict(int))
    for key in sorted(cells):
        byname = cells[key]
        if key not in headroom:
            continue
        _, ref, _, _, _ = headroom[key]
        for name, row in byname.items():
            if name.startswith(FIXED) or name == HINDSIGHT:
                continue
            if name == "adaptive-oracle":
                continue          # clairvoyant, reported separately
            tally[name][verdict(row, ref)] += 1
    order = ["wins error", "wins completion", "ties", "more tasks, worse error",
             "loses error", "loses completion"]
    print(f"  {'policy':<24}" + "".join(f"{o:>24}" for o in order))
    for name in sorted(tally):
        print(f"  {name:<24}"
              + "".join(f"{tally[name][o]:>24}" for o in order))

    wins = [(k, v) for k, v in headroom.items() if v[3] > TOL or v[4] > TOL]
    print(f"\n  cells with any headroom at all: {len(wins)}/{len(headroom)}")
    report_efficiency(cells, levels, bound)

    reactive_wins = sum(1 for k in headroom
                        if REACTIVE in cells[k]
                        and verdict(cells[k][REACTIVE], headroom[k][1])
                        in ("wins error", "wins completion"))
    print(f"  cells where {REACTIVE} beats the best fixed level: "
          f"{reactive_wins}/{len(headroom)}")


if __name__ == "__main__":
    main()
