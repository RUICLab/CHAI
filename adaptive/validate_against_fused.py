"""Validate adaptive.power_model against a Fused simulator log.

Usage: python -m adaptive.validate_against_fused [ext.tab] [trace.csv]

Two checks, both against Fused's own ``ext.tab``:
1. Charging phase (MCU off): harvester current and capacitor voltage.
2. Full run: Fused's recorded MCU current is replayed as the load, gated by the
   Python supervisor, and the capacitor voltage and powered state are compared
   across every power cycle.
"""

from __future__ import annotations

import sys

import numpy as np

from adaptive.power_model import PowerParams, load_trace, simulate


def load_ext_tab(path):
    cols, rows = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith("%"):
                cols = line[1:].split()
                continue
            rows.append([float(x) for x in line.split()])
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(cols)}


def main(ext_path="fusedBin/fusedLogs/ext.tab", trace_path="traces/RF_2.csv"):
    log = load_ext_tab(ext_path)  # replays Fused's own icc and keepAlive signals
    trace = load_trace(trace_path)
    params = PowerParams()
    vcap, isup, vcc, icc = (log["externalCircuitry.v_cap"], log["externalCircuitry.i_supply"],
                            log["vcc"], log["icc"])
    n = len(vcap)
    first_boot = int(np.argmax(vcc > 0))

    charge = simulate(trace, 0.0, first_boot, params)
    err_i = np.abs(charge.i_harvest - isup[:first_boot]).max()
    err_v = np.abs(charge.v_cap - vcap[:first_boot]).max()
    print(f"[1] charging phase, {first_boot} steps")
    print(f"    harvester current max |err| = {err_i:.3e} A")
    print(f"    capacitor voltage max |err| = {err_v:.3e} V")

    keep = log["externalCircuitry.keepAlive"]
    full = simulate(trace, lambda k, on: icc[k], n, params,
                    keep_alive=lambda k, v, running: bool(keep[k]))
    fused_on = vcc > 0
    ev = np.abs(full.v_cap - vcap)
    agree = (full.powered == fused_on).mean()
    fused_boots = int(np.sum(fused_on[1:] & ~fused_on[:-1]))
    print(f"[2] full run, {n} steps ({n * params.timestep:.1f} s simulated)")
    print(f"    capacitor voltage max |err| = {ev.max():.3e} V, mean {ev.mean():.3e} V")
    print(f"    powered-state agreement      = {agree * 100:.3f}%")
    print(f"    boots: fused {fused_boots}, python {full.boots}")
    print(f"    energy harvested {full.energy_harvested * 1e3:.3f} mJ, "
          f"spilled {full.energy_spilled * 1e3:.3f} mJ, "
          f"to MCU {full.energy_to_load * 1e3:.3f} mJ, "
          f"to supervisor {full.energy_to_supervisor * 1e3:.3f} mJ")
    return err_i, err_v, ev.max(), agree, fused_boots, full.boots


if __name__ == "__main__":
    main(*sys.argv[1:])
