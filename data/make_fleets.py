"""Build bigger fleets (plants15/25/50.csv + demand15/25/50.csv) with make_data.py's recipe.

    python make_fleets.py            # writes the six files next to this script

Source: rts_gmlc_gen.csv in this folder, a copy of RTS-GMLC's RTS_Data/SourceData/gen.csv
(the table make_data.py reads; credit DOE/NREL/ALLIANCE, RTS-GMLC). Every unit is a real
RTS-GMLC thermal unit, derived exactly as in make_data.derive: pmax rounded to a multiple of
50 MW (the safety-margin rule is counted in 50 MW blocks), and pmin, no-load, startup and ramp
scaled with it. Marginal cost and min up/down times are the real ones.

The 10 provided plants keep their plants10.csv values. Extra units are added round-robin over
the fuel types (coal steam, gas CC, gas CT, oil CT, oil steam), biggest first within a type, so
every fleet keeps a realistic mix of base-load, mid-merit and peaking units. Nuclear is left out
(RTS-GMLC's one nuclear unit never switches in a day). Coal steam units of 150 MW or more are
on at midnight, like COAL and COAL_MID in the provided data. RTS-GMLC has 72 usable thermal units, so
fleets above 72 reuse units as identical twins (suffix _B), the way RTS-GMLC itself pairs twin units.

Demand is made the way demand10.csv was: the provided 24-hour shape (demand.csv) times the new
fleet's capacity over the 5-plant capacity (850 MW), reserve 10% of demand.
"""
import csv
from pathlib import Path

import numpy as np

from make_data import COLUMNS, derive

HERE = Path(__file__).resolve().parent
GEN = HERE / "rts_gmlc_gen.csv"
SIZES = (15, 25, 50, 100)
TYPE_ORDER = [("Coal", "STEAM", "COAL_ST", "coal steam"),
              ("NG", "CC", "GAS_CC", "gas combined cycle"),
              ("NG", "CT", "GAS_CT", "gas combustion turbine"),
              ("Oil", "CT", "OIL_CT", "oil combustion turbine"),
              ("Oil", "STEAM", "OIL_ST", "oil steam")]
BLOCK_MW, RESERVE, CAP5 = 50, 0.10, 850


def rounded_pmax(pmax):
    return max(BLOCK_MW, int(round(pmax / BLOCK_MW)) * BLOCK_MW)


def build_fleet(n_plants):
    with open(GEN, newline="") as f:
        gen = list(csv.DictReader(f))
    with open(HERE / "plants10.csv", newline="") as f:
        base = list(csv.DictReader(f))
    used = {r["rts_gen_uid"] for r in base}
    pools = []
    for fuel, unit_type, prefix, label in TYPE_ORDER:
        rows = [g for g in gen if g["Fuel"] == fuel and g["Unit Type"] == unit_type and g["GEN UID"] not in used]
        rows.sort(key=lambda g: -float(g["PMax MW"]))
        pools.append((prefix, label, rows))
    fleet, counters = list(base), {}
    twins = []                                   # units already used, to clone once the pool runs out
    while len(fleet) < n_plants:
        added = False
        for prefix, label, rows in pools:
            if rows and len(fleet) < n_plants:
                g = rows.pop(0)
                k = counters[prefix] = counters.get(prefix, 0) + 1
                pmax_new = rounded_pmax(float(g["PMax MW"]))
                on0 = 1 if (prefix == "COAL_ST" and pmax_new >= 150) else 0
                fleet.append(derive(g, f"{prefix}_{k}", label, pmax_new, on0))
                twins.append((g, prefix, label, pmax_new, on0))
                added = True
        if not added:                            # pool exhausted: identical twins of the units above
            for g, prefix, label, pmax_new, on0 in twins:
                if len(fleet) < n_plants:
                    k = counters[prefix + "_B"] = counters.get(prefix + "_B", 0) + 1
                    fleet.append(derive(g, f"{prefix}_{k}_B", label, pmax_new, on0))
            if len(fleet) < n_plants:
                raise ValueError(f"not enough units even with twins: {len(fleet)}")
    return fleet


def main():
    with open(HERE / "demand.csv", newline="") as f:
        shape = np.array([float(r["demand_mw"]) for r in csv.DictReader(f)])
    for n in SIZES:
        fleet = build_fleet(n)
        with open(HERE / f"plants{n}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(fleet)
        cap = sum(float(p["pmax_mw"]) for p in fleet)
        with open(HERE / f"demand{n}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hour", "demand_mw", "reserve_mw"])
            for t, d in enumerate(shape):
                d = round(d * cap / CAP5)
                w.writerow([t, d, round(RESERVE * d)])
        print(f"plants{n}.csv: {n} plants, {cap:.0f} MW; demand{n}.csv peak {round(shape.max() * cap / CAP5)} MW")


if __name__ == "__main__":
    main()
