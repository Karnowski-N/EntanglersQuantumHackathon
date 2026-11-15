"""Build renewables.csv: RTS-GMLC's day-ahead solar and wind forecasts for the same day as demand.csv.

    python make_renewables.py path/to/DAY_AHEAD_pv.csv path/to/DAY_AHEAD_wind.csv \
                              path/to/DAY_AHEAD_rtpv.csv path/to/DAY_AHEAD_regional_Load.csv

You do not need to run this. It documents exactly how renewables.csv was made (credit DOE/NREL/ALLIANCE,
RTS-GMLC). Source files, all under RTS_Data/timeseries_data_files/ in
https://github.com/GridMod/RTS-GMLC: PV/DAY_AHEAD_pv.csv (utility-scale solar), WIND/DAY_AHEAD_wind.csv,
RTPV/DAY_AHEAD_rtpv.csv (rooftop solar) and Load/DAY_AHEAD_regional_Load.csv.

One row per hour of the day demand.csv was made from (July 14 2020, make_data.DAY). The MW columns are
region 3 totals (the region whose load shape became demand.csv; a unit belongs to region 3 when its id
starts with 3): solar_mw sums the 14 PV units, wind_mw the 3 wind farms, rooftop_solar_mw the 20 rooftop
units. region3_load_mw and system_load_mw are the day-ahead load of region 3 and of all three regions,
so a notebook can scale the renewables to a fleet: multiplying by (fleet peak demand / system_load_mw peak)
gives the fleet RTS-GMLC's own renewable share (16.5% of the day's energy from utility solar + wind);
multiplying by (fleet peak demand / region3_load_mw peak) gives region 3's share (55%), which needs
curtailment at night because region 3 exports its wind in RTS-GMLC.
"""
import csv
import sys
from pathlib import Path

from make_data import DAY, REGION

OUT = Path(__file__).resolve().parent / "renewables.csv"


def day_rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if (int(r["Year"]), int(r["Month"]), int(r["Day"])) == DAY]


def region_total(rows, region=REGION):
    units = [c for c in rows[0] if c not in ("Year", "Month", "Day", "Period") and c.startswith(str(region))]
    return [sum(float(r[c]) for c in units) for r in rows], len(units)


def main(pv_path, wind_path, rtpv_path, load_path):
    solar, n_pv = region_total(day_rows(pv_path))
    wind, n_wind = region_total(day_rows(wind_path))
    rooftop, n_rtpv = region_total(day_rows(rtpv_path))
    load = day_rows(load_path)
    region_load = [float(r[str(REGION)]) for r in load]
    system_load = [sum(float(r[k]) for k in ("1", "2", "3")) for r in load]
    assert len(solar) == len(wind) == len(rooftop) == len(load) == 24
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "solar_mw", "wind_mw", "rooftop_solar_mw", "region3_load_mw", "system_load_mw"])
        for t in range(24):
            w.writerow([t, round(solar[t], 1), round(wind[t], 1), round(rooftop[t], 1),
                        round(region_load[t], 1), round(system_load[t], 1)])
    print(f"{OUT.name}: region {REGION}, {DAY}: {n_pv} PV units (peak {max(solar):.0f} MW), {n_wind} wind farms "
          f"(peak {max(wind):.0f} MW), {n_rtpv} rooftop units (peak {max(rooftop):.0f} MW); "
          f"region load peak {max(region_load):.0f} MW, system load peak {max(system_load):.0f} MW")


if __name__ == "__main__":
    main(*sys.argv[1:5])
