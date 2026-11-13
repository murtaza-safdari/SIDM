#!/usr/bin/env python3
"""The 2026-11-12 pass-fraction table from the rows of xclean_rows.py, on its files: displaced = j040-j059 of
<chan>_200GeV_1p2GeV_4p8mm, prompt-like = j000-j003 of the six PROMPT points; eras 2022 and 2023.
Columns: "all" v1 (stored), v2 (stored), v2 fixed (rebuilt); "all + charge" v1, v2 (stored), v2 fixed.

Usage: xclean_step1.py ROWS_DIR
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xclean_stats import DISPLACED, PROMPT, Rows  # noqa: E402

rows_dir = sys.argv[1]
print("| sample | events v1 / v2 | pre v1 / v2 | all v1 | all v2 | all v2 fixed | all+q v1 | all+q v2 | all+q v2 fixed |")
print("|---|---|---|---|---|---|---|---|---|")
for chan in ["4Mu", "2Mu2E"]:
    for kind, samples, chunks in [("displaced", [DISPLACED.format(chan=chan)], range(40, 60)),
                                  ("prompt-like", [p.format(chan=chan) for p in PROMPT], range(4))]:
        for era in ["2022", "2023"]:
            a, b = Rows(rows_dir, "v1", era, samples, chunks), Rows(rows_dir, "v2", era, samples, chunks)
            vals = [a["all_stored"], b["all_stored"], b["all_formula"], a["allq_stored"], b["allq_stored"],
                    b["allq_formula"]]
            print(f"| {era} {chan} {kind} | {a.n_events:,} / {b.n_events:,} | {len(a):,} / {len(b):,} | "
                  + " | ".join(f"{v.mean():.4f}" for v in vals) + " |")
