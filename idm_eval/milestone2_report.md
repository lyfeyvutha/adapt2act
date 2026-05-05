# Milestone 2 IDM Data-Mixing Evaluation

Status: complete
Rollout runs: 36/36 parsed successfully

## Processed Datasets

| dataset | selected episodes | train pairs | val pairs | mixing |
|---|---:|---:|---:|---|
| mw_equal_100_100 | 200 | 36000 | 4000 | `{"metaworld-door-close": 100, "metaworld-drawer-open": 100}` |
| mw_equal_50_50 | 100 | 18000 | 2000 | `{"metaworld-door-close": 50, "metaworld-drawer-open": 50}` |
| mw_ratio_70_30_total200 | 200 | 36000 | 4000 | `{"metaworld-door-close": 0.7, "metaworld-drawer-open": 0.3}` |
| mw_ratio_70_30_total300 | 300 | 54000 | 6000 | `{"metaworld-door-close": 0.7, "metaworld-drawer-open": 0.3}` |

## MetaWorld Door-Close Evaluation

| dataset | train seed | parsed | success mean | reward mean | best val loss |
|---|---:|---:|---:|---:|---:|
| mw_equal_100_100 | 0 | 3/3 | 0.000000 | 1676.199455 | 0.00302338 |
| mw_equal_100_100 | 1 | 3/3 | 0.000000 | 1767.819270 | 0.00306359 |
| mw_equal_100_100 | 2 | 3/3 | 0.000000 | 1721.352178 | 0.00223021 |
| mw_equal_50_50 | 0 | 3/3 | 0.000000 | 516.947126 | 0.00073319 |
| mw_equal_50_50 | 1 | 3/3 | 0.000000 | 158.093437 | 0.00067247 |
| mw_equal_50_50 | 2 | 3/3 | 0.000000 | 1463.453113 | 0.00068890 |
| mw_ratio_70_30_total200 | 0 | 3/3 | 0.000000 | 1730.457724 | 0.00258674 |
| mw_ratio_70_30_total200 | 1 | 3/3 | 0.000000 | 1788.508321 | 0.00245200 |
| mw_ratio_70_30_total200 | 2 | 3/3 | 0.000000 | 1619.415128 | 0.00017526 |
| mw_ratio_70_30_total300 | 0 | 3/3 | 0.000000 | 512.156131 | 0.00239429 |
| mw_ratio_70_30_total300 | 1 | 3/3 | 0.000000 | 385.948259 | 0.00195391 |
| mw_ratio_70_30_total300 | 2 | 3/3 | 0.000000 | 1580.490427 | 0.00087953 |
