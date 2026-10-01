| Scenario | Search reference | Fresh mean [min, max] tok/s/GPU | Mean total goodput tok/s | Mean SLA pass | Mean allocated GPUs | Fresh replays |
|---|---:|---:|---:|---:|---:|---:|
| Static | 34.3868 | 34.3868 [34.3868, 34.3868] | 1375.471 | 78.51% | 40.000 | 1 |
| + KV Router | 23.2101 | 23.2134 [23.1997, 23.2211] | 1671.362 | 89.38% | 72.000 | 3 |
| + Planner | 41.9101 | 41.7679 [41.6052, 41.8529] | 1049.319 | 58.32% | 25.123 | 3 |
