# Dashboard Ejecutivo (Bloque 3)

Script principal:
- `scripts/build_model_dashboard_executive.py`

## Objetivo
Consolidar resultados de:
- `model_dashboard_v1` (rendimiento general + estabilidad)
- `model_dashboard_v2_probcal` (probabilidad + calibracion)

para entregar:
- ranking final ponderado,
- recomendacion por escenario,
- reporte ejecutivo HTML unico.

## Ejecucion
```powershell
python scripts/build_model_dashboard_executive.py `
  --block1-root artifacts/model_dashboard_v1 `
  --block2-root artifacts/model_dashboard_v2_probcal `
  --output-root artifacts/model_dashboard_v3_executive
```

## Salidas
- `artifacts/model_dashboard_v3_executive/logs/executive_model_table.csv`
- `artifacts/model_dashboard_v3_executive/logs/ranking_profiles.csv`
- `artifacts/model_dashboard_v3_executive/logs/profile_weights.json`
- `artifacts/model_dashboard_v3_executive/logs/recommendations.json`
- `artifacts/model_dashboard_v3_executive/logs/executive_summary.json`
- `artifacts/model_dashboard_v3_executive/plots/pareto_f1_vs_ece.(png|html)`
- `artifacts/model_dashboard_v3_executive/plots/profile_scores.(png|html)`
- `artifacts/model_dashboard_v3_executive/plots/rank_heatmap_profiles.(png|html)`
- `artifacts/model_dashboard_v3_executive/plots/top_models_radar.(png|html)`
- `artifacts/model_dashboard_v3_executive/report/executive_report.html`

## Notas
- Perfil `performance_first`: prioriza F1-macro y accuracy.
- Perfil `calibration_first`: prioriza ECE, log-loss y brier.
- Perfil `balanced_reliable`: combina rendimiento, calibracion y estabilidad.
