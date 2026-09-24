# 项目运行手册

以下命令都在项目根目录 `C:\Users\saint\Desktop\澳洲等签分析` 运行。

## 1. 使用现有 cqfenv 环境

```powershell
Set-Location "C:\Users\saint\Desktop\澳洲等签分析"
$py = "C:\Users\saint\cqfenv\python.exe"
& $py --version
& $py -c "import nbformat, nbclient, ipykernel; print('notebook environment ready')"
```

本项目的核心流水线只使用 Python 标准库，不需要额外安装包。`cqfenv` 已包含 Jupyter、nbformat、nbclient 和 ipykernel。

## 2. 一键运行现有数据、审计、训练、图表、测试和预测

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -ExecuteNotebook
```

此命令不会重新抓取互联网数据，只使用仓库中已经保存的数据，因此最适合第一次验证项目是否跑通。

## 3. 更新在线数据后重新训练

```powershell
# 更新 Home Affairs BP0015 官方工作簿并刷新月度流量
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -RefreshOfficial -ExecuteNotebook

# 同时刷新 VisaDashboard 众包个案；该步骤取决于网站公开接口是否可用
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -RefreshOfficial -RefreshCrowd -ExecuteNotebook
```

在线刷新失败时，原有规范数据不会被覆盖；写入采用临时文件加原子替换。

## 4. 单独运行预测

默认使用可部署的 v3 Weibull AFT 模型：

```powershell
& $py -m visa_wait.cli forecast `
  --lodged-at "2026-09-24T15:30+08:00" `
  --as-of "2026-09-24T15:30+08:00" `
  --education-level PhD `
  --study-sector "Postgraduate Research" `
  --submit-location "Outside Australia"
```

对已经等待一段时间的申请，将 `--as-of` 改成当前时间。输出的 P10/P50/P80/P90 是在该时点仍未决定这一条件下的决策时间分布。

保守的 Home Affairs 当前 P50/P90 锚定版本仍可使用：

```powershell
& $py -m visa_wait.cli forecast --engine anchor `
  --lodged-at "2026-09-24T15:30+08:00" `
  --study-sector "Postgraduate Research" `
  --submit-location "Outside Australia"
```

## 5. 打开或重跑 notebook

交互打开：

```powershell
& "C:\Users\saint\cqfenv\Scripts\jupyter-lab.exe" .\notebooks\01_data_audit_and_survival_baselines.ipynb
```

无界面重跑：

```powershell
& $py .\scripts\execute_notebook.py `
  .\notebooks\01_data_audit_and_survival_baselines.ipynb `
  --kernel cqfenv --timeout 300 `
  --html-output .\reports\notebook_data_audit_and_survival.html
```

项目使用 `nbclient` 直接执行，以避开用户级 `nbconvert` 配置对 `jupyter_contrib_nbextensions` 的可选依赖。

## 6. 主要产物

- `data/14_official_monthly_flows_2015-2026.csv`：官方月度流量。
- `reports/data_quality_audit.md`：众包个案质量审计。
- `models/subclass500_survival_v3.json`：包含编码器、完整模型参数、时间切分与留出验证指标的部署 artifact。
- `reports/figures/backtest_calibration.svg`：模型比较图。
- `notebooks/01_data_audit_and_survival_baselines.ipynb`：带输出的可复现分析。

## 7. 如何判断是否跑通

满足以下条件即为成功：

1. `validate` 返回 `valid: true`。
2. 测试显示全部 `OK`。
3. `models/subclass500_survival_v3.json` 的 `deployment.engine` 为 `weibull_aft`。
4. `forecast` 返回递增的 P10、P50、P80、P90 时间。
5. notebook 执行结束且没有 `CellExecutionError`。

当前部署置信等级为 `limited`。这不是程序故障，而是留出验证显示排序能力有限、样本来自单一自选择平台；前端必须展示此等级和 80% 区间，不应只展示一个日期。
