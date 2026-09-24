# PG-500 waiting-time data pipeline

面向澳大利亚 Student visa（subclass 500）等签时长研究的可追溯数据与基线分析项目。

## 快速使用

```powershell
python 等签算法.py collect visadashboard --snapshot-date 2026-09-24
python 等签算法.py official-refresh
python 等签算法.py validate
python 等签算法.py profile
python 等签算法.py audit
python 等签算法.py predict --education-level PhD --submit-location "Outside Australia"
python 等签算法.py train
python 等签算法.py forecast --lodged-at "2026-09-24T15:30+08:00" --study-sector "Postgraduate Research"
python 等签算法.py visualize
python -m unittest discover -s tests -v
```

Windows 上使用现有 `C:\Users\saint\cqfenv` 环境的一键运行方式、在线刷新方式和 notebook 命令见 [RUNBOOK.md](RUNBOOK.md)。

采集器默认遵守 `robots.txt`、限速、重试，且只调用公开只读接口。输出使用原子替换，抓取失败不会破坏上一版数据。

## 数据保护与质量

- 不落库源站 `_id`、签证官编号、账号、用户名或自由文本备注。
- 专业、学校等字段只保留宽泛分组；每行用不可逆哈希形成稳定记录号。
- 已获签案例记录事件时间；未获签案例在快照日右删失，适合生存分析。
- `record_quality_score < 40` 的记录默认不进入基线预测。
- 基线预测按 `duplicate_key` 折叠疑似重复时间线，原始标准化表仍保留全部源记录以便审计。
- 社媒适配器默认不抓取 X、微博、小红书或封闭群。只接受符合平台规则的公开 URL、官方 API 或经同意的结构化导出。

## 模型输出怎么读

`predict` 是 Kaplan–Meier 历史经验基线，能利用仍在等待的记录，并通过 bootstrap 给出中位数估计的 95% 区间。`p10/p50/p90` 是已观察样本的处理时长分位，不是个人获签承诺；`confidence_grade` 衡量样本量和删失程度，不代表签证结果概率。

`train` 会在 2024-12-31 截止点重建右删失训练集，并用 2025-01 至 2025-08 递签批次做时间外测试；比较全局/分层 Kaplan–Meier、Cox PH、Weibull AFT 与随机生存森林。官方 BP0015 流量只以一个月滞后的上下文代理进入 PGR/境外记录，且不会被解释为在审库存。

`forecast` 默认读取序列化的 v3 Weibull AFT，输出在当前仍未决定这一条件下的 P10/P50/P80/P90、未来 30/60/90/180 天决定概率、80% 预测区间和基于最终留出集的数据置信等级。分钟会从递签时间传递到结果，但训练标签大多只有日期精度，因此模型本质分辨率仍是天。当前最终留出区分度有限，输出会明确标记 `limited`；可用 `--engine anchor` 调用较保守的 Home Affairs P50/P90 锚定方法作对照。训练产物见 `models/subclass500_survival_v3.json`，数据问题见 `reports/data_quality_audit.md`。

## 合规社媒采集

```powershell
# X 官方 recent-search API；令牌仅从环境变量读取，绝不写入数据文件
$env:X_BEARER_TOKEN="..."
python 等签算法.py social-collect x

# 微博、小红书、X、Reddit 等获准公开 URL；每行一个 URL
python 等签算法.py social-collect urls --input config/social_public_urls.txt
```

社媒适配器不存储原文、账号、用户名或头像，只保留日期、宽泛特征、来源 URL 和内容哈希。解析不完整的记录标记 `requires_manual_review=yes`；不绕过登录、验证码、付费墙、robots.txt 或反爬机制。

官方统计、众包个案和社媒个案不可直接等权混合。数据口径与来源清单见 [data/README.md](data/README.md)。
