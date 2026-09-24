# PG-500 waiting-time data pipeline

面向澳大利亚 Student visa（subclass 500）等签时长研究的可追溯数据与基线分析项目。

## 快速使用

```powershell
python 等签算法.py collect visadashboard --snapshot-date 2026-09-24
python 等签算法.py validate
python 等签算法.py profile
python 等签算法.py predict --education-level PhD --submit-location "Outside Australia"
python -m unittest discover -s tests -v
```

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

官方统计、众包个案和社媒个案不可直接等权混合。数据口径与来源清单见 [data/README.md](data/README.md)。
