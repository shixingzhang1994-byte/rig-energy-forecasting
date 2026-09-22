# 论文当前状态

最后更新：2026-09-22

## 结论

CAS 投稿候选已完成证据同步并于 2026-09-22 编译为 20 个 PDF 页面（1 页 Highlights，正文编号 19 页）。摘要、结果、讨论、局限、结论和数据可用性声明已纳入 V24、V27 与 V28。当前状态是“论文证据版完成，等待作者核验与投稿预检”，尚不能称为已提交或现场验证完成。

## 投稿前 P0

- 由作者逐项核验所有单位、CRediT、共同第一作者、通讯作者、利益冲突、资助来源及资助方角色。
- 将 V24/V28 配置、清单、派生表和分析脚本加入新的公开归档版本；当前 v1.0.0 不能复现新增结果。
- 按目标期刊当前作者指南执行格式、页数、匿名/非匿名、声明和投稿清单预检。
- 保留“真实测量外部验证不等于目标钻机闭环现场验证”的措辞边界；不得把 4/6 证据块核算写成 66.7% 现场验证。
- 参考文献已完成全库在线核验：`PASS: 50/50 verified, 0 errors, 0 warnings (0 checks skipped)`；3 条未在正文引用且存在元数据问题的冗余记录已删除。
- CAS 模板仍在 front matter 生成约 117 pt 的内部 `Overfull \\hbox` 日志提示；渲染检查未见页面裁切或可见越界。
- 由作者确认作者单位、CRediT、利益冲突、资助来源角色与最终仓库链接。

## 当前入口

- 当前稿：[`submission/main.tex`](submission/main.tex)
- 当前 PDF：[`submission/main.pdf`](submission/main.pdf)
- V24 证据：[`../rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json`](../rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json)
- V27 失败单元：[`../rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json`](../rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json)
- V28 最终证据：[`../rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json`](../rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json)
- 当前 claims matrix：[`review/claims-matrix-v28.md`](review/claims-matrix-v28.md)
- 引用检查：[`review/citation-check-v28-online.json`](review/citation-check-v28-online.json)

旧的 40 页 `elsarticle` 稿和早期预检记录只用于追溯，不能继续称为当前投稿稿。当前可引用的结论必须以 V24/V28 证据台账与本稿为准。
