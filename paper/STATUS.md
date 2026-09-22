# 论文当前状态

最后更新：2026-09-22

## 结论

CAS 投稿候选已完成证据同步和技术预检，并于 2026-09-22 编译为 21 个 PDF 页面（1 页 Highlights，正文编号 20 页）。摘要、结果、讨论、局限、结论和数据可用性声明已纳入 V24、V27 与 V28，数据可用性已改指 V24/V28 公开归档标签。当前状态是“技术文件通过本地预检，等待作者事实确认与投稿系统人工核验”，尚不能称为已提交或现场验证完成。

## 投稿前 P0

- 由作者逐项核验所有单位、CRediT、共同第一作者、通讯作者、利益冲突、资助来源及资助方角色；仓库只能确认内部一致性，不能替作者作事实签署。
- 在 Editorial Manager 中人工确认图形摘要和独立利益冲突 `.docx` 是否为本稿必需；当前已有 `Highlights.docx`，没有图形摘要文件。
- V24/V28 公开 Git 包已建立；DOI 候选归档完成后仍需实际上传并回填 DOI，旧 v1.0.0 继续只代表 V21。
- 保留“真实测量外部验证不等于目标钻机闭环现场验证”的措辞边界；不得把 4/6 证据块核算写成 66.7% 现场验证。
- 参考文献已完成全库在线核验：`PASS: 50/50 verified, 0 errors, 0 warnings (0 checks skipped)`；3 条未在正文引用且存在元数据问题的冗余记录已删除。
- CAS 模板仍在 front matter 生成约 117 pt 的内部 `Overfull \\hbox` 日志提示；21 页全量渲染检查未见页面裁切或可见越界，所有字体均已嵌入。
- 预检报告：[`preflight/PREFLIGHT_REPORT_20260922.md`](preflight/PREFLIGHT_REPORT_20260922.md)；通用检查器对多行 AI 标题和单数 `competing interest` 各有一项已人工核实的误报。

## 当前入口

- 当前稿：[`submission/main.tex`](submission/main.tex)
- 当前 PDF：[`submission/main.pdf`](submission/main.pdf)
- V24 证据：[`../rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json`](../rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json)
- V27 失败单元：[`../rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json`](../rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json)
- V28 最终证据：[`../rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json`](../rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json)
- 当前 claims matrix：[`review/claims-matrix-v28.md`](review/claims-matrix-v28.md)
- 引用检查：[`review/citation-check-v28-online.json`](review/citation-check-v28-online.json)

旧的 40 页 `elsarticle` 稿和早期预检记录只用于追溯，不能继续称为当前投稿稿。当前可引用的结论必须以 V24/V28 证据台账与本稿为准。
