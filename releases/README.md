# 发布包

本目录只放可复现发布物，不作为日常开发目录。

| 路径 | 状态 | 用途 |
|---|---|---|
| `github-reproducibility-package/` | V21 冻结、独立 Git 仓库 | GitHub 发布工作副本 |
| `reserve_soc_reproducibility_package_v1.0.0-draft.zip` | V21 Zenodo 草稿；压缩完整性和 SHA-256 已于 2026-09-21 验证 | 归档/上传候选 |
| `reserve_soc_reproducibility_package_v1.0.0-draft.zip.sha256` | 与 ZIP 匹配 | 完整性校验 |

原先并列存在的两个解压 Zenodo 草稿目录已清理，因为 ZIP 完整包含 `reproducibility_package_zenodo_draft_v2/`，且其内部 `MANIFEST.sha256` 校验通过。V24、V27 和 2026-09-21 外部数据审计尚未进入本发布包。

```bash
sha256sum -c reserve_soc_reproducibility_package_v1.0.0-draft.zip.sha256
unzip -t reserve_soc_reproducibility_package_v1.0.0-draft.zip
```
