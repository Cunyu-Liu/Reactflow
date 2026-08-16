# ReactFlow-Delta D0-X 来源内容冻结报告

已冻结 RMDB 当前五组 release 的 metadata-only asset universe：共 1,024 个唯一 RDAT asset，和历史 1,024 accession seed 一一匹配；每个 asset 均保存上游 GitHub asset ID、字节数、release tag、URL 和 `sha256:` digest。

这不是 D0-X 数据 Gate PASS：1,024 条的初始 disposition 全为 `NOT_SEARCHED`，raw RDAT 召回、既有 raw requalification、profile parse、非 RMDB 渠道搜索、人工抽查和 exact-pair 计数均未执行。`exact_delta_pairs` 保持 `null / NOT_RUN_NOT_ZERO`。

首次 Python HTTPS 请求因远端 CA 链不完整而失败，未生成 snapshot；随后使用系统 HTTPS 客户端（不关闭 TLS 校验）将五个 GitHub API JSON 保存到 Git 外部 artifact root，再由已测试的同一归一化器离线验证和冻结。

关键字节：

- asset JSONL SHA-256：`f8b421a798ba23e947457497661ffc6ede78f0c6c5a01f74f37a29a86f4ff254`
- summary SHA-256：`8013a14b3851f3ada290fb8b15b986245f4a09f4202edf0aaef67cd14c83bdb0`
- RMDB repository commit：`339b4fefc9a7092d0847d1d4017a3eadf0771fd7`
- generator source commit：`3e9d596ade3568beeba0f538ab762401d589d03e`

没有下载任何 RDAT payload，没有生成 dataset/split，没有访问 test，也没有训练。

