# ReactFlow-Delta D0-X authority 激活报告

D0-X 已配置为唯一 runnable phase；其权限仅覆盖冻结 source universe 的召回、hash requalification、profile parse、crosswalk、coverage/retention audit 和 candidate inventory。

绑定：

- execution source commit：`d63f4c2fda683065040a91decc3a9febb1ecb2c2`
- source universe SHA-256：`b8aabd83a4eab37d86baa4c47d1b4f8b7116cdf6e2018b96c9d68229078c6df1`
- D0 config SHA-256：`214716e6fd51d583d935105799ba78278ede7fd3b2dde3d49d19a72596bf4b4f`
- license policy SHA-256：`f03f11beebe2e8371b9ec241c0d847cf019e804afc629237f076467f01201965`
- parser fixture manifest SHA-256：`8f44b9abf8b3f9ef4d12498cb652c992628ef3fcf699cb533fccc2f361ac2bbb`

9 项 parser/snapshot tests 已通过；authority preflight 必须在实际网络/raw 读取之前再次通过，并且实际执行必须位于 isolated clean worktree。当前批次没有执行 bulk raw recall。

以下权限继续为 false：D1-X、dataset、split、normalization/threshold fitting、baseline/P2/EPRO、confirmatory test、跨项目导出、湿实验、push 和 PR。

