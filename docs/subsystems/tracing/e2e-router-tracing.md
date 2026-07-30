# Router → OpenInfer E2E 请求链路追踪

> **TL;DR:** client → vllm-router → openinfer → prefill/decode 单 trace 已打通并验证（Tempo/Grafana 本地栈）：router 开 OTel 即导出 span 并注入 traceparent；openinfer 侧 `openinfer-vllm-frontend/src/trace_context.rs` middleware 提取 traceparent、按 `X-Request-Id → external_req_id`（容忍 `cmpl-`/`chatcmpl-` 前缀）关联，bridge 以其为 `request` 根 span 的 parent。上游 PR vllm-project/vllm#50370（HTTP 层填 `trace_headers`）合入后按 openinfer#790 迁移并删除 middleware。
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — 路由表；无现成 tracing 子系统文档（此前 tracing 工作只在 deploy/tracing + 代码里）
  - `deploy/tracing/docker-compose.yml` / `tempo.yaml` — 已有 Tempo+Grafana 本地栈，OTLP gRPC 4317，`OPENINFER_TRACE_OTLP_ENDPOINT` 指向即用；已发 span：`request → {queue, prefill, decode}`
  - `openinfer-core/src/tracing.rs` — fastrace → OTLP 导出初始化；`OPENINFER_TRACE_OTLP_ENDPOINT` 未设时完全无开销
  - `openinfer-vllm-frontend/src/bridge.rs` — `Span::root("request", SpanContext::random())`（bridge.rs:396）：**目前总是开新 trace**，这是断链点；`EngineCoreRequest` 由 ZMQ 从 vllm-server 送来
  - `openinfer-qwen3/src/scheduler/phase_trace.rs` — queue/prefill/decode 作为 `request` 的子 span（经 `GenerateRequest.trace_parent`）
  - `openinfer-vllm-frontend/src/lib.rs` — `vllm_server::serve_with_router_extension(config, shutdown, extend_router)` 提供 `FnOnce(Router) -> Router` 钩子，可在不 patch git 依赖的情况下加 middleware
  - vllm-server（git pin 8e61b64 + 上游 main 均确认）— `resolve_request_context` 只提取 `X-Request-Id`/`X-data-parallel-rank`；协议里 `EngineCoreRequest.trace_headers` 字段存在但 HTTP 层从不填它；**升 pin 不能解决问题**
  - vllm-router（本地 /data/code/workspace-rustllm/router）— `--enable-trace` + `--otlp-traces-endpoint`；server span `http_request`（提取 client 的 traceparent 作 parent）+ client span `http_client_request`，并把 span context 注入 `traceparent`/`tracestate` 发往 worker；OTel 关闭时也原样转发 client 的 trace 头；所有 client header（含 `X-Request-Id`）原样转发；`service.name=vllm-router`，endpoint 裸 host:port 会自动补 `http://`
  - fastrace 0.7.17 — `SpanContext::decode_w3c_traceparent(&str) -> Option<SpanContext>`（`collector/id.rs:281`，pub）；`EngineCoreRequest.external_req_id` 可作关联键
  - `openinfer-engine/src/tracing_state.rs` — 全局 AtomicBool 开关；测试中不改它（并行竞争），故 middleware 核心做成不读全局标志的纯函数单测

## Execution Log

### Step 1: openinfer 侧 W3C trace context 提取（MVP）
- 新增 `openinfer-vllm-frontend/src/trace_context.rs`：`TraceContextStash`（Arc<Mutex<HashMap>>，TTL 120s / CAP 4096，take 一次性弹出）+ `stash_trace_context` axum middleware；核心逻辑 `stash_from_headers` 不读全局开关，便于无竞争单测
- `lib.rs`：`mod trace_context`；stash 在 `serve_model_on_host_with_router_extension` 顶部创建，clone 进 engine task（bridge 字段）与 extend_router 包装（`from_fn_with_state` layer 包最外层，保证先于 vllm-server 读头）
- `bridge.rs`：`LocalEngineBridge` 加 `trace_stash` 字段；`start_request` 解构 `external_req_id`；tracing 开启时 `take(id) → decode_w3c_traceparent → Span::root parent`，miss/非法回退 `SpanContext::random()`（现状）
- 单测 3 个（roundtrip+W3C 解码、注入 id、无头忽略），`cargo test --release -p openinfer-vllm-frontend --lib` 30/30 通过；clippy 干净（修了一个单模式 match→if let）
- 结果：成功

### Step 2: 环境确认
- docker OK；Tempo/Grafana 已通过 `deploy/tracing/docker-compose.yml up -d` 启动（4317/3000）
- GPU：RTX 5070 Ti 16GB；模型：`/data/models/Qwen3-4B`（repo 内无 models/ 目录）
- router 与 openinfer 的 release 构建并行后台进行

### Step 3: 起栈 + 首次验证（失败 → 修复 → 通过）
- GPU 被 pegainfer-2 的旧 openinfer（PID 725807，13.5GB）占满 → 用户确认后 kill，服务正常起（`OPENINFER_TRACE_OTLP_ENDPOINT=http://127.0.0.1:4317`，:8000）
- router 起在 :8090：`vllm-router --worker-urls http://127.0.0.1:8000 --policy round_robin --enable-trace --otlp-traces-endpoint 127.0.0.1:4317`（构建零改动，libzmq 系统在）
- 首次验证：router 两 span 成链，但 openinfer 另开随机 trace——**接合失败**
- 定位：直连 openinfer 带 `traceparent`+`X-Request-Id: dbg12345` 仍失败；查 openinfer 侧 trace 的 `request_id=cmpl-dbg12345-2b7dfbb4` → vllm-server 把 X-Request-Id 加上 **`cmpl-`/`chatcmpl-` 前缀**后才作为 `external_req_id`（llm/request.rs: prepare() 直接沿用 route 前缀 id），stash 键（裸 header 值）查不中
- 修复：`TraceContextStash::take_for_external_req_id`——先精确查，再剥 `chatcmpl-`/`cmpl-` 前缀查；附单测 `lookup_tolerates_vllm_api_prefixes`；31/31 通过，clippy 干净
- 复验（trace id `aaaa1111…`）：**单条 trace 全链路通过**——
  client(5555eeee) → router `http_request`(334.79ms) → router `http_client_request` → openinfer `request`(319.89ms) → `queue`(0.04ms) → `prefill`(72.06ms) → `decode`(247.67ms)，parent 逐级正确
- 结果：成功

### Unexpected
- router 的 `http_client_request` span duration 显示 ~5.3s（远超 server span 334ms），疑似其非流式路径 span 结束时机/泄漏问题——router 侧仪表缺陷，不影响链路验证，考虑向上游反馈
- Tempo `/api/traces` 的 span id 是 base64，且同 trace id 的多次请求 span 会合块——调试时每次换新 trace id

### Step 4: 负载 + 双端点验证（通过）
- 16 req / conc=8 / distinct prompts 经 router：client p50 756ms（含 Python urllib 每请求新建连接开销）；Tempo 16/16 条完整链路
  - 阶段 p50：queue 0.02ms / prefill 25.1ms（max 101.7，合批）/ decode 359.9ms（32 tok，bs≈8，~11ms/tok @5070 Ti）/ request 466.3ms
  - 抽样单 trace：router server span 388.50ms vs openinfer request 386.14ms、起点差 1.02ms → **router 自身开销 ~1ms 量级**；bridge 开 span 前另有 ~6.8ms 在 vllm-server HTTP/tokenize（无 span，呈间隙）
- `/v1/chat/completions`（chatcmpl- 前缀）同链路验证通过：client span → router 两 span → openinfer 四 span，parent 逐级正确
- 无 client traceparent 场景：router `http_request` 成为 trace root，openinfer 正常挂载（负载测试即此场景）
- 结果：成功

### Step 4.5: Tempo 查询坑
- `/api/traces/<id>` 可能只返回已到块的部分 span（ingester 5s 级刷新），重取即全——不是丢 span

### Step 5: 上游 PR + tracking issue
- 上游 PR：https://github.com/vllm-project/vllm/pull/50370 「[Rust Frontend] Propagate W3C trace headers to engine-core requests」（xiaguan fork，DCO signed，off upstream/main e5f48dfda）
  - 事前查重发现 **#44567**（同目标）被 maintainer 以「新增协议面」为由拒过；本 PR 刻意走最小透传：无握手、无 gating、纯提取填 `trace_headers`，PR 正文写明该立场
  - 改动 10 文件：`resolve_request_context` 提取 traceparent/tracestate → `ResolvedRequestContext.trace_headers` → completions/chat/generate convert → 经 vllm-text/vllm-chat  additive 字段透传进 llm `GenerateRequest`（llm/engine-core-client 零改动）；grpc/tokenize 仅编译必需的 `None` 补齐
  - 门禁：fmt/clippy(-D warnings) 干净；nextest vllm-text+chat 321 过、vllm-server 328 过（含 12 个新测试）
- tracking issue：https://github.com/openinfer-project/openinfer/issues/790 —— 记录 MVP 现状、上游 PR、迁移三步（bump pin → bridge 改读 `trace_headers["traceparent"]` → 删 trace_context.rs 及 lib.rs 接线）
- Grafana 访问：本机 3000 与用户 mac 本地 Grafana 冲突，容器改绑 4000（repo compose 文件未动）；用户浏览器曾落到别的 Grafana 12 实例（有 Sign in/Bookmarks），以 `curl localhost:4000/api/health` 返回 11.3.0 判定转发正确
- 匿名 Viewer 在 Grafana 11.3 下被 `datasources:explore` 拒绝（日志 Access denied，UI 表现为 Explore 无结果）→ 运行中容器改 `GF_AUTH_ANONYMOUS_ORG_ROLE=Editor` 后恢复；repo compose 文件仍写 Viewer，待用户确认后同步修正
- PR 跟进：仅有 bot 占位评论，无 maintainer 响应；按用户判断（maintainer 或想自己做）在 PR 下发 direction/timeline 询问 comment（#issuecomment-5126068994），表明可改造/可关闭并接受上游自有实现

## Debrief

- **Outcome**: client → vllm-router → openinfer → prefill/decode 全链路单 trace 打通并在 Tempo 验证（两个端点、有无 client traceparent、16 并发负载均通过）；router 自身开销 ~1ms 量级。本地 MVP（middleware + stash + bridge parent 接线）落地并测试；上游 PR vllm-project/vllm#50370 已开；迁移由 openinfer#790 跟踪。
- **Pitfalls encountered**:
  - 接合首败根因：vllm-server 给 `X-Request-Id` 加 `cmpl-`/`chatcmpl-` 前缀才作 `external_req_id`——协议事实只能靠实测 trace 属性（`request_id=cmpl-dbg12345-…`）定位，读代码时漏了 route 层前缀这一步
  - 直连对照实验（绕开 router 打 openinfer）是二分定位的关键：先把锅缩到 openinfer 侧，再用 span 属性锁定前缀问题
  - 上游曾有同目标 PR（#44567）因「新增协议面/gating」被拒——给上游提 PR 前必须先查被拒历史，最小透传才是可接受的形状
  - Tempo `/api/traces` 部分返回（ingester 刷新窗口）和同 trace id 合块，调试时各坑一次
- **Lessons learned**:
  - 验证 trace 链路用确定性 traceparent（自带 trace id）+ Tempo HTTP API 断言，比肉眼翻 UI 快且可复现
  - 「git 依赖缺功能」不一定 fork：`serve_with_router_extension` 这类框架钩子 + middleware 能补；但 workaround 要配 tracking issue 和上游 PR，否则会烂在树里
- **Follow-ups**:
  - openinfer#790：上游 PR 合入并 bump pin 后删 middleware
  - vllm-router `http_client_request` span 偶发 ~5s 虚长（非流式路径），疑似 router 仪表 bug，可单独给 vllm-project/router 提 issue
  - vllm-server 自身 OTel 仪表化（HTTP/tokenize 段目前是 trace 里的空白间隙），上游已有相关 PR 系列（#39438、#39905），跟进即可
  - `docs/roadmap/roadmap-2026-h2.md` — "observability wiring" 在 H2 计划内，本任务属其一环
  - `docs/subsystems/router/kv-aware-routing.md` — 此前 Dynamo router 多轮路由实验（不同 router，但 e2e 测量方法可借鉴）
  - 无 tracing 相关的 past task doc

- **Plan**:
  1. openinfer 侧加 W3C trace context 提取（本 repo 唯一代码改动）：
     - 新模块 `openinfer-vllm-frontend/src/trace_context.rs`：axum middleware 读 `traceparent` 头；有则取 `X-Request-Id`（缺则生成短 id 并注入请求头，vllm-server 会把它当 `external_req_id`），存入有界共享 map（`external_req_id → traceparent`，带 TTL/容量淘汰）
     - 在 `serve_model_on_host_with_router_extension` 里对所有 serving 变体统一挂上该 layer
     - `bridge.rs` add_request：tracing 开启时按 `external_req_id` 查 map（pop），`decode_w3c_traceparent` 成功则以其为 parent 建 `request` 根 span；否则回退 `SpanContext::random()`（现状行为）
     - 单测：middleware 注入/无头路径 + bridge 用 stashed context 建 span（参考 phase_trace.rs 的 TestReporter 模式）
  2. 起本地栈：`docker compose -f deploy/tracing/docker-compose.yml up -d`（Tempo 4317 / Grafana 3000）
  3. 起 openinfer：`OPENINFER_TRACE_OTLP_ENDPOINT=http://127.0.0.1:4317 cargo run --release -- --model-path models/Qwen3-4B --port 8000`（先确认模型权重与 GPU 在机）
  4. 构建并起 router（/data/code/workspace-rustllm/router）：`cargo build --release`，`vllm-router --worker-urls http://127.0.0.1:8000 --port 8090 --enable-trace --otlp-traces-endpoint <4317>`（flag 格式执行时核对）
  5. 验证单链路：经 router 发 `/v1/completions`；用 Tempo HTTP API（`:3200/api/search` + `/api/traces/<id>`）断言**一条 trace** 内含 router 的 `http_request` → `http_client_request` → openinfer 的 `request` → `queue`/`prefill`/`decode`，parent 关系正确
  6. e2e 效果观察：小并发负载（vllm-bench 或多轮 curl），看 router 转发开销、queue wait、prefill/decode 分解；client 自带 traceparent 的场景也验一次（router 提取后整链应挂在 client span 下）
  7. 收尾：更新 deploy/tracing 注释、docs/index.md，写 Debrief

- **决策（2026-07-30 review）**: 双轨——本地 MVP（middleware 绕过）先行验证链路；同时给 vllm-project/vllm 提上游 PR（Rust server 提取 traceparent 填 `trace_headers`，与 Python parity）；pegainfer 本 repo 开 tracking issue 记录「middleware → upstream」迁移。上游合入并 bump pin 后删 middleware，bridge 改从 `EngineCoreRequest.trace_headers` 读（bridge 改动两轨共享，不浪费）。

- **Risks / open questions**:
  - 关联靠 `X-Request-Id → external_req_id`：middleware 注入 id 的时机必须在 vllm-server 读头之前（axum layer 顺序），需实测验证
  - router 的 `--otlp-traces-endpoint` 格式（host:port vs URL）与其 `service.name`，执行时核对
  - 单机单 worker 时 router 的 policy 无所谓，但 router 默认重试/熔断可能在压测时产生额外 span，注意甄别
  - 模型权重 `models/Qwen3-4B` 与 GPU 可用性未确认
