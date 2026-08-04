# Resolver 所有权与 native tail 定形：#830 的教训与后继设计

> **TL;DR:** #830（glm52 迁移 pegainfer-kv-store）八轮评审 18 条 finding 的复盘结论：**plain 路径符合 design.md 的双分配器模型，偏航只在 native P/D 路径**——它把权威级分配（RequestKv 生命周期、keyed tail 装载）放进了 resolver 任务，破坏了"resolve 分配零义务、可让步"的前提；评审补出的 HeadroomLedger 恰是 design.md 明文警告的"第二本账"。后继 PR 的设计：① native full 页改走 radix-first（与 plain 同路，零义务）；② **keyed tail 用 pad-to-boundary 消灭**（save 本就按块粒度运整页 slab，padding 零字节代价，换来 radix 身份 + 失败前置到 admission 之前）；③ 仲裁按 design.md 未决预定形态归位（池内 async reserve + admission 侧 prefetched 抵扣），台账族与 keyed API 族整体删除。#830 冻结为记录，其 20+ 契约测试作为后继的行为验收标准。
>
> Status: 设计评审中。前置阅读：`design.md`（尤其「请求管线：线性所有权链」与「池的并发事实」）。

## 一、#830 十八条 finding 的归因

| 类 | 代表 finding | 根因 |
| --- | --- | --- |
| 容量竞态(~9 条) | native 互饿死锁、headroom claim 生命周期、probe pin 绕账、plain 记账非原子、tail 一次性分配 | native resolver 持有**不可让步的权威级分配**,与调度器在池上竞争 |
| 异步所有权(~5 条) | keyed-tail 无界等待、lease 泄漏 ×2、取消不穿透 | `timeout_at` 丢观察者不丢工作;其中 lease-settle/DMA-parking 是 design.md 230 行本就要求的「guard 陪 op 走到 settle」,#830 初版漏实现 |
| 删掉的防御 | 匿名 key 序号、resolver 负载可见性、tail 等待语义 | 旧串行状态机的行为被当实现细节丢弃(见 conventions 待立项:迁移防御清单) |
| 真边界 | handoff 静默降级、teardown arena UAF | 与所有权正交,修复直接继承 |

**关键对照**:design.md 231 行早已定义了权限模型——admission 的 lifetime 预留是**权威**(honor-or-reject),resolve 的 reservation 是**机会主义**(恢复的是 radix 共享缓存块,零义务、拿不到就让步);仲裁被有意推迟,未决 252 行连回归形态都写明,并警告「外挂信号量镜像可用量会造第二本账」。#830 的 native 路径在 resolver 里 `pool.new_request` + schedule 生命周期 + tail 装进私页——私产不可让步,模型前提崩塌;评审八轮随后补出的 debt 台账正是被警告的第二本账。**补丁的形状(影子记账、锁内 verify-and-book、cancel 穿针)就是架构报警**。

## 二、后继设计

### 2.1 plain 路径:照抄 #830 现状(它是符合设计的)

resolve task 线性持有 req:probe hold(RAII,零分配)→ guarded tier query(lease 必达 settle)→ 机会主义 reservation(可让步)→ load(已提交 DMA 不可取消,reservation 随 detached task)→ commit 入 radix → 投递 `(req, KvPrefix{hold})` 给 scheduler 收件箱。权威分配只发生在 admission。

继承自 #830 的修复:`LeaseGuard`/`spawn_guarded_query`(迟到命中的 lease 自动释放)、load 的 detach 语义与 `flush_loads` 排水、`PrefixProbe::truncate_held`(hold 与 credit 钉页对齐)、resolver 负载可见性思想(waiting = pending + inflight)、handoff 能力不匹配显式拒绝、teardown 排水超时泄漏 worker。

### 2.2 native P/D:radix-first + pad-to-boundary

**full 页**:D 的恢复就是一次 `wait_for_full_hit` 的 `resolve_prefix`——零义务共享块,admission 断言命中长度并做全部权威分配。resolver 里不再出现 RequestKv。

**tail(committed_len % 64 的半页)**:三不状态(不可封存/无 radix 身份/必须落私页)曾催生 keyed 旁路 API。定形:**P 侧 pad 到页边界**。

- P:半页用词表外保留 pad id 补满 token 链 → 页封存,正常 Handoff-class save。**零字节代价**:save 按块粒度搬运,半页的 slab 本来就整页在运;padding 只是给垃圾行一个名字,换取 radix 身份。pad id 混入既有 `native_mtp_cache_salt`,杜绝与真实续写的 hash 碰撞。
- D resolver:边界页随 full 页一起按 padded-hash 恢复进 radix(仍是零义务缓存块)。
- D admission(native 臂):full 块自然 match;边界块按 padded-hash 匹配(~20 行,committed_len 在 handoff 元数据里)+ **copy-on-restore** 拷进请求私页(对齐组全家一起拷:mla+idxk+MTP L78 镜像,~3.3MB D2D 微秒级)。decode 从 committed_len 处 append 自然覆写 pad 位;attention 被 seq_len 界住,永远读不到 pad 行。
- 失败语义:padded 页缺失 = 命中短缺 = **admission 拒绝,发生在占 slot 之前**,router 兜底。(对比曾考虑的"admission 后再取 tail":取回失败发生在 slot 已占之后,15s deadline × slot 数是存储抖动即冻结整 rank 的 DoS 面;及"D 重算尾巴":每请求 ≤63 行 context 挤进 decode 步流,c64 突发时 ~2000 行,违背 PD 分离的 decode 纯度——均否决。)
- turn-2 prefix 复用终止在最后一个真 full 页,与现状一致,无损失。

### 2.3 仲裁归位(design.md 未决 252 行的预定形态)

- 池内 `async reserve_blocks(n)`:waiters 挂在唯一真相旁,TOCTOU 在分配器内部收口;
- admission 自管预算:resolve 已命中块按 `prefetched_blocks` 抵扣(qwen3 先例),不设独立水位 API,**不建第二本账**。

## 三、kv-store 变更清单(相对 #830 尖端 d303852)

| 处置 | 内容 |
| --- | --- |
| **保留** | `LeaseGuard`/`GuardedQuery`/`spawn_guarded_query`;load 的 detach+`flush_loads`;`truncate_held`;mock tier 测试基建;`seal`/`retire`/`flush_saves` 主干 |
| **删除** | `HeadroomLedger` 族(`assume_active_headroom`/`with_headroom_sync`/`settle_headroom`/`schedule_prefill_resolver` 门)、`reserve_headroom`、store 内 `CancelProbe` 穿针、**keyed 族**(`seal_keyed`/`resolve_keyed_block`/`KeyedFetchError`/`KeyedLoadParking`) |
| **重塑** | 池内 async reserve(waiters);admission 侧 prefetched 抵扣;P 侧 pad-and-seal(落在 glm52 P 逻辑,store 无新 API) |

破坏性变更成本:零——`pegainfer-kv-store` 当前唯一消费者是 glm52。

## 四、验收与迁移

- **行为验收 = #830 的契约测试全集**(互饿不可形成、resolver 不搁浅 active、取消及时释放、lease settle、parking、handoff 拒绝、teardown 泄漏……),机制换、断言不换;keyed 族测试随 API 删除,由 padded 路径的新契约测试接棒(padded 页缺失拒绝、copy-on-restore 对齐组完整性、pad 行不可见)。
- 后继实现 PR 基于 main 重做(不基于 #830 分支);#830 冻结为设计论证记录。
- 真机验收:1P1D+router 复刻 #830 迁移时的验收矩阵(GSM8K n200、multi-turn c16 头对头、240/240 零失败)+ c64 全量 trace 回放(#833 战役的 harness 现成)。

## 待讨论(后续逐块)

- scheduler 侧:收件箱/admission 在新形态下的结构(native 臂的 padded-match 放哪、bypass 谓词简化后的样子);
- P 侧 pad-and-seal 的落点(prefill_tp 封存路径)与 handoff 信封变更(committed_len 语义不变,tail_len 字段退役);
- 迁移防御清单立项(`docs/conventions/`):重写类 PR 必须逐条注明旧防御结构的接班人。
