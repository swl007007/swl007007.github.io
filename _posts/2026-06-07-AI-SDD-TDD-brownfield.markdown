---
layout:     post
title:      "Vibe Coding 的三驾马车：一次 Brownfield + SDD + TDD 的真实开发复盘"
date:       2026-06-07 12:00:00
author:     "William Shi"
header-img: "img/post-bg-universe.jpg"
catalog: true
tags:
    - Review
---


# Vibe Coding 的三驾马车：一次 Brownfield + SDD + TDD 的真实开发复盘

> 以 GeoDT branch-tree interpretability diagnostic 为例，说明 AI coding agent 在遗留 ML 项目里如何从“能生成代码”走向“能生成可信、可审计、可复现的结果”。

## 1. 结论先行：AI 写代码不难，难的是在旧系统里写对

<img src="https://github.com/swl007007/swl007007.github.io/blob/308b5095d28c2a7f77bdb0564bd13a14caeb8402/img/diagram.png?raw=true" alt="branch tree comparison" width="600"/>


过去一段时间里，我越来越确定：Vibe Coding 真正困难的部分，不是让 AI 写出代码，而是让 AI 在一个已经存在、已经有历史包袱、已经有隐含契约的项目里写对代码。

在新项目中，agent 可以从零搭积木，错了大不了拆掉重来。但在 Brownfield ML 项目中，很多“看起来合理”的改动都会踩到旧 pipeline、历史 artifact、研究假设或评估口径。GeoDT branch-tree interpretability 这次开发让我更清楚地看到，Workflow Pack 1.0 的价值不在于多跑几个 slash command，而在于把三件事绑成一个闭环：

- **Brownfield**：先定义什么默认不能动。
- **SDD（Spec-Driven Development）**：再定义当前需求到底授权动什么。
- **TDD（Test-Driven Development）**：最后定义在动手之前必须先证明什么失败。

三者合在一起，才可能把 Vibe Coding 从“凭感觉让模型往前冲”，变成一种可审计、可回滚、可复现的工程协作方式。

单独使用其中任何一个都不够。只讲 Brownfield，容易停在“别破坏旧系统”的保守姿态，却说不清当前功能到底要做什么；只讲 SDD，容易生成漂亮的 spec、plan 和 tasks，却无法保证实现阶段真的按风险顺序推进；只讲 TDD，则可能在错误 scope 上严谨地写测试，把一个本不该做的改动验证得很充分。

GeoDT 的经验说明：在 legacy research / ML 项目中，AI 生成 artifact 的门槛不能只是“能跑通”，而必须是“能证明它从哪里来、为什么没污染旧系统、如何复现”。

## 2. 案例背景：为什么“画两棵树”不是一个小需求

GeoDT 可以粗略理解为一种“先按地理空间分支，再在每个分支上训练本地决策树”的模型流程。代码读取 FEWSNET 和相关特征数据，按时间窗口构造训练/测试样本，再用已有空间 partitioning 逻辑把 admin/group 分配到不同 branch。每个 terminal branch 有自己的 local `DecisionTreeClassifier` checkpoint；而 root/global tree 和 `dt_rules_*.csv` 主要代表全局或根节点规则导出，并不等于最终被空间分支 dispatch 后使用的本地树。

这一区别是整个案例的风险核心。用户最初想看的并不是“随便一棵决策树”，而是“某个具体 GeoDT run 中，两个相邻或不同 terminal branches 的 local rules 是否存在可解释差异”。因此，这次 interpretability diagnostic 的核心不是画图能力，而是确认以下证据是否来自同一套可信来源：

- 哪个 monthly archive 被选中；
- 哪些 terminal branches 被比较；
- 哪些 branch-specific checkpoints 被加载；
- feature names 是否和 checkpoint 兼容；
- dispatch assignments 是否属于同一个 run；
- 输出图和 metadata 是否能被复现。

表面上，这个需求只是“画两棵树”。实际它牵涉 archived model artifacts、branch-specific checkpoints、root/global `dt_rules` 的语义、feature-name 对齐、archive discovery、no-overwrite、audit-only、reproduction mode 和 metadata provenance。

它足够小，可以用一个 standalone diagnostic 交付，不需要重构主训练 pipeline；但它也足够危险：只要 agent 把 root/global `dt_rules` 当成 branch-local explanation，或者把某个不完整 archive 当成完整证据源，就会产出一张“看起来很专业、实际在误导”的图。正因为这个功能夹在“可视化很小”和“解释风险很大”之间，它特别适合检验 Workflow Pack 1.0 是否真的能在 legacy ML repo 里形成安全带。

本文不把 GeoDT 当成一次普通可视化功能复盘，而把它当作一个具体样本：在 AI coding agent 参与的 legacy ML 开发中，Brownfield、SDD 和 TDD 如何分别约束边界、授权需求、固定验证顺序，并共同减少“误导性成功”。

## 3. Brownfield：先告诉 agent 默认不能动什么

<img src="https://github.com/swl007007/swl007007.github.io/blob/308b5095d28c2a7f77bdb0564bd13a14caeb8402/img/counterexample.png?raw=true" alt="counterexample" width="900"/>

### 3.1 遗留 ML 项目的风险不是缺代码，而是可误用的历史产物太多

GeoDT 项目里已经存在训练入口、batch launcher、checkpoint、partition artifacts、`dt_rules` CSV、evaluation outputs、deliverables 和多轮历史 specs。对 agent 来说，这些东西都很诱人：文件名像、结构像、内容也像，似乎随便拿来就能完成“可解释性可视化”。

但 legacy ML repo 的风险恰恰在这里：历史产物不是天然授权。能被读取，不代表能被重新解释；能被复用，不代表语义仍然适配当前需求。GeoDT 这次如果没有先做 Brownfield scan 和 change-surface 约束，最容易发生的错误不是代码写不出来，而是 agent 用了一个“存在但不该用”的 artifact，然后把错误来源包装成合理输出。

### 3.2 GeoDT 的关键 Brownfield 边界

这次 spec 明确要求保留现有训练、prediction dispatch、branch training、checkpoint naming、root/global `dt_rules_*.csv` export、lag schedule、batch launcher 和 production deliverables。新功能只能增加一个 standalone exploratory diagnostic，不能修改标准 GeoDT pipeline，也不能把诊断需求反向写进训练逻辑。

这个边界很关键，因为用户的目标只是“观察 branch tree 的 local rules 差异”，不是改变 GeoDT 如何训练、如何预测、如何导出规则。Brownfield 在这里起到的是“先画禁区”的作用：即使某个改动能让可视化更方便，只要它会碰 training、dispatch、export 或 deliverables，就默认不被授权。

### 3.3 Brownfield 把“看起来合理”的捷径变成非法操作

最危险的捷径，是把 root/global `dt_rules_*.csv` 当成 branch-specific local tree explanation。这个捷径非常有迷惑性，因为 `dt_rules` 文件名里有 “DT rules”，内容也像规则表；如果只是为了尽快产出图，很容易说服自己“先用这个”。

但 Brownfield 边界要求现有 `dt_rules` 行为保持原义，不允许为了可视化方便重新解释旧 artifact。最终 diagnostic 必须从 branch-specific local DecisionTree checkpoints 出发，并在 metadata 里明确记录 root/global exclusion。这样，一个潜在的语义偷换在实现前就被判定为非法操作，而不是等 reviewer 看到图之后再猜它到底画的是什么。

### 3.4 Brownfield 也保护输出侧

GeoDT spec 不只保护源代码，还保护生成物。已有 checkpoints、`space_partitions/`、`dt_rules/`、`other_outputs/`、`deliverables/` 都是 read-only inputs。诊断输出必须写到 diagnostics output，且默认 no-overwrite；脚本也需要在计划写入 PNG 和 metadata 前做 preflight，已有文件应触发 structured failure summary，除非显式传入 `--overwrite`。

这看似只是文件安全，但在研究型 ML 项目里非常重要。Output artifact 经常会被论文、报告、deliverable 或后续分析引用。一旦 agent 覆盖了旧图、旧 metadata 或 production-style artifact，哪怕代码逻辑是对的，也会破坏证据链。

### 3.5 没有 Brownfield，最可能发生什么

如果没有 Brownfield，agent 很可能直接修改 `app/main_model_DT.py` 或 `src/model/GeoRF_DT.py`，也可能顺手调整 exporter、batch launcher 或 artifact layout，让主 pipeline 为一个 exploratory figure 让路。更隐蔽的风险是，它可能并不大改代码，却把 `dt_rules` 作为 branch rule source，把不完整 archive 当作可视化来源，或者把 diagnostic outputs 写进已有生产目录。

最终得到的图也许能生成，甚至很美观，但 reviewer 无法判断它是否改变了旧 pipeline、污染了旧 artifact，或者只是把 global tree 包装成 local explanation。Brownfield 的价值，就是在实现前先让这些“更快但更危险”的路线全部失效。

## 4. SDD：把愿望变成授权链

### 4.1 SDD 的核心不是写文档，而是建立 source-of-truth

GeoDT 的授权链不是某一条 prompt，也不是 memory 里的一段总结，而是 resolved analyze findings / human notes、`tasks.md`、`plan.md`、`spec.md`、evidence pack、supplementary docs，最后才是 `implementation-micro-plan.md`。

在这条链路里，micro-plan 只能执行上游决定，不能反向创造 scope；migrated artifacts 和 memory summaries 只能作为 reference-only 约束，不能授权新行为。这一点在 GeoDT 里尤其重要，因为历史上下文很多，agent 很容易从旧 spec、旧 evidence 或记忆中抽出一句“好像相关”的话扩大范围。

SDD 要解决的不是“多写文档”的问题，而是让每个实现动作都能追溯到当前 active feature 的明确授权，而不是追溯到 agent 的联想。

### 4.2 Spec 把“我想看”变成可验收 contract

用户原始目标是生成一张 branch-specific local tree comparison figure，背后真正想确认的是：某个具体 GeoDT 模型里，相邻或不同 branch 的 DT rules 是否存在可解释差异。

GeoDT spec 把这个愿望拆成三条 user stories：

1. **Figure generation**：从 branch-local checkpoints 生成比较图。
2. **Audit-only branch selection**：共享同一套 selection pipeline，但不渲染图、不产生副作用。
3. **Metadata-driven reproduction**：能从 metadata 重新验证 checkpoint 和 top-k feature signatures。

这样拆分后，需求就不再是“画一张图”这么模糊，而是每条 story 都有独立测试和可衡量的验收条件。Spec 在这里把“我想看”转换成“什么条件下这张图才算可信”。

### 4.3 Plan 把 feature 放回 legacy architecture

GeoDT plan 明确了 change surface：新增 `scripts/plot_geodt_branch_tree_comparison.py`，新增 focused tests，写 diagnostics outputs；同时列出 read-only 和 forbidden modules。

这个 plan 的意义不是把实现步骤写得更漂亮，而是把功能重新放回 legacy architecture 里。它只能作为旁路 diagnostic 存在，不能把训练、prediction dispatch、branch training、lag schedule 或 production prediction scripts 拉进修改范围。对 agent 来说，这相当于挡住了“为了方便加载模型就顺手改主流程”的冲动。

因此，Plan 真正限定的问题不是“如何重新设计 GeoDT”，而是：如何在不改变旧系统的前提下，完成 archive discovery、checkpoint compatibility、feature-name validation、selection metadata 和 output safety。

### 4.4 Tasks 把依赖关系变成 gate

GeoDT tasks 的关键不是任务数量，而是 T018A 这样的 foundation gate：T006–T018 必须先完成 characterization、fixtures 和 Brownfield regression gates，任何会触碰 `scripts/plot_geodt_branch_tree_comparison.py` 行为变化的任务都不能提前开始。

这个 gate 来自真实开发中的一次重要收敛：当 runtime/archive artifacts、feature-name source、selection logic 和 validation target 还没有足够 evidence 时，不能让 agent 先写脚本再补解释。Tasks 在这里不是待办清单，而是门禁系统。它规定先证明 archive 长什么样、哪些 artifact 构成 minimum complete set、哪些 legacy behavior 必须保护，再进入 GREEN implementation。

### 4.5 Review loop 应该修 source-of-truth，而不只是修代码

GeoDT 过程中出现的 HIGH/CRITICAL 问题，多数不是普通 bug，而是 source-of-truth、dependency、readiness 或 duplicate decision-source 问题。例如，selection pipeline 如果同时服务 figure generation 和 audit-only，却被安排在较晚任务里，就会造成 US1 rendering 依赖还没实现的行为；micro-plan 如果没有明确 T001–T018A 与 T024+ 的边界，agent 就可能过早进入行为实现。

这类问题不能只在代码里打补丁。正确修复应该回到 requirement、task dependency、micro-plan gate 或 validation contract，阻止同类错误再次接近实现阶段。Review loop 的价值就在于它不只问“这段代码有没有 bug”，还问“为什么流程允许这个 bug 接近实现阶段”。

### 4.6 没有 SDD，最可能发生什么

如果没有 SDD，agent 可能跳过 audit-only，直接画图；可能没有 reproduction mode；可能只处理 preferred archive，而不定义 fallback；可能缺少 metadata fields；可能让 figure mode 和 audit mode 各自实现 selection scoring；也可能在遇到不完整 archive 时悄悄降级到 root/global `dt_rules`。

每个决定单看都像工程取舍，合起来就是不可审计的解释性产物。SDD 的作用，是把这些取舍提前暴露出来，让它们必须经过 spec、plan、tasks 和 evidence 的交叉验证，而不是在实现后被包装成“合理默认值”。

## 5. TDD：先证明风险存在，再实现功能

### 5.1 这不是普通 UI 图，而是 data/artifact behavior

GeoDT branch-tree comparison 的正确性不取决于 PNG 是否生成，而取决于它是否满足一组不可见条件：

- 是否来自 branch-specific checkpoints；
- 是否绑定 final dispatch assignments；
- 是否使用兼容的 feature names；
- 是否按 deterministic Jaccard score 选择 branch pair；
- 是否拒绝 unreadable 或 misleading pair；
- 是否记录足够的 metadata 以供复现。

同样一张图，如果 checkpoint 来源错了、feature names 对不上、branch assignment 不是同一 run 的，视觉上可能完全看不出来。TDD 在这里的价值，就是先把这些不可见的正确性条件变成测试可观察的失败，而不是等图生成后再凭肉眼判断。

### 5.2 RED 不能写成一句“add tests”

Workflow Pack 的 TDD policy 要求 RED step 指定 exact test target、command 和 expected failure reason。GeoDT micro-plan 对每个行为任务都要求先写 contract、fixture、artifact、metadata 或 smoke test，再实现最小代码。

这里的关键是：RED 不是一句“加测试”，而是要写清楚当前系统为什么还做不到。例如：

- root/global-only archive 应该被拒绝；
- existing output path 应该触发 no-overwrite failure；
- feature-count mismatch 应该失败；
- audit-only 不应该生成 PNG；
- reproduction metadata drift 应该被发现。

只有 RED 失败原因足够具体，GREEN 才不会变成“大改一把直到能跑”。

### 5.3 测试不是锦上添花，而是需求本体

GeoDT 的 focused tests 覆盖 archive preflight、checkpoint classification、feature-name validation、figure rendering、metadata output、no-overwrite failure summary 和 shared selection-pipeline behavior。它们不是为已有实现补覆盖率，而是在实现之前定义什么叫可信 diagnostic。

换句话说，测试本身就是 GeoDT spec 的可执行版本。没有这些测试，PNG 只能证明 matplotlib 能画图，不能证明解释证据可信。

### 5.4 TDD 阻止“先实现再补验证”的幻觉

如果先写脚本，再补测试，agent 往往会让测试贴合已有实现，甚至把实现中的错误假设固化成 expected behavior。GeoDT 的 micro-plan 反过来要求行为任务先复用已存在的 RED gates，再做最小 GREEN change。

这个顺序在 branch-pair selection、metadata writer、audit-only wrapper 和 reproduction check 中尤其重要。它逼迫每一步都回答：我现在是在让哪个已知失败变绿？而不是凭感觉扩展功能。

### 5.5 Data/artifact behavior 也需要 test-first

很多人会把 TDD 和传统业务逻辑绑定在一起，好像只有 API、函数或 UI 才适合 test-first。但 GeoDT 证明，archive resolver、feature-name selector、metadata writer、failure summary、reproduction mismatch 都是 behavior-changing tasks，也都需要 test-first。

比如 metadata 不只是写一个 JSON 文件。它要承载 selected archive、artifact-provider path、branch assignment source、feature-name source、checkpoint paths、root/global exclusion、K、score、tie-break、readability、output paths 和 failure summary。只要这些字段缺失或语义不稳定，后续 reviewer 就无法复现选择过程。Data/artifact behavior 越依赖文件系统和历史输出，越需要先用 fixture 和 schema 固定住。

### 5.6 没有 TDD，最可能发生什么

没有 TDD，代码可能能跑通 happy path：给一个完整 archive，生成一张 PNG，再输出一个简单 JSON。但它没有证明自己拒绝 root/global fallback、拒绝 unbounded search、拒绝 overwrite、拒绝 feature-count mismatch、拒绝 reproduction artifact drift，也没有证明 audit-only 和 figure-generation 使用同一个 selection pipeline。

最终 review 只能看到“有图”，看不到“为什么这张图可信”。在解释性 ML 任务里，这种成功最危险，因为它不像 crash 那样明显失败，而是以一种看似完成的形式把错误带进分析结果。

## 6. 三者如何形成闭环

Brownfield、SDD 和 TDD 不是三个并列口号，而是在 GeoDT spec 中分别回答了三个问题：

| 问题 | 由谁回答 | 在 GeoDT 中的表现 |
| --- | --- | --- |
| 什么默认不能动？ | Brownfield | 训练、dispatch、branch training、lag schedule、batch launcher、production prediction、root/global `dt_rules` export 和历史 artifacts 都是 protected surface。 |
| 当前需求授权动什么？ | SDD | 只允许新增 standalone exploratory CLI、focused tests、metadata/audit/reproduction/failure outputs，以及必要的小型 helper。 |
| 什么时候可以动？ | TDD | 在 RED gate、fixtures、preflight 和 validation contract 就绪之前，不进入行为实现。 |

这个闭环的重要性在于：即使某个文件位于 allowed surface 内，也不代表 agent 可以立即实现。新增 diagnostic script 是 allowed surface，但在 archive completeness、feature-name source、root/global exclusion、no-overwrite 和 selection pipeline 还没有 RED gate 前，过早写脚本仍然是不安全的。

`sdd-superpower-micro-plan` 在这里扮演粘合层。它不是替代 Speckit artifacts，而是把 task graph 降解成可执行波次，并把 Brownfield、source-of-truth order、TDD classification、RED command、expected failure、GREEN scope 和 validation command 放到每个 task 前面。

GeoDT 的 micro-plan 后来专门补强了 readiness gates：T001–T018A 可以安全推进 characterization 和 RED gates；T024+ 行为变化必须等待前置 evidence；T032/T033/T035 又必须等待 T041–T043 shared selection pipeline 完成。这样，Brownfield 的“不能碰什么”、SDD 的“允许做什么”、TDD 的“先证明什么失败”不再停留在抽象原则，而是变成每个任务开始前都要检查的执行条件。

Implementation constraint prompt 则把这些规则带进执行现场：读 micro-plan；以 Speckit artifacts 为 source of truth；Brownfield Mode ON；不碰 read-only paths；T018A 前不创建或编辑 diagnostic script；selection pipeline 实现一次；行为变化任务必须 strict TDD；validation 未执行不能标记完成。

实际复盘中，很多风险不是因为 spec 没写，而是因为执行上下文太长、任务太碎，agent 容易只盯着下一步。constraint prompt 就像现场安全卡，把关键规则压缩成每次动手前都要重新面对的清单。

## 7. GeoDT 案例中的五类失败模式

### 7.1 Source-of-truth drift

当 `.specify/feature.json`、migrated artifacts、memory summary、tasks.md 和 micro-plan 指向不同 scope 时，agent 会在错误 feature 上认真工作。Reference-only artifacts 很有价值，因为它们记录了过去的分析、决策和风险；但它们也很危险，因为一旦被当成当前授权，就会让 scope 漂移。

Workflow Pack 的修复方式，是声明 active feature、source-of-truth order，并把 migrated artifacts 标成 reference-only。旧证据可以提醒 regression risk，但不能替当前 spec 做决定。这样，agent 就不能再用“记忆里好像说过”作为改行为的理由。

### 7.2 Implementation readiness 过早放行

GeoDT 不是 T001 完成后就可以写脚本。T018A 把 characterization、fixture、regression gate 和 evidence-dependent assumptions 全部作为行为实现前置条件。后来 micro-plan 还明确写出，work is not blocked 只表示可以从 T001 开始，不表示 T024+ 的行为实现已经安全。

这个 distinction 很重要，因为 agent 常常把“项目没有被 blocker 卡住”理解成“可以直接实现”。Workflow Pack 在这里把 readiness 拆细：可以开始调查，不等于可以改行为；可以写 RED，不等于可以写 GREEN；可以生成 synthetic artifact，不等于可以触碰 production output。

### 7.3 重复 selection logic

如果 figure-generation 和 audit-only mode 各自实现 Jaccard scoring、tie-break 和 readability gate，两个模式会逐渐分叉。GeoDT 的 review loop 发现这个问题后，把 T041–T043 提前为 shared selection pipeline，并要求 T032 和 T044 只能消费它，不能重复实现。

这个决定看起来只是代码复用，实际是审计一致性要求。Audit-only 的意义，是在不渲染图的情况下验证同一套 selection decision；如果它和 figure mode 不是同一条路径，就会变成另一个“看起来像验证、实际验证别的东西”的 artifact。

### 7.4 输出不可审计

没有 metadata 的 figure 只是图片，不是解释性证据。GeoDT spec 要求 metadata 记录 selected archive、artifact-provider path、branch assignment source、feature-name source、checkpoint paths、root/global exclusion、K、score、tie-break、readability、output paths 和 failure summary。后来的 metadata writer 还扩展到记录 candidate archive decisions、selected branch IDs、selected pair score、rejected higher-scoring pairs 和 neutral class-label policy。

这样一来，reviewer 不需要猜图是怎么来的，也不需要从文件名反推证据链。Metadata 把“这张图为什么可信”从口头解释变成了机器可检查、人工可审计的 contract。

### 7.5 验证只证明“能跑”

GeoDT 的 validation 必须映射到风险：focused tests、synthetic CLI smoke、audit-only smoke、reproduction smoke、source-boundary check、known baseline vs new regression summary。最终 focused test suite 通过 15 个 tests，synthetic CLI smoke 在 `/tmp` 临时 archive 下生成 PNG 和 metadata，确认 outputs 存在、workflow_mode 正确、selected pair readable，同时不触碰 production artifacts。

这个验证组合不是为了追求命令数量，而是为了避免“只证明 happy path 能跑”。Workflow Pack 要求 validation status 和 acceptance readiness 分开，不允许用“测试没跑但看起来对”冒充完成。

## 8. HIGH/CRITICAL review loop 的真正作用

GeoDT 中的严重 review 问题不是 reviewer 太严格，而是流程缺口的报警器。它们暴露的是 source-of-truth drift、readiness gate 不足、shared pipeline 缺失、metadata contract 不完整或 validation 未对齐风险。

因此，处理 HIGH/CRITICAL 的正确层级应该是：

- 如果问题来自 scope，就修 spec、tasks 或 source-of-truth；
- 如果问题来自 architecture，就修 plan；
- 如果问题来自 execution order，就修 task dependencies 和 micro-plan；
- 如果问题来自验证不足，就修 RED gate 和 validation command。

不要把所有严重问题都推到代码层解决。比如 selection pipeline 分叉风险，真正的修复不是在两个函数里复制同样逻辑，而是把 shared selection pipeline 提前为 foundational path；readiness 过早放行，真正的修复也不是提醒 agent “小心一点”，而是让 T018A 成为硬 gate。

Review loop 的结束条件不应该是“代码改完”，而应该是“下一轮同类问题更难出现”。GeoDT 的收敛来自 gate、contract、metadata 和 validation 的加固，而不是某一次局部实现。

## 9. 从 GeoDT 抽象出的 Workflow Pack 1.0 原则

### 原则一：reference-only artifacts 只能约束，不能授权

M0 baseline、migrated specs、memory summaries 和旧 evidence 可以提示 regression risk，但不能授权当前 feature 改行为。历史记录可以帮助我们理解 `dt_rules` 的语义、branch checkpoint 的存在、previous validation 的范围和 artifact hygiene 的风险；但这些信息只能让我们更谨慎，不能让我们绕过 active spec。

### 原则二：legacy behavior 默认受保护

在 Brownfield 项目中，未被 spec 明确允许的行为变化就是禁止项。GeoDT 没有为了 branch-tree figure 去改 training、dispatch、lag schedule、batch launcher 或 production prediction scripts，这不是保守，而是降低 review surface 和回滚风险。

### 原则三：SDD artifacts 必须单向流动

`spec.md` 定义需求，`plan.md` 定义架构和 change surface，`tasks.md` 定义依赖和验收路径，`implementation-micro-plan.md` 只能执行这些决定。若 micro-plan 暗含了上游没有授权的 scope，正确做法是回到上游修正，而不是让下游文件反向改写需求。

### 原则四：TDD 是行为变化的进入条件

任何 runtime、CLI、API、data artifact、metadata、output-path、config behavior 的变化，都必须先有 RED gate。Archive preflight、checkpoint classification、feature-name compatibility、selection scoring、readability gate、metadata schema、no-overwrite failure 和 reproduction mismatch 都不是“实现完再测”的边角料，而是 behavior 本身。

### 原则五：metadata 是 ML interpretability 的一等产物

对解释性、预测、诊断和可视化任务来说，metadata 不是附属文件，而是证明 artifact provenance、selection logic、assumption status 和 reproducibility 的核心 contract。GeoDT 的 PNG 如果没有 metadata，最多是一张示意图；有了 metadata，它才变成一条可审计证据链。

### 原则六：validation 必须对应风险，而不是对应方便运行的命令

容易跑的 happy-path smoke 不等于充分验证。GeoDT 的风险是 misleading explanation，因此 validation 必须覆盖 wrong source、wrong feature names、wrong branch、overwrite、unbounded search、unreadable figure 和 reproduction drift。验证命令应该从风险清单倒推，而不是从最省事的命令正推。

## 10. 可复用执行模板

下一次在 Brownfield research / ML repo 中开发解释性、预测、可视化、scenario 或 deliverable 相关功能时，可以按以下顺序走：

1. **声明 active feature 和 source-of-truth order**：确认当前 active feature directory、spec/plan/tasks/evidence、resolved analyze notes 和 reference-only artifacts。不要让 memory 或 migrated artifacts 反向定义 scope。
2. **写 Brownfield change surface**：明确 allowed files、read-only files、forbidden files、runtime config touched、data artifacts read/write、rollback path 和 compatibility risks。
3. **把 artifact assumptions 变成 preflight requirements**：如果功能读取历史模型、数据、文件系统或输出目录，先写 archive/data characterization、bounded discovery、minimum artifact set、compatibility evidence 和 structured failure summary。
4. **用 tasks.md 设计 gate，而不是只列 todo**：把 characterization、fixtures、focused regression、implementation、validation 和 documentation 分层。行为实现任务必须依赖对应 RED gates。
5. **生成 implementation micro-plan**：为每个 task 写清 requirement source、dependencies、files to inspect/edit、TDD classification、RED target、command、expected failure、GREEN minimal change、validation command 和 done criteria。
6. **保持 RED-GREEN-REFACTOR 粒度**：不要把多个 task 合成“大改一把”。每个行为变化都先观察预期失败，再做最小实现，再跑 focused validation，最后才允许局部 refactor。
7. **最后做 source-boundary 和 implementation delta**：检查 read-only files、production artifacts、unexpected outputs、validation status、known baseline failures 和 accepted deviations。完成状态必须由验证支撑，而不是由 agent 自述支撑。

这个模板的目的不是让流程显得更正式，而是把风险前置。对 ML artifact 来说，清楚失败通常比错误成功更有价值。

## 11. 结尾：Workflow Pack 1.0 的价值是减少误导性成功

GeoDT branch-tree interpretability 的核心成果不只是一个 branch-tree comparison diagnostic，而是证明了一种工作流：先保护 legacy system，再让需求经受 SDD 交叉验证，最后用 TDD 把每个风险变成可复现 gate。

这个 diagnostic 能生成图，但更重要的是，它能解释图的证据来源、selection rationale、metadata contract 和 validation status。对 Vibe Coding 来说，这比“AI 写出一个脚本”更有意义，因为它证明 agent 可以在不破坏旧系统的前提下，围绕一个高风险解释性需求产出可审计结果。

在 legacy ML 项目里，最快的失败不一定是程序报错，而是生成了看似正确、实际错误的研究 artifact。Brownfield + SDD + TDD 的成本发生在前面：要写 change surface，要修 source-of-truth drift，要加 readiness gate，要先写 RED，要跑 focused validation。但它减少的是后面最昂贵的返工：解释错误、artifact 污染、scope 漂移、review 反复，以及已经被引用的结果需要撤回。

今天的 LLM coding agent 不是确定性编译器，而是概率性协作者。Prompt、spec、tests、metadata 和 review loop 的作用，是在模型可能走向的巨大空间里插路标、留绳索、做记录，让下一次 traversal 有复现的可能。SDD 把目标和路径显性化，TDD 用小步可观察失败控制实现节奏，Brownfield 则提供站场地图：哪些轨道还在服务旧系统，哪些坑不能踩，哪些东西即使看起来可用也不应该碰。

GeoDT 这个案例给我的结论是：如果输出要被论文、报告或政策分析引用，那么“能生成”远远不够。它必须能证明自己为什么可信、从哪里来、没有破坏什么，以及如何复现。Workflow Pack 1.0 的价值，不是让开发显得更正式，而是让 agent 的成功不再停留在表面：它既要完成任务，也要留下足够证据，让人知道这个完成是真的完成。

## Appendix：真实 metadata 片段

下面这个片段摘自一次真实 GeoDT branch-tree diagnostic run 的 sidecar metadata：`other_outputs/geodt_branch_tree_diagnostic_real_run/geodt_branch_tree_compare_2024-10_fs1_011_vs_1_metadata.json`。原文件还包含完整 candidate archive decision、branch eligibility、signature 和 readability 信息；这里保留最能说明 provenance 与可审计性的字段。

```json
{
  "workflow_mode": "figure-generation",
  "archive_discovery_input": {
    "type": "archive-path",
    "value": "deliverables/GeoDTExperiment_archived/GeoDTResults/result_GeoDT_2024_fs1_2024-10_visual"
  },
  "selected_archive": "deliverables/GeoDTExperiment_archived/GeoDTResults/result_GeoDT_2024_fs1_2024-10_visual",
  "artifact_provider_path": "result_GeoDT_1",
  "branch_assignment_source": {
    "type": "X_branch_id.npy",
    "path": "result_GeoDT_1/space_partitions/X_branch_id.npy",
    "status": "selected"
  },
  "feature_name_source": {
    "type": "feature_name_reference.csv",
    "path": "result_GeoDT_1/feature_name_reference.csv",
    "status": "selected",
    "checkpoint_feature_count": 159,
    "feature_name_count": 159,
    "feature_name_compatibility_result": "selected",
    "evidence_validation_status": "confirmed"
  },
  "selected_branch_ids": ["011", "1"],
  "checkpoint_paths_used": {
    "011": "result_GeoDT_1/checkpoints/dt_011",
    "1": "result_GeoDT_1/checkpoints/dt_1"
  },
  "root_global_exclusion_example": {
    "branch_id": "",
    "checkpoint_path": "result_GeoDT_1/checkpoints/dt_",
    "checkpoint_classification": "root/global",
    "eligible": false,
    "exclusion_reasons": [
      "root/global checkpoint is excluded unless explicitly terminal",
      "checkpoint classification is root/global",
      "branch has no assignment count in selected source"
    ]
  },
  "selected_pair_score": 1.0,
  "selected_signatures_excerpt": {
    "011": {
      "split_feature_set": [
        "CPI",
        "crop",
        "gpp_mean_lag4m",
        "nightlight_lag4m",
        "sg_nitrogen_5-15cm"
      ]
    },
    "1": {
      "split_feature_set": [
        "EVI_l5",
        "FAO_price",
        "Rainf_f_tavg_mean_lag4m",
        "distance_to_nearest_acled_lag4m",
        "market_distance"
      ]
    }
  },
  "readability_result": {
    "passes": true,
    "reason": "readable",
    "plotted_max_depth": 3
  },
  "output_figure_paths": [
    "other_outputs/geodt_branch_tree_diagnostic_real_run/geodt_branch_tree_compare_2024-10_fs1_011_vs_1.png"
  ],
  "metadata_sidecar_path": "other_outputs/geodt_branch_tree_diagnostic_real_run/geodt_branch_tree_compare_2024-10_fs1_011_vs_1_metadata.json"
}
```

这个例子展示的重点不是字段多，而是每个关键判断都有落点：选中了哪个 archive、从哪里拿 branch assignment、用哪个 feature-name source 验证 checkpoint、实际比较了哪两个 branch-local checkpoints、为什么 root/global checkpoint 被排除、selection score 是多少，以及图和 metadata 最终写到了哪里。这样 reviewer 不需要凭文件名或口头说明猜测图的来源，而是可以沿 metadata 直接追溯证据链。
