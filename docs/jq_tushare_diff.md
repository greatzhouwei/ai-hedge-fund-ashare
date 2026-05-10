# 聚宽 vs Tushare 三维度打分差异清单

> 创建日期: 2026-05-10
> 背景: 对齐聚宽三维度综合评分策略与 Tushare/DuckDB 版本的打分结果

---

## 说明

评分逻辑（权重、阈值、加分规则）两边已完全对齐。以下差异集中在**数据来源**和**计算方式**，导致相同公式下输入值不同，最终得分不同。

---

## 差异 1: 基本面 — EPS 来源不同

- **聚宽**: `indicator.eps`（聚宽财务指标表提供的 EPS）
- **Tushare**: `income.basic_eps`（利润表中的基本每股收益）
- **影响**: 两个值在多只股票上不同。例如 600208.SH，聚宽 EPS=0.05，Tushare=0.233
- **解决方向**: 确认 Tushare 是否有与 `indicator.eps` 对等的字段；或在 `fina_indicator` 中查找 `eps` 字段

## 差异 2: 基本面 — 股东权益字段不同

- **聚宽**: `balance.equities_parent_company_owners`
- **Tushare**: `balance.total_hldr_eqy_exc_min_int`
- **影响**: A 股中通常相同或非常接近，一般不影响得分
- **解决方向**: 验证两字段在 A 股中是否等价；如有差异，确认哪个更接近聚宽语义

## 差异 3: 成长性 — EPS 增长率计算方式不同

- **聚宽**: 优先用 TTM，fallback 用**累计 basic_eps 同比**  
  公式: `(今年Q3累计EPS - 去年Q3累计EPS) / 去年Q3累计EPS`
- **Tushare**: **TTM basic_eps 同比**（先转单季度，再 rolling(4) 求和，再同比）
- **影响**: 在非 Q4 时点结果不同；600208.SH 聚宽 +62.7% vs Tushare -25.8%
- **解决方向**: 将 Tushare 的 EPS 增长率改为**累计同比**，与聚宽保持一致

## 差异 4: 成长性 — FCF 增长率实质内容不同（关键差异）

- **聚宽**: 用**经营现金流净额**(`net_operate_cash_flow`)的累计同比，**没有减去资本支出**  
  （代码注释明确说"用经营活动现金流量净额代替自由现金流"）
- **Tushare**: 真正的 FCF（OCF - Capex）TTM 同比
- **影响**: 聚宽成长性的 FCF 增长实际上是 OCF 增长，不是真实的 FCF 增长
- **解决方向**: 将 Tushare 的 FCF 增长改为**OCF 累计同比**，去掉 Capex 扣除，与聚宽保持一致

## 差异 5: 成长性 — 营收增长率优先级不同

- **聚宽**: 优先用 `indicator.inc_total_revenue_year_on_year`（平台提供的营收同比增长率），fallback 手动 TTM
- **Tushare**: 直接用手动 TTM 营收同比
- **影响**: 当 indicator 提供的 YoY 和手动 TTM 不一致时，结果不同
- **解决方向**: 在 Tushare 中优先使用 `fina_indicator.tr_yoy` 或 `q_sales_yoy`，fallback 手动 TTM

## 差异 6: 成长性 — Margin 计算基础不同

- **聚宽**: 用累计 income 数据直接计算毛利率/营业利润率/净利率，优先 TTM margin
- **Tushare**: 用 TTM margin（单季度转后 rolling 求和），fallback `fina_indicator`
- **影响**: Q1/Q2/Q3 时，累计 margin 和 TTM margin 可能不同
- **解决方向**: 统一使用累计值直接计算 margin（不转单季度），与聚宽保持一致

## 差异 7: 成长性 — PEG 的净利润增长来源不同

- **聚宽**: `inc_net_profit_to_shareholders_year_on_year`（归母净利润同比增长率）
- **Tushare**: `dt_netprofit_yoy`（扣非净利润同比增长率）
- **影响**: 扣非和归母增长可能不同，导致 PEG 不同
- **解决方向**: 将 Tushare 的 PEG 净利润增长改为 `netprofit_yoy`（归母净利润同比），与聚宽对齐

## 差异 8: 成长性 — 数据合并方式不同

- **聚宽**: `get_financial_metrics` 对 income / indicator / balance / cashflow 做 **inner join**，某些季度只要任一表缺失就会被剔除
- **Tushare**: 各表独立查询，不做 join
- **影响**: 聚宽可能因某表缺数据而丢失可用季度，导致 fallback 到不同的增长率
- **解决方向**: Tushare 侧不做 inner join 是合理的（数据更完整），此差异无需修复，但需记录

## 差异 9: 数据去重策略不同

- **聚宽**: `get_history_fundamentals` 内部自动去重
- **Tushare**: 手动 `drop_duplicates(subset=["end_date"], keep="first")`
- **影响**: 当同一 end_date 有多行时，保留策略可能不同
- **解决方向**: 检查 Tushare 数据重复的原因；如为数据质量问题，在入库时修复

## 差异 10: 技术面 — 无实质差异

- 趋势/均值回归/动量/波动率的权重、阈值、公式完全一致
- **解决方向**: 无需修复

---

## 优先级排序

| 优先级 | 差异 | 理由 |
|--------|------|------|
| P0 | 差异 3 (EPS增长率) | 导致多只股票 growth 得分偏差巨大 |
| P0 | 差异 4 (FCF增长率) | 聚宽实际用 OCF 替代 FCF，概念不同 |
| P1 | 差异 1 (EPS来源) | 直接影响基本面 health 打分 |
| P1 | 差异 5 (营收增长率) | 影响 growth 子模块核心指标 |
| P1 | 差异 7 (PEG净利润来源) | 影响 valuation 子模块 |
| P2 | 差异 6 (Margin计算) | 影响 margin_trends 子模块 |
| P2 | 差异 2 (股东权益字段) | 影响较小，验证即可 |
| P2 | 差异 8 (数据合并) | Tushare 侧更合理，无需修复 |
| P3 | 差异 9 (去重策略) | 数据质量问题，可在 ETL 层解决 |
| - | 差异 10 (技术面) | 无需修复 |
