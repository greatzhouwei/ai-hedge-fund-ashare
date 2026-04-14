# 改造计划：AI Hedge Fund 纯 A 股适配（Tushare）

## 背景与目标

将现有 `ai-hedge-fund` 系统从美股数据源（`financialdatasets.ai`）**彻底改造为仅支持 A 股市场**，数据接口统一使用 **Tushare**。无需保留美股兼容性，无需市场识别层。

## 关键约束

- 股票代码格式：Tushare 标准 `ts_code`，如 `000001.SZ`、`600000.SH`、`688001.SH`
- 日期格式：现有代码使用 `YYYY-MM-DD`，Tushare 接口要求 `YYYYMMDD`，需在数据层统一转换
- Tushare 新闻接口**不支持按股票代码查询**，`get_company_news` 将暂时返回空列表（后续可扩展）
- 缓存为内存缓存（`src/data/cache.py`），无需持久化格式调整

## Tushare 接口映射

| 现有函数 | Tushare 接口 | 说明 |
|---------|-------------|------|
| `get_prices` | `pro.daily(ts_code=..., start_date=..., end_date=...)` | 日线行情，返回字段含 `open/high/low/close/vol/amount` |
| `get_financial_metrics` | `pro.fina_indicator(ts_code=...)` + `pro.daily_basic(ts_code=..., trade_date=...)` | 财务指标 + 每日估值指标（总市值、PE、PB） |
| `search_line_items` | `pro.income` / `pro.balancesheet` / `pro.cashflow` | 根据 line_item 名称匹配到对应报表接口查询 |
| `get_insider_trades` | `pro.stk_holdertrade(ts_code=...)` | 股东增减持数据 |
| `get_company_news` | **返回 `[]`** | Tushare 无按代码查新闻的免费接口，暂做空处理 |
| `get_market_cap` | `pro.daily_basic` 的 `total_mv` 字段 | 总市值，单位：万元 |

## 改造步骤

### Phase 1：依赖与环境配置
- [x] 1.1 `pyproject.toml`
  - [x] 添加 `tushare = "^1.4.0"` 到 `[tool.poetry.dependencies]`
  - [x] 移除 `FINANCIAL_DATASETS_API_KEY` 相关说明（保留其他 LLM key）
- [x] 1.2 `.env.example`
  - [x] 将 `FINANCIAL_DATASETS_API_KEY` 替换为 `TUSHARE_TOKEN=your-tushare-token`
- [x] 1.3 全项目替换 `FINANCIAL_DATASETS_API_KEY` 为 `TUSHARE_TOKEN`
  - [x] `src/utils/api_key.py` 及所有 agent 文件
  - [x] `app/frontend/src/components/settings/api-keys.tsx`
  - [x] `app/backend/services/backtest_service.py`
  - [x] `app/run.sh` 与 `app/run.bat`

### Phase 2：数据模型清理
- [x] 2.1 `src/data/models.py`
  - [x] `CompanyFacts` 中移除美股特有字段：`cik`, `sic_code`, `sic_industry`, `sic_sector`, `sec_filings_url`
  - [x] 添加 A 股相关可选字段：`area` (地区), `industry` (行业，来自 Tushare `stock_basic`)
  - [x] 其余模型（`Price`, `FinancialMetrics`, `LineItem`, `InsiderTrade`）字段名称保持通用，确认无需修改

### Phase 3：核心数据层替换
- [x] 3.1 `src/tools/api.py` 完全重写
  - [x] 移除所有 `financialdatasets.ai` HTTP 请求逻辑
  - [x] 引入 Tushare Pro API：`pro = ts.pro_api(os.environ.get("TUSHARE_TOKEN"))`
  - [x] 实现日期格式转换辅助函数：`to_tushare_date(date_str) -> str`
  - [x] 重写 `get_prices`：调用 `pro.daily`，将 `vol` 映射为 `volume`，`trade_date` 映射为 `time`
  - [x] 重写 `get_financial_metrics`：合并 `fina_indicator` 和 `daily_basic` 的结果，填充 `FinancialMetrics` 字段
  - [x] 重写 `search_line_items`：根据 `line_items` 列表匹配到 `income`/`balancesheet`/`cashflow` 接口，返回 `LineItem` 列表
  - [x] 重写 `get_insider_trades`：调用 `pro.stk_holdertrade`，映射增减持字段到 `InsiderTrade` 模型
  - [x] 重写 `get_company_news`：直接返回 `[]`（带 warning log）
  - [x] 重写 `get_market_cap`：调用 `pro.daily_basic` 取 `total_mv`（万元）
  - [x] 所有函数添加适当的错误处理和降级（Tushare 返回空 DataFrame 时返回 `[]` 或 `None`）

### Phase 4：CLI 与交互层适配
- [x] 4.1 `src/cli/input.py`
  - [x] 修改 `--tickers` 的 help text：示例从 `AAPL,MSFT,GOOGL` 改为 `000001.SZ,600519.SH,000858.SZ`
- [x] 4.2 `src/backtesting/engine.py`
  - [x] 将基准比较代码中的 `SPY` 改为 `000300.SH`（沪深300）
  - [x] 修改 `_prefetch_data()` 中的基准股票代码
- [x] 4.3 `src/backtesting/benchmarks.py`
  - [x] 确认通用性，默认基准 ticker 调用端已改为 `000300.SH`（沪深300）

### Phase 5：Agent 清理
- [x] 5.1 `src/agents/risk_manager.py`
  - [x] 将 `"Convert to dollar position limit"` 注释改为 `"Convert to RMB position limit"`
- [x] 5.2 全局搜索与货币符号清理
  - [x] 扫描所有 `src/agents/*.py` 及 `src/utils/display.py`、`src/backtester.py`
  - [x] 将所有实际数据输出中的 `$` 替换为 `¥`（涉及 risk_manager, nassim_taleb, mohnish_pabrai, valuation, warren_buffett, display, backtester）

### Phase 6：测试更新
- [x] 6.1 `tests/test_api_rate_limiting.py`
  - [x] 重写为测试 Tushare 客户端的异常处理（空 DataFrame、网络超时等）
- [x] 6.2 全量回归测试
  - [x] 运行 `poetry run pytest`（69 tests passed）
  - [x] 修复 `tests/backtesting/integration/conftest.py` 的 Windows 编码问题（`encoding="utf-8"`）

### Phase 7：文档更新
- [x] 7.1 `CLAUDE.md`
  - [x] 更新运行命令示例为 A 股代码
  - [x] 更新环境变量说明（`TUSHARE_TOKEN`）
  - [x] 明确说明系统仅支持 A 股
- [x] 7.2 `README.md`
  - [x] 更新安装和运行说明，替换为 A 股相关内容
  - [x] 更新环境变量配置示例

## 验证方式

- [x] 安装依赖：`poetry install`（已通过清华镜像完成）
- [ ] 配置 `.env`：`TUSHARE_TOKEN=xxx`（需用户自行配置真实 token）
- [ ] 运行 CLI：
  ```bash
  poetry run python src/main.py --tickers 000001.SZ,600519.SH --start-date 2024-01-01 --end-date 2024-03-01
  ```
- [ ] 确认各 Agent 能正常获取数据并输出信号
- [ ] 运行回测：
  ```bash
  poetry run python src/backtester.py --tickers 000001.SZ --start-date 2024-01-01 --end-date 2024-12-31
  ```
- [x] 运行测试：`poetry run pytest`（69 tests passed）

## 风险与注意事项

| 风险 | 级别 | 应对 |
|------|------|------|
| Tushare 部分接口需积分 | 中 | `pro.daily`、`fina_indicator`、`stk_holdertrade` 等为基础免费接口，应无问题；如遇积分限制，输出清晰错误提示 |
| `search_line_items` 映射复杂 | 中 | 需要建立 line_item 名称到 Tushare 字段名的映射字典，映射不到的返回空或跳过 |
| 新闻数据缺失 | 低 | 暂返回空列表，不影响核心分析流程 |
| 财报指标与美股模型字段不完全对齐 | 中 | 对于 `FinancialMetrics` 中 Tushare 没有的字段，保持 `None`，在 Agent prompt 中说明部分指标可能不可用 |
| 基准从 SPY 改为沪深300 | 低 | 需在回测引擎和 benchmarks 中同步修改 |
