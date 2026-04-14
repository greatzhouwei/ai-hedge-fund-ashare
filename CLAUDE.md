# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI Hedge Fund 是一个用于教育目的的 AI 驱动对冲基金概念验证项目。系统通过多个分析师 Agent（如 Warren Buffett、Cathie Wood、Michael Burry 等）并行分析股票数据，由 Risk Manager 评估风险，最终由 Portfolio Manager 输出交易决策。

**注意：本项目仅用于教育和研究，不适用于真实交易。**

**当前开发计划**：项目正在按照 [`plan.md`](./plan.md) 进行 A 股适配改造（数据源迁移至 Tushare）。所有后续功能开发、Bug 修复、代码审查均应参考 `plan.md` 中的任务清单与优先级，并在完成任务后及时更新 `plan.md` 中的勾选状态。

## 常用命令

### CLI 运行对冲基金（仅支持 A 股）
```bash
# 基础运行
poetry run python src/main.py --tickers 000001.SZ,600519.SH,000858.SZ

# 指定日期范围
poetry run python src/main.py --tickers 000001.SZ,600519.SH,000858.SZ --start-date 2024-01-01 --end-date 2024-03-01

# 使用本地 Ollama 模型
poetry run python src/main.py --tickers 000001.SZ,600519.SH,000858.SZ --ollama
```

### 回测
```bash
# 运行回测
poetry run python src/backtester.py --tickers 000001.SZ,600519.SH

# 使用新的模块化回测 CLI
poetry run python -m src.backtesting.cli --tickers 000001.SZ,600519.SH --start-date 2024-01-01 --end-date 2024-03-01 --analysts-all
```

### Web 应用
```bash
# 自动启动前后端（推荐）
cd app && ./run.sh        # macOS/Linux
cd app && ./run.bat       # Windows

# 手动启动后端（从项目根目录）
poetry run uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000

# 手动启动前端
cd app/frontend && npm install && npm run dev
```

### 代码格式化与检查
```bash
# Python 格式化
poetry run black src/ app/
poetry run isort src/ app/
poetry run flake8 src/ app/

# 前端 lint
cd app/frontend && npm run lint
```

### 测试
```bash
# 运行所有测试
poetry run pytest

# 运行单个测试文件
poetry run pytest tests/backtesting/test_portfolio.py

# 运行单个测试函数
poetry run pytest tests/backtesting/test_portfolio.py::test_some_function
```

## 架构概览

### 核心工作流（LangGraph）

决策流程基于 `langgraph.StateGraph`，定义在 `src/main.py` 的 `create_workflow()` 中：

1. `start_node` 分发任务
2. 选中的 **analyst agents** 并行执行
3. 全部完成后进入 `risk_management_agent`
4. 随后进入 `portfolio_manager`
5. 最后到达 `END`

状态由 `src/graph/state.py` 中的 `AgentState` 管理，包含 `messages`、`data`、`metadata` 三个字段，使用自定义归约函数合并。

### Agent 实现模式

所有 Agent 位于 `src/agents/`，遵循统一模式：
- 函数签名接收 `state: AgentState`
- 通过 `src.tools.api` 获取金融数据（价格、财务指标、内幕交易、新闻等）
- 使用 `src/utils/llm.call_llm()` 调用 LLM
- 输出结构化信号（`signal`: bullish/bearish/neutral、`confidence`: 0-100、`reasoning`）

Agent 的注册和元数据配置集中在 `src/utils/analysts.py` 的 `ANALYST_CONFIG` 中，新增 Agent 时必须在此注册。

### 数据层（Tushare）

- **外部 API**：**Tushare Pro API**（需要 `TUSHARE_TOKEN`）
- **本地缓存**：`src/data/cache.py` 实现了基于内存的数据缓存，价格、财务指标、内幕交易、新闻等均有缓存
- **数据模型**：`src/data/models.py` 使用 Pydantic 定义 API 响应结构
- **统一入口**：`src/tools/api.py` 封装所有数据获取逻辑，内部调用 Tushare 接口并将数据映射到统一模型
- **支持市场**：仅支持 A 股市场（代码格式如 `000001.SZ`、`600519.SH`）

### 回测引擎

`src/backtesting/` 包含模块化的回测系统：
- `engine.py` — `BacktestEngine`：按工作日循环，每日调用 Agent 决策、执行交易、记录组合价值
- `trader.py` — `TradeExecutor`：处理 buy/sell/short/cover/hold 动作
- `portfolio.py` — `Portfolio`：管理现金、持仓、成本基础、保证金、已实现盈亏
- `metrics.py` — `PerformanceMetricsCalculator`：夏普比率、索提诺比率、最大回撤
- `controller.py` — `AgentController`：包装 `run_hedge_fund` 调用，统一异常处理
- `benchmarks.py` — `BenchmarkCalculator`：与 `000300.SH`（沪深300）对比收益

旧版 `src/backtester.py` 保留向后兼容，内部逻辑已迁移到 `src/backtesting/`。

### Web 应用

`app/` 目录包含完整前后端应用：

**后端 (`app/backend/`)**
- FastAPI，端口 8000，CORS 允许 `localhost:5173`
- SQLite 数据库（项目根目录 `hedge_fund.db`），SQLAlchemy + Alembic
- 路由模块：`hedge_fund.py`（执行图）、`flows.py`（工作流 CRUD）、`flow_runs.py`（运行记录）、`api_keys.py`（密钥管理）、`ollama.py`（本地模型状态）
- `app/backend/services/graph.py` 核心功能：根据前端 React Flow 图结构动态构建 LangGraph，支持多个 Portfolio Manager 和风险经理的映射

**前端 (`app/frontend/`)**
- React + TypeScript + Vite，端口 5173
- `@xyflow/react` 构建可视化工作流编辑器
- shadcn/ui + Tailwind CSS
- 状态管理通过多个 React Context（`flow-context`、`tabs-context`、`node-context`）

### 环境变量

API 密钥通过项目根目录的 `.env` 文件管理（复制自 `.env.example`）。运行前至少需要设置：
- `TUSHARE_TOKEN`（Tushare Pro 接口 token）
- 至少一个 LLM 提供商密钥（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GROQ_API_KEY` 等）

支持的模型提供商包括：OpenAI、Anthropic、Groq、DeepSeek、Google Gemini、xAI、Ollama（本地）、Azure OpenAI、OpenRouter、GigaChat、Moonshot（Kimi）。

### Docker

`docker/Dockerfile` 基于 `python:3.11-slim`，使用 Poetry 安装依赖，设置 `PYTHONPATH=/app`。

---

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
