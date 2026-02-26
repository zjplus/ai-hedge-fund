import datetime
import os
import pandas as pd
import numpy as np
import time
import threading
import random

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
    CompanyFacts,
)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("请安装 yfinance: uv add yfinance")

# Global cache instance
_cache = get_cache()

# ---------------------------------------------------------------------------
# yfinance 请求节流 & 重试
# ---------------------------------------------------------------------------
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0.0
_MIN_REQUEST_INTERVAL = 0.35  # 两次请求之间最少间隔（秒）
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # 重试基础等待秒数（指数退避）

# Ticker 对象 & info 缓存（进程内，减少重复 HTTP 请求）
_ticker_cache: dict[str, yf.Ticker] = {}
_info_cache: dict[str, dict] = {}
_info_cache_ts: dict[str, float] = {}
_INFO_CACHE_TTL = 300  # info 缓存有效期 5 分钟


def _throttle():
    """全局限流：确保两次请求之间有足够间隔。"""
    global _LAST_REQUEST_TIME
    with _REQUEST_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_TIME
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        _LAST_REQUEST_TIME = time.monotonic()


def _retry_on_rate_limit(func, *args, **kwargs):
    """带指数退避的重试封装，处理 429 / 401 限流错误。"""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            _throttle()
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(kw in err_msg for kw in [
                "rate limit", "too many requests", "429", "unauthorized", "invalid crumb"
            ])
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⏳ yfinance 限流，{delay:.1f}s 后重试 (第 {attempt + 2}/{_MAX_RETRIES} 次)...")
                time.sleep(delay)
                last_exc = e
                # 限流后清理 crumb 缓存
                try:
                    yf.utils.get_json = lambda *a, **k: None  # noqa – 不可靠，仅占位
                except:
                    pass
            else:
                raise
    raise last_exc


def _detect_market(ticker: str) -> str:
    """根据 ticker 格式检测市场类型。
    - 港股: 0700.HK, 9988.HK
    - A股: 600519.SS (上交所), 000858.SZ (深交所)
    - 美股: AAPL, MSFT
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith(".HK"):
        return "HK"
    elif ticker_upper.endswith(".SS") or ticker_upper.endswith(".SZ"):
        return "CN"
    else:
        return "US"


def get_market_params(ticker: str) -> dict:
    """获取市场特定的参数（无风险利率、风险溢价、税率、年交易日数）。"""
    market = _detect_market(ticker)
    if market == "HK":
        return {
            "market": "HK",
            "market_name": "香港",
            "risk_free_rate": 0.04,
            "equity_risk_premium": 0.065,
            "corporate_tax_rate": 0.165,
            "trading_days_per_year": 245,
            "currency": "HKD",
        }
    elif market == "CN":
        return {
            "market": "CN",
            "market_name": "中国A股",
            "risk_free_rate": 0.025,
            "equity_risk_premium": 0.07,
            "corporate_tax_rate": 0.25,
            "trading_days_per_year": 242,
            "currency": "CNY",
        }
    else:
        return {
            "market": "US",
            "market_name": "美国",
            "risk_free_rate": 0.045,
            "equity_risk_premium": 0.05,
            "corporate_tax_rate": 0.25,
            "trading_days_per_year": 252,
            "currency": "USD",
        }


def _get_yf_ticker(ticker: str) -> yf.Ticker:
    """获取 yfinance Ticker 对象（带缓存）。"""
    if ticker not in _ticker_cache:
        _ticker_cache[ticker] = yf.Ticker(ticker)
    return _ticker_cache[ticker]


def _get_info(ticker: str) -> dict:
    """获取 ticker.info 并缓存，避免同一 ticker 重复请求 .info。"""
    now = time.monotonic()
    if ticker in _info_cache and (now - _info_cache_ts.get(ticker, 0)) < _INFO_CACHE_TTL:
        return _info_cache[ticker]

    yf_ticker = _get_yf_ticker(ticker)
    info = _retry_on_rate_limit(lambda: yf_ticker.info)
    _info_cache[ticker] = info or {}
    _info_cache_ts[ticker] = time.monotonic()
    return _info_cache[ticker]


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """通过 yfinance 获取价格数据。"""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    try:
        yf_ticker = _get_yf_ticker(ticker)
        # yfinance end_date 是不包含当天的，需要加一天
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
        df = _retry_on_rate_limit(
            lambda: yf_ticker.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True)
        )

        if df.empty:
            return []

        prices = []
        for idx, row in df.iterrows():
            prices.append(Price(
                open=float(row["Open"]),
                close=float(row["Close"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"]),
                time=idx.strftime("%Y-%m-%d"),
            ))

        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices
    except Exception as e:
        print(f"获取价格数据失败 ({ticker}): {e}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """通过 yfinance 获取财务指标。"""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    try:
        yf_ticker = _get_yf_ticker(ticker)
        info = _get_info(ticker)

        market_params = get_market_params(ticker)
        currency = info.get("currency", market_params["currency"])

        # 基本财务指标
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")

        # 估值比率
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        pb_ratio = info.get("priceToBook")
        ps_ratio = info.get("priceToSalesTrailing12Months")
        ev_to_ebitda = info.get("enterpriseToEbitda")
        ev_to_revenue = info.get("enterpriseToRevenue")
        peg_ratio = info.get("pegRatio")

        # 利润率
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        net_margin = info.get("profitMargins")

        # 回报率
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")

        # 杠杆指标
        debt_to_equity = info.get("debtToEquity")
        if debt_to_equity is not None:
            debt_to_equity = debt_to_equity / 100.0  # yfinance 返回百分比

        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")

        # 成长性
        revenue_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")

        # 每股指标
        eps = info.get("trailingEps")
        bvps = info.get("bookValue")
        fcf_per_share = None
        free_cash_flow = info.get("freeCashflow")
        shares_outstanding = info.get("sharesOutstanding")
        if free_cash_flow and shares_outstanding:
            fcf_per_share = free_cash_flow / shares_outstanding

        # FCF yield
        fcf_yield = None
        if free_cash_flow and market_cap and market_cap > 0:
            fcf_yield = free_cash_flow / market_cap

        # Interest coverage
        interest_coverage = None
        try:
            financials = _retry_on_rate_limit(lambda: yf_ticker.financials)
            if financials is not None and not financials.empty:
                ebit = None
                interest = None
                for col_name in ["EBIT", "Operating Income"]:
                    if col_name in financials.index:
                        ebit = financials.loc[col_name].iloc[0]
                        break
                if "Interest Expense" in financials.index:
                    interest = abs(financials.loc["Interest Expense"].iloc[0])
                if ebit and interest and interest > 0:
                    interest_coverage = float(ebit / interest)
        except:
            pass

        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency=currency,
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            price_to_earnings_ratio=pe_ratio,
            price_to_book_ratio=pb_ratio,
            price_to_sales_ratio=ps_ratio,
            enterprise_value_to_ebitda_ratio=ev_to_ebitda,
            enterprise_value_to_revenue_ratio=ev_to_revenue,
            free_cash_flow_yield=fcf_yield,
            peg_ratio=peg_ratio,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=debt_to_equity,
            debt_to_assets=None,
            interest_coverage=interest_coverage,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=info.get("payoutRatio"),
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcf_per_share,
        )

        result = [metrics]
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
        return result
    except Exception as e:
        print(f"获取财务指标失败 ({ticker}): {e}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """通过 yfinance 获取财务报表明细项。"""
    try:
        yf_ticker = _get_yf_ticker(ticker)
        market_params = get_market_params(ticker)
        currency = market_params["currency"]

        # 获取财务报表数据
        financials = _retry_on_rate_limit(lambda: yf_ticker.financials)  # 年度损益表
        balance = _retry_on_rate_limit(lambda: yf_ticker.balance_sheet)  # 资产负债表
        cashflow = _retry_on_rate_limit(lambda: yf_ticker.cashflow)  # 现金流量表

        if period == "quarterly":
            financials = _retry_on_rate_limit(lambda: yf_ticker.quarterly_financials)
            balance = _retry_on_rate_limit(lambda: yf_ticker.quarterly_balance_sheet)
            cashflow = _retry_on_rate_limit(lambda: yf_ticker.quarterly_cashflow)

        # 合并所有数据源
        all_data = {}
        for source in [financials, balance, cashflow]:
            if source is not None and not source.empty:
                for col_name in source.index:
                    # 将 yfinance 的字段名转为 snake_case
                    snake_name = col_name.replace(" ", "_").lower()
                    all_data[snake_name] = source.loc[col_name]

        # 构建 line items 列表 - 按时间段返回多条
        results = []
        # 获取可用的时间段
        date_columns = set()
        for source in [financials, balance, cashflow]:
            if source is not None and not source.empty:
                for col in source.columns:
                    date_columns.add(col)

        sorted_dates = sorted(date_columns, reverse=True)[:limit]

        # yfinance 字段名到常用字段名的映射
        field_mapping = {
            # 损益表
            "revenue": ["Total Revenue", "Revenue", "Operating Revenue"],
            "net_income": ["Net Income", "Net Income Common Stockholders"],
            "operating_income": ["Operating Income", "EBIT"],
            "gross_profit": ["Gross Profit"],
            "ebitda": ["EBITDA", "Normalized EBITDA"],
            "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
            "income_tax_expense": ["Tax Provision", "Income Tax Expense"],
            "research_and_development": ["Research And Development", "Research Development"],
            "depreciation_and_amortization": ["Depreciation And Amortization", "Reconciled Depreciation"],
            "total_revenue": ["Total Revenue", "Revenue"],
            "cost_of_revenue": ["Cost Of Revenue"],
            "selling_general_and_administrative": ["Selling General And Administration"],
            # 资产负债表
            "total_assets": ["Total Assets"],
            "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liab"],
            "total_equity": ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
            "total_debt": ["Total Debt", "Long Term Debt"],
            "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
            "short_term_debt": ["Current Debt", "Short Long Term Debt"],
            "cash_and_equivalents": ["Cash And Cash Equivalents", "Cash"],
            "cash_and_short_term_investments": ["Cash Cash Equivalents And Short Term Investments"],
            "inventory": ["Inventory"],
            "accounts_receivable": ["Accounts Receivable", "Net Receivables"],
            "accounts_payable": ["Accounts Payable"],
            "current_assets": ["Current Assets"],
            "current_liabilities": ["Current Liabilities"],
            "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
            "book_value_per_share": ["Book Value"],
            # 现金流量表
            "operating_cash_flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
            "capital_expenditure": ["Capital Expenditure", "Capital Expenditures"],
            "free_cash_flow": ["Free Cash Flow"],
            "dividends_paid": ["Common Stock Dividend Paid", "Cash Dividends Paid"],
            "share_repurchase": ["Repurchase Of Capital Stock", "Common Stock Payments"],
            "issuance_of_debt": ["Issuance Of Debt", "Long Term Debt Issuance"],
            "repayment_of_debt": ["Repayment Of Debt", "Long Term Debt Payments"],
            "net_income_from_cashflow": ["Net Income From Continuing Operations"],
        }

        for date_col in sorted_dates:
            date_str = date_col.strftime("%Y-%m-%d") if hasattr(date_col, "strftime") else str(date_col)

            # 过滤到 end_date 之前的数据
            if date_str > end_date:
                continue

            item_data = {
                "ticker": ticker,
                "report_period": date_str,
                "period": period,
                "currency": currency,
            }

            for requested_item in line_items:
                value = None
                # 尝试直接在各个数据源中查找
                yf_names = field_mapping.get(requested_item, [requested_item])
                for source in [financials, balance, cashflow]:
                    if source is not None and not source.empty and date_col in source.columns:
                        for yf_name in yf_names:
                            if yf_name in source.index:
                                val = source.loc[yf_name, date_col]
                                if pd.notna(val):
                                    value = float(val)
                                    break
                    if value is not None:
                        break

                item_data[requested_item] = value

            results.append(LineItem(**item_data))

        return results[:limit]
    except Exception as e:
        print(f"获取财务报表明细失败 ({ticker}): {e}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """通过 yfinance 获取内部人交易数据。
    注意：港股和A股的内部人交易数据可能不完整。"""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    try:
        yf_ticker = _get_yf_ticker(ticker)
        insider_df = _retry_on_rate_limit(lambda: yf_ticker.insider_transactions)

        if insider_df is None or insider_df.empty:
            return []

        trades = []
        for _, row in insider_df.iterrows():
            # 解析日期
            trade_date = None
            if "Start Date" in row.index and pd.notna(row.get("Start Date")):
                trade_date = pd.to_datetime(row["Start Date"]).strftime("%Y-%m-%d")
            elif "Date" in row.index and pd.notna(row.get("Date")):
                trade_date = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")

            if trade_date is None:
                continue

            # 过滤日期范围
            if trade_date > end_date:
                continue
            if start_date and trade_date < start_date:
                continue

            shares = row.get("Shares", 0) or 0
            value = row.get("Value", 0) or 0
            price = 0
            if shares != 0 and value != 0:
                price = abs(value / shares)

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=row.get("Insider", None),
                name=row.get("Insider", None),
                title=row.get("Position", None) or row.get("Relationship", None),
                is_board_director=None,
                transaction_date=trade_date,
                transaction_shares=float(shares) if shares else None,
                transaction_price_per_share=float(price) if price else None,
                transaction_value=float(value) if value else None,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=row.get("Text", None),
                filing_date=trade_date,
            ))

        trades = trades[:limit]
        if trades:
            _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
        return trades
    except Exception as e:
        print(f"获取内部人交易数据失败 ({ticker}): {e}")
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """通过 yfinance 获取公司新闻。"""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    try:
        yf_ticker = _get_yf_ticker(ticker)
        news_list = _retry_on_rate_limit(lambda: yf_ticker.news)

        if not news_list:
            return []

        articles = []
        for item in news_list:
            # yfinance news 结构
            pub_date = None
            if "providerPublishTime" in item:
                pub_date = datetime.datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")
            elif "content" in item and "pubDate" in item.get("content", {}):
                pub_date = item["content"]["pubDate"][:10]

            if pub_date is None:
                pub_date = end_date

            # 过滤日期范围
            if pub_date > end_date:
                continue
            if start_date and pub_date < start_date:
                continue

            title = item.get("title", "")
            if "content" in item:
                title = item["content"].get("title", title)

            url = item.get("link", "") or item.get("url", "")
            if "content" in item:
                url = item["content"].get("canonicalUrl", {}).get("url", url)

            source = item.get("publisher", "")
            if "content" in item:
                source = item["content"].get("provider", {}).get("displayName", source)

            author = ""
            if "content" in item and "byline" in item.get("content", {}):
                author = item["content"].get("byline", "")

            articles.append(CompanyNews(
                ticker=ticker,
                title=title,
                author=author,
                source=source,
                date=pub_date,
                url=url,
                sentiment=None,
            ))

        articles = articles[:limit]
        if articles:
            _cache.set_company_news(cache_key, [a.model_dump() for a in articles])
        return articles
    except Exception as e:
        print(f"获取公司新闻失败 ({ticker}): {e}")
        return []


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """通过 yfinance 获取市值。"""
    try:
        info = _get_info(ticker)
        market_cap = info.get("marketCap")
        if market_cap:
            return float(market_cap)

        # fallback: 从 financial_metrics 获取
        metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
        if metrics and metrics[0].market_cap:
            return metrics[0].market_cap

        return None
    except Exception as e:
        print(f"获取市值失败 ({ticker}): {e}")
        return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """将价格列表转换为 DataFrame。"""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
