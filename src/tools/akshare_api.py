"""
AkShare 数据源 —— 作为 yfinance 的备份数据源。
主要用于 A 股和港股数据获取，也支持美股。
AkShare 是免费的，不需要 API key。

使用方法：
    uv add akshare
"""

import datetime
import time
import random
import pandas as pd
import numpy as np

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

try:
    import akshare as ak
except ImportError:
    ak = None

# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

# 个股信息缓存（轻量接口结果缓存，避免重复请求）
_individual_info_cache: dict[str, pd.DataFrame] = {}


def _is_available() -> bool:
    """检查 akshare 是否可用。"""
    return ak is not None


def _retry_call(func, *args, **kwargs):
    """带重试的 akshare 调用，处理网络断连等问题。"""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_msg = str(e).lower()
            is_retryable = any(kw in err_msg for kw in [
                "connection aborted", "remotedisconnected", "connection reset",
                "timeout", "timed out", "connection refused", "too many requests",
            ])
            if is_retryable and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"    [akshare] 网络异常，{delay:.1f}s 后重试 (第 {attempt + 2}/{_MAX_RETRIES} 次)...")
                time.sleep(delay)
            else:
                raise
    raise last_exc


def _convert_ticker_for_ak(ticker: str) -> tuple[str, str]:
    """
    将通用 ticker 格式转换为 akshare 所需格式。
    返回 (symbol, market)。
    
    - A股: 600519.SS -> (600519, CN), 000858.SZ -> (000858, CN)
    - 港股: 0700.HK -> (00700, HK)  
    - 美股: AAPL -> (AAPL, US)
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith(".SS") or ticker_upper.endswith(".SZ"):
        symbol = ticker_upper.split(".")[0]
        return symbol, "CN"
    elif ticker_upper.endswith(".HK"):
        symbol = ticker_upper.split(".")[0].zfill(5)
        return symbol, "HK"
    else:
        return ticker_upper, "US"


def _get_individual_info(symbol: str) -> pd.DataFrame:
    """获取单只 A 股个股信息（轻量接口，带缓存）。
    返回 DataFrame，包含 item/value 两列，如：
        总市值, 流通市值, 行业, 上市时间, 股票代码, 股票简称 等。
    """
    if symbol in _individual_info_cache:
        return _individual_info_cache[symbol]

    df = _retry_call(ak.stock_individual_info_em, symbol=symbol)
    if df is not None and not df.empty:
        _individual_info_cache[symbol] = df
    return df if df is not None else pd.DataFrame()


def _get_info_value(df: pd.DataFrame, item_name: str):
    """从 stock_individual_info_em 结果中按 item 名称取值。"""
    if df.empty:
        return None
    row = df[df["item"] == item_name]
    if row.empty:
        return None
    return row.iloc[0]["value"]


def _safe_float(val) -> float | None:
    """安全转换为 float，失败返回 None。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """安全转换为 int，失败返回 None。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 公开 API 函数（与 api.py 签名兼容）
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """通过 akshare 获取股票历史价格数据。"""
    if not _is_available():
        return []

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market == "CN":
            # A 股日线数据
            df = _retry_call(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",  # 前复权
            )
            if df is None or df.empty:
                return []
            prices = []
            for _, row in df.iterrows():
                prices.append(Price(
                    open=float(row["开盘"]),
                    close=float(row["收盘"]),
                    high=float(row["最高"]),
                    low=float(row["最低"]),
                    volume=int(row["成交量"]),
                    time=str(row["日期"])[:10],
                ))
            return prices

        elif market == "HK":
            # 港股日线数据
            df = _retry_call(
                ak.stock_hk_hist,
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty:
                return []
            prices = []
            for _, row in df.iterrows():
                prices.append(Price(
                    open=float(row["开盘"]),
                    close=float(row["收盘"]),
                    high=float(row["最高"]),
                    low=float(row["最低"]),
                    volume=int(row["成交量"]),
                    time=str(row["日期"])[:10],
                ))
            return prices

        else:
            # 美股日线数据
            df = _retry_call(
                ak.stock_us_hist,
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty:
                return []
            prices = []
            for _, row in df.iterrows():
                prices.append(Price(
                    open=float(row["开盘"]),
                    close=float(row["收盘"]),
                    high=float(row["最高"]),
                    low=float(row["最低"]),
                    volume=int(row["成交量"]),
                    time=str(row["日期"])[:10],
                ))
            return prices

    except Exception as e:
        print(f"  [akshare] 获取价格数据失败 ({ticker}): {e}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """通过 akshare 获取财务指标（主要支持 A 股）。"""
    if not _is_available():
        return []

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market != "CN":
            # akshare 的财务指标主要支持 A 股，其他市场返回空
            return []

        # 获取 A 股个股财务指标
        df = _retry_call(ak.stock_financial_analysis_indicator, symbol=symbol, start_year="2020")
        if df is None or df.empty:
            return []

        metrics_list = []
        for _, row in df.iterrows():
            report_date = str(row.get("日期", ""))[:10]
            if report_date > end_date:
                continue

            metrics = FinancialMetrics(
                ticker=ticker,
                report_period=report_date,
                period=period,
                currency="CNY",
                market_cap=None,
                enterprise_value=None,
                price_to_earnings_ratio=_safe_float(row.get("市盈率")),
                price_to_book_ratio=_safe_float(row.get("市净率")),
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=_safe_float(row.get("销售毛利率(%)")),
                operating_margin=None,
                net_margin=_safe_float(row.get("销售净利率(%)")),
                return_on_equity=_safe_float(row.get("净资产收益率(%)")),
                return_on_assets=_safe_float(row.get("总资产利润率(%)")),
                return_on_invested_capital=None,
                asset_turnover=_safe_float(row.get("总资产周转率(次)")),
                inventory_turnover=_safe_float(row.get("存货周转率(次)")),
                receivables_turnover=_safe_float(row.get("应收账款周转率(次)")),
                days_sales_outstanding=_safe_float(row.get("应收账款周转天数(天)")),
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=_safe_float(row.get("流动比率")),
                quick_ratio=_safe_float(row.get("速动比率")),
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=_safe_float(row.get("资产负债率(%)")),
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=_safe_float(row.get("主营业务收入增长率(%)")),
                earnings_growth=_safe_float(row.get("净利润增长率(%)")),
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=_safe_float(row.get("基本每股收益(元)")),
                book_value_per_share=_safe_float(row.get("每股净资产(元)")),
                free_cash_flow_per_share=None,
            )
            # 百分比字段需要除以 100
            if metrics.gross_margin is not None:
                metrics.gross_margin = metrics.gross_margin / 100.0
            if metrics.net_margin is not None:
                metrics.net_margin = metrics.net_margin / 100.0
            if metrics.return_on_equity is not None:
                metrics.return_on_equity = metrics.return_on_equity / 100.0
            if metrics.return_on_assets is not None:
                metrics.return_on_assets = metrics.return_on_assets / 100.0
            if metrics.debt_to_equity is not None:
                metrics.debt_to_equity = metrics.debt_to_equity / 100.0
            if metrics.revenue_growth is not None:
                metrics.revenue_growth = metrics.revenue_growth / 100.0
            if metrics.earnings_growth is not None:
                metrics.earnings_growth = metrics.earnings_growth / 100.0

            metrics_list.append(metrics)

        return metrics_list[:limit]

    except Exception as e:
        print(f"  [akshare] 获取财务指标失败 ({ticker}): {e}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """通过 akshare 获取财务报表明细项（主要支持 A 股）。"""
    if not _is_available():
        return []

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market != "CN":
            return []

        # 获取利润表、资产负债表、现金流量表
        try:
            income_df = _retry_call(ak.stock_financial_report_sina, stock=symbol, symbol="利润表")
        except Exception:
            income_df = pd.DataFrame()
        try:
            balance_df = _retry_call(ak.stock_financial_report_sina, stock=symbol, symbol="资产负债表")
        except Exception:
            balance_df = pd.DataFrame()
        try:
            cashflow_df = _retry_call(ak.stock_financial_report_sina, stock=symbol, symbol="现金流量表")
        except Exception:
            cashflow_df = pd.DataFrame()

        # akshare 新浪财报的字段映射
        field_mapping = {
            "revenue": "营业收入",
            "total_revenue": "营业总收入",
            "net_income": "净利润",
            "operating_income": "营业利润",
            "gross_profit": "营业利润",
            "cost_of_revenue": "营业成本",
            "interest_expense": "利息支出",
            "income_tax_expense": "所得税费用",
            "research_and_development": "研发费用",
            "selling_general_and_administrative": "销售费用",
            "total_assets": "资产总计",
            "total_liabilities": "负债合计",
            "total_equity": "所有者权益合计",
            "total_debt": "长期借款",
            "long_term_debt": "长期借款",
            "short_term_debt": "短期借款",
            "cash_and_equivalents": "货币资金",
            "inventory": "存货",
            "accounts_receivable": "应收账款",
            "accounts_payable": "应付账款",
            "current_assets": "流动资产合计",
            "current_liabilities": "流动负债合计",
            "operating_cash_flow": "经营活动产生的现金流量净额",
            "capital_expenditure": "购建固定资产、无形资产和其他长期资产支付的现金",
            "dividends_paid": "分配股利、利润或偿付利息支付的现金",
        }

        # 从各报表中提取数据
        results = []
        # 使用利润表的日期作为报告期
        source_df = income_df if not income_df.empty else (balance_df if not balance_df.empty else cashflow_df)
        if source_df.empty:
            return []

        date_col = "报告日" if "报告日" in source_df.columns else source_df.columns[0]
        report_dates = source_df[date_col].unique()
        report_dates = sorted([str(d)[:10] for d in report_dates if str(d)[:10] <= end_date], reverse=True)[:limit]

        for report_date in report_dates:
            item_data = {
                "ticker": ticker,
                "report_period": report_date,
                "period": period,
                "currency": "CNY",
            }

            for requested_item in line_items:
                value = None
                cn_name = field_mapping.get(requested_item, requested_item)

                for source in [income_df, balance_df, cashflow_df]:
                    if source.empty:
                        continue
                    d_col = "报告日" if "报告日" in source.columns else source.columns[0]
                    row_match = source[source[d_col].astype(str).str[:10] == report_date]
                    if row_match.empty:
                        continue
                    if cn_name in row_match.columns:
                        val = row_match[cn_name].iloc[0]
                        value = _safe_float(val)
                        if value is not None:
                            break

                item_data[requested_item] = value

            results.append(LineItem(**item_data))

        return results[:limit]

    except Exception as e:
        print(f"  [akshare] 获取财务报表明细失败 ({ticker}): {e}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """通过 akshare 获取内部人交易数据（仅支持 A 股）。"""
    if not _is_available():
        return []

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market != "CN":
            return []

        # A 股股东增减持
        df = _retry_call(ak.stock_inner_trade_xq, symbol=symbol)
        if df is None or df.empty:
            return []

        trades = []
        for _, row in df.iterrows():
            trade_date = str(row.get("变动日期", ""))[:10]
            if not trade_date or trade_date == "":
                continue
            if trade_date > end_date:
                continue
            if start_date and trade_date < start_date:
                continue

            shares = _safe_float(row.get("变动数量(万股)"))
            if shares is not None:
                shares = shares * 10000  # 万股 -> 股

            price = _safe_float(row.get("成交均价"))
            value = None
            if shares is not None and price is not None:
                value = shares * price

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=str(row.get("股东名称", "")),
                name=str(row.get("股东名称", "")),
                title=str(row.get("变动人", "")),
                is_board_director=None,
                transaction_date=trade_date,
                transaction_shares=shares,
                transaction_price_per_share=price,
                transaction_value=value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=str(row.get("变动原因", "")),
                filing_date=trade_date,
            ))

        return trades[:limit]

    except Exception as e:
        print(f"  [akshare] 获取内部人交易数据失败 ({ticker}): {e}")
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """通过 akshare 获取公司新闻（仅支持 A 股）。"""
    if not _is_available():
        return []

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market != "CN":
            return []

        # 个股新闻
        df = _retry_call(ak.stock_news_em, symbol=symbol)
        if df is None or df.empty:
            return []

        articles = []
        for _, row in df.iterrows():
            pub_date = str(row.get("发布时间", ""))[:10]
            if not pub_date:
                continue
            if pub_date > end_date:
                continue
            if start_date and pub_date < start_date:
                continue

            articles.append(CompanyNews(
                ticker=ticker,
                title=str(row.get("新闻标题", "")),
                author="",
                source=str(row.get("新闻来源", "")),
                date=pub_date,
                url=str(row.get("新闻链接", "")),
                sentiment=None,
            ))

        return articles[:limit]

    except Exception as e:
        print(f"  [akshare] 获取公司新闻失败 ({ticker}): {e}")
        return []


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """通过 akshare 获取市值（使用轻量单股接口）。"""
    if not _is_available():
        return None

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market == "CN":
            df = _get_individual_info(symbol)
            market_cap = _safe_float(_get_info_value(df, "总市值"))
            return market_cap
        elif market == "HK":
            # 港股暂用实时行情接口（无轻量替代）
            df = _retry_call(ak.stock_hk_spot_em)
            if df is None or df.empty:
                return None
            row = df[df["代码"] == symbol]
            if row.empty:
                return None
            market_cap = _safe_float(row.iloc[0].get("总市值"))
            return market_cap
        else:
            return None

    except Exception as e:
        print(f"  [akshare] 获取市值失败 ({ticker}): {e}")
        return None


def get_company_name(ticker: str) -> str | None:
    """通过 akshare 获取公司名称（使用轻量单股接口）。"""
    if not _is_available():
        return None

    symbol, market = _convert_ticker_for_ak(ticker)

    try:
        if market == "CN":
            df = _get_individual_info(symbol)
            name = _get_info_value(df, "股票简称")
            return str(name) if name else None
        elif market == "HK":
            df = _retry_call(ak.stock_hk_spot_em)
            if df is None or df.empty:
                return None
            row = df[df["代码"] == symbol]
            if row.empty:
                return None
            return str(row.iloc[0].get("名称", ticker))
        else:
            return None
    except Exception as e:
        print(f"  [akshare] 获取公司名称失败 ({ticker}): {e}")
        return None
