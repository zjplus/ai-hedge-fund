"""
Tushare Pro 数据源 —— 作为 yfinance 的高质量备份数据源。
主要用于 A 股数据获取，数据质量和稳定性优于免费数据源。
Tushare Pro 需要 token，在 .env 中配置 TUSHARE_TOKEN。

使用方法：
    uv add tushare
    在 .env 中设置: TUSHARE_TOKEN=your-tushare-token
"""

import os
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
    import tushare as ts
except ImportError:
    ts = None

# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

# tushare pro api 实例（延迟初始化）
_pro_api = None


def _get_pro_api():
    """获取 tushare pro api 实例（带 token）。"""
    global _pro_api
    if _pro_api is not None:
        return _pro_api

    if ts is None:
        return None

    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        return None

    ts.set_token(token)
    _pro_api = ts.pro_api()
    return _pro_api


def _is_available() -> bool:
    """检查 tushare 是否可用（已安装且有 token）。"""
    return _get_pro_api() is not None


def _retry_call(func, *args, **kwargs):
    """带重试的 tushare 调用，处理网络断连和限流。"""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_msg = str(e).lower()
            is_retryable = any(kw in err_msg for kw in [
                "connection", "timeout", "timed out", "抱歉",
                "每分钟", "最多访问", "refused", "too many",
            ])
            if is_retryable and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"    [tushare] 请求异常，{delay:.1f}s 后重试 (第 {attempt + 2}/{_MAX_RETRIES} 次)...")
                time.sleep(delay)
            else:
                raise
    raise last_exc


def _convert_ticker_for_ts(ticker: str) -> tuple[str, str]:
    """
    将通用 ticker 格式转换为 tushare 所需格式。
    返回 (ts_code, market)。

    - A股: 600519.SS -> (600519.SH, CN), 000858.SZ -> (000858.SZ, CN)
    - 港股: 0700.HK -> (00700.HK, HK)  （tushare 港股支持有限）
    - 美股: AAPL -> (AAPL, US)  （tushare 不支持美股）
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith(".SS"):
        # 上交所：.SS -> .SH（tushare 用 .SH 表示上交所）
        symbol = ticker_upper.split(".")[0]
        return f"{symbol}.SH", "CN"
    elif ticker_upper.endswith(".SZ"):
        symbol = ticker_upper.split(".")[0]
        return f"{symbol}.SZ", "CN"
    elif ticker_upper.endswith(".HK"):
        symbol = ticker_upper.split(".")[0].zfill(5)
        return f"{symbol}.HK", "HK"
    else:
        return ticker_upper, "US"


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
    """通过 tushare 获取股票历史价格数据（仅支持 A 股）。"""
    if not _is_available():
        return []

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return []

        pro = _get_pro_api()
        df = _retry_call(
            pro.daily,
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return []

        # tushare daily 返回的列：ts_code, trade_date, open, high, low, close, vol, amount 等
        # 按日期升序排列
        df = df.sort_values("trade_date", ascending=True)

        prices = []
        for _, row in df.iterrows():
            trade_date_str = str(row["trade_date"])
            formatted_date = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"
            prices.append(Price(
                open=float(row["open"]),
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                volume=int(row["vol"] * 100),  # tushare vol 单位是手，转换为股
                time=formatted_date,
            ))
        return prices

    except Exception as e:
        print(f"  [tushare] 获取价格数据失败 ({ticker}): {e}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """通过 tushare 获取财务指标（仅支持 A 股）。"""
    if not _is_available():
        return []

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return []

        pro = _get_pro_api()

        # 获取财务指标数据（fina_indicator 接口）
        df = _retry_call(
            pro.fina_indicator,
            ts_code=ts_code,
            fields=(
                "ts_code,ann_date,end_date,eps,bps,roe,roe_dt,roa,"
                "grossprofit_margin,netprofit_margin,currentratio,quickratio,"
                "debt_to_assets,turn_days,assets_turn,inv_turn,ar_turn,"
                "or_yoy,netprofit_yoy,dt_netprofit_yoy"
            ),
        )
        if df is None or df.empty:
            return []

        # 获取每日指标（市值、PE、PB 等）
        daily_basic_df = None
        try:
            daily_basic_df = _retry_call(
                pro.daily_basic,
                ts_code=ts_code,
                trade_date=end_date.replace("-", ""),
                fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv",
            )
        except Exception:
            pass

        metrics_list = []
        for _, row in df.iterrows():
            report_date_raw = str(row.get("end_date", ""))
            if len(report_date_raw) == 8:
                report_date = f"{report_date_raw[:4]}-{report_date_raw[4:6]}-{report_date_raw[6:8]}"
            else:
                report_date = report_date_raw[:10]

            if report_date > end_date:
                continue

            # 从 daily_basic 取估值指标
            pe_ratio = None
            pb_ratio = None
            ps_ratio = None
            market_cap = None
            if daily_basic_df is not None and not daily_basic_df.empty:
                pe_ratio = _safe_float(daily_basic_df.iloc[0].get("pe_ttm"))
                pb_ratio = _safe_float(daily_basic_df.iloc[0].get("pb"))
                ps_ratio = _safe_float(daily_basic_df.iloc[0].get("ps_ttm"))
                total_mv = _safe_float(daily_basic_df.iloc[0].get("total_mv"))
                if total_mv is not None:
                    market_cap = total_mv * 10000  # tushare total_mv 单位是万元

            # roe/debt_to_assets 是百分比需要转换
            roe_val = _safe_float(row.get("roe"))
            roa_val = _safe_float(row.get("roa"))
            gross_margin_val = _safe_float(row.get("grossprofit_margin"))
            net_margin_val = _safe_float(row.get("netprofit_margin"))
            debt_to_assets_val = _safe_float(row.get("debt_to_assets"))
            revenue_growth_val = _safe_float(row.get("or_yoy"))
            earnings_growth_val = _safe_float(row.get("netprofit_yoy"))

            metrics = FinancialMetrics(
                ticker=ticker,
                report_period=report_date,
                period=period,
                currency="CNY",
                market_cap=market_cap,
                enterprise_value=None,
                price_to_earnings_ratio=pe_ratio,
                price_to_book_ratio=pb_ratio,
                price_to_sales_ratio=ps_ratio,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=gross_margin_val / 100.0 if gross_margin_val is not None else None,
                operating_margin=None,
                net_margin=net_margin_val / 100.0 if net_margin_val is not None else None,
                return_on_equity=roe_val / 100.0 if roe_val is not None else None,
                return_on_assets=roa_val / 100.0 if roa_val is not None else None,
                return_on_invested_capital=None,
                asset_turnover=_safe_float(row.get("assets_turn")),
                inventory_turnover=_safe_float(row.get("inv_turn")),
                receivables_turnover=_safe_float(row.get("ar_turn")),
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=_safe_float(row.get("currentratio")),
                quick_ratio=_safe_float(row.get("quickratio")),
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=None,
                debt_to_assets=debt_to_assets_val / 100.0 if debt_to_assets_val is not None else None,
                interest_coverage=None,
                revenue_growth=revenue_growth_val / 100.0 if revenue_growth_val is not None else None,
                earnings_growth=earnings_growth_val / 100.0 if earnings_growth_val is not None else None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=_safe_float(row.get("eps")),
                book_value_per_share=_safe_float(row.get("bps")),
                free_cash_flow_per_share=None,
            )

            metrics_list.append(metrics)

        return metrics_list[:limit]

    except Exception as e:
        print(f"  [tushare] 获取财务指标失败 ({ticker}): {e}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """通过 tushare 获取财务报表明细项（仅支持 A 股）。"""
    if not _is_available():
        return []

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return []

        pro = _get_pro_api()

        # 获取利润表
        try:
            income_df = _retry_call(
                pro.income,
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "revenue,total_revenue,oper_cost,sell_exp,admin_exp,rd_exp,"
                    "int_exp,income_tax,n_income,operate_profit,ebit,ebitda"
                ),
            )
        except Exception:
            income_df = pd.DataFrame()

        # 获取资产负债表
        try:
            balance_df = _retry_call(
                pro.balancesheet,
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "total_assets,total_liab,total_hldr_eqy_inc_min,"
                    "lt_borr,st_borr,money_cap,inventories,accounts_receiv,"
                    "acct_payable,total_cur_assets,total_cur_liab,total_share"
                ),
            )
        except Exception:
            balance_df = pd.DataFrame()

        # 获取现金流量表
        try:
            cashflow_df = _retry_call(
                pro.cashflow,
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "n_cashflow_act,c_pay_acq_const_fiam,div_paym"
                ),
            )
        except Exception:
            cashflow_df = pd.DataFrame()

        # tushare 字段映射到通用字段名
        field_mapping = {
            # 利润表
            "revenue": "revenue",
            "total_revenue": "total_revenue",
            "net_income": "n_income",
            "operating_income": "operate_profit",
            "gross_profit": "operate_profit",
            "cost_of_revenue": "oper_cost",
            "interest_expense": "int_exp",
            "income_tax_expense": "income_tax",
            "research_and_development": "rd_exp",
            "selling_general_and_administrative": "sell_exp",
            "ebitda": "ebitda",
            "ebit": "ebit",
            # 资产负债表
            "total_assets": "total_assets",
            "total_liabilities": "total_liab",
            "total_equity": "total_hldr_eqy_inc_min",
            "total_debt": "lt_borr",
            "long_term_debt": "lt_borr",
            "short_term_debt": "st_borr",
            "cash_and_equivalents": "money_cap",
            "inventory": "inventories",
            "accounts_receivable": "accounts_receiv",
            "accounts_payable": "acct_payable",
            "current_assets": "total_cur_assets",
            "current_liabilities": "total_cur_liab",
            "shares_outstanding": "total_share",
            # 现金流量表
            "operating_cash_flow": "n_cashflow_act",
            "capital_expenditure": "c_pay_acq_const_fiam",
            "dividends_paid": "div_paym",
        }

        # 合并所有报表的报告期
        all_dates = set()
        for df_src in [income_df, balance_df, cashflow_df]:
            if not df_src.empty and "end_date" in df_src.columns:
                for d in df_src["end_date"].unique():
                    d_str = str(d)
                    if len(d_str) == 8:
                        formatted = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
                    else:
                        formatted = d_str[:10]
                    if formatted <= end_date:
                        all_dates.add(formatted)

        report_dates = sorted(all_dates, reverse=True)[:limit]

        results = []
        for report_date in report_dates:
            # 反转为 tushare 格式
            ts_date = report_date.replace("-", "")

            item_data = {
                "ticker": ticker,
                "report_period": report_date,
                "period": period,
                "currency": "CNY",
            }

            for requested_item in line_items:
                value = None
                ts_field = field_mapping.get(requested_item)
                if ts_field is None:
                    item_data[requested_item] = None
                    continue

                # 在各个报表中查找
                for df_src in [income_df, balance_df, cashflow_df]:
                    if df_src.empty or "end_date" not in df_src.columns:
                        continue
                    row_match = df_src[df_src["end_date"].astype(str) == ts_date]
                    if row_match.empty:
                        continue
                    if ts_field in row_match.columns:
                        val = row_match[ts_field].iloc[0]
                        value = _safe_float(val)
                        if value is not None:
                            # tushare 财务数据单位是元，不需要额外转换
                            break

                item_data[requested_item] = value

            results.append(LineItem(**item_data))

        return results[:limit]

    except Exception as e:
        print(f"  [tushare] 获取财务报表明细失败 ({ticker}): {e}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """通过 tushare 获取股东增减持数据（仅支持 A 股）。"""
    if not _is_available():
        return []

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return []

        pro = _get_pro_api()
        # stk_holdertrade: 股东增减持
        df = _retry_call(
            pro.stk_holdertrade,
            ts_code=ts_code,
            start_date=start_date.replace("-", "") if start_date else None,
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return []

        trades = []
        for _, row in df.iterrows():
            trade_date_raw = str(row.get("ann_date", ""))
            if len(trade_date_raw) == 8:
                trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:8]}"
            else:
                trade_date = trade_date_raw[:10]

            if not trade_date or trade_date > end_date:
                continue
            if start_date and trade_date < start_date:
                continue

            shares = _safe_float(row.get("vol"))  # 变动数量（万股）
            if shares is not None:
                shares = shares * 10000  # 万股 -> 股

            price = _safe_float(row.get("price"))
            value = None
            if shares is not None and price is not None:
                value = abs(shares * price)

            after_shares = _safe_float(row.get("after_vol"))
            if after_shares is not None:
                after_shares = after_shares * 10000

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=str(row.get("holder_name", "")),
                name=str(row.get("holder_name", "")),
                title=str(row.get("holder_type", "")),
                is_board_director=None,
                transaction_date=trade_date,
                transaction_shares=shares,
                transaction_price_per_share=price,
                transaction_value=value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=after_shares,
                security_title=str(row.get("in_de", "")),  # 增减持方向
                filing_date=trade_date,
            ))

        return trades[:limit]

    except Exception as e:
        print(f"  [tushare] 获取股东增减持数据失败 ({ticker}): {e}")
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """通过 tushare 获取公司公告/新闻（仅支持 A 股）。
    注意：tushare 的新闻接口权限要求较高，可能不可用。"""
    if not _is_available():
        return []

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return []

        pro = _get_pro_api()
        # 尝试获取公告数据
        df = _retry_call(
            pro.anns,
            ts_code=ts_code,
            start_date=start_date.replace("-", "") if start_date else None,
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return []

        articles = []
        for _, row in df.iterrows():
            pub_date_raw = str(row.get("ann_date", ""))
            if len(pub_date_raw) == 8:
                pub_date = f"{pub_date_raw[:4]}-{pub_date_raw[4:6]}-{pub_date_raw[6:8]}"
            else:
                pub_date = pub_date_raw[:10]

            if not pub_date or pub_date > end_date:
                continue
            if start_date and pub_date < start_date:
                continue

            articles.append(CompanyNews(
                ticker=ticker,
                title=str(row.get("title", "")),
                author="",
                source="tushare/公告",
                date=pub_date,
                url=str(row.get("url", "")),
                sentiment=None,
            ))

        return articles[:limit]

    except Exception as e:
        print(f"  [tushare] 获取公司新闻失败 ({ticker}): {e}")
        return []


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """通过 tushare 获取市值（使用 daily_basic 接口）。"""
    if not _is_available():
        return None

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return None

        pro = _get_pro_api()
        df = _retry_call(
            pro.daily_basic,
            ts_code=ts_code,
            trade_date=end_date.replace("-", ""),
            fields="ts_code,trade_date,total_mv",
        )

        if df is None or df.empty:
            # 如果指定日期没有数据，尝试往前查找最近的交易日
            start_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=10)
            df = _retry_call(
                pro.daily_basic,
                ts_code=ts_code,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_date.replace("-", ""),
                fields="ts_code,trade_date,total_mv",
            )
            if df is None or df.empty:
                return None

        total_mv = _safe_float(df.iloc[0].get("total_mv"))
        if total_mv is not None:
            return total_mv * 10000  # tushare total_mv 单位是万元，转换为元
        return None

    except Exception as e:
        print(f"  [tushare] 获取市值失败 ({ticker}): {e}")
        return None


def get_company_name(ticker: str) -> str | None:
    """通过 tushare 获取公司名称（使用 stock_basic 接口）。"""
    if not _is_available():
        return None

    ts_code, market = _convert_ticker_for_ts(ticker)

    try:
        if market != "CN":
            return None

        pro = _get_pro_api()
        df = _retry_call(
            pro.stock_basic,
            ts_code=ts_code,
            fields="ts_code,name,fullname",
        )
        if df is None or df.empty:
            return None

        name = df.iloc[0].get("name")
        return str(name) if name else None

    except Exception as e:
        print(f"  [tushare] 获取公司名称失败 ({ticker}): {e}")
        return None
