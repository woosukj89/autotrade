#!/usr/bin/env python3
"""
EDGAR → SQLite Pipeline
=======================

Pulls last ~20 years of fundamental data (Revenue, Net Income, Assets, Debt) and
"cash flow to investors" components (Dividends, Buybacks, Equity Issuance, Financing CF)
from the SEC EDGAR XBRL **Company Facts** API, then stores normalized time series in SQLite.

Why Company Facts?
- It's the SEC's standardized JSON built from XBRL filings (10-K/10-Q), so you avoid parsing raw XBRL.
- Completely free; just be polite with a proper User-Agent and light rate limiting.

Tables
------
- companies(cik INTEGER PRIMARY KEY, ticker TEXT, title TEXT)
- facts(
    cik INTEGER, tag TEXT, fy INTEGER, fp TEXT, form TEXT, filed TEXT,
    end TEXT, val REAL, accn TEXT, unit TEXT,
    PRIMARY KEY(cik, tag, fy, fp, end, unit)
  )

Convenience SQL views are created for common tags and a derived "cash_flow_to_investors" metric.

Usage
-----
  pip install -r requirements.txt
  python edgar_to_sqlite.py --db fundamentals.sqlite \
      --tickers-file tickers.txt --max-companies 500 --min-year 2004

Alternatively, pass --all-sec to ingest the whole SEC list (slow):
  python edgar_to_sqlite.py --db fundamentals.sqlite --all-sec --max-companies 3000

Notes
-----
- "Cash flow to investors" is tracked via components:
  * DividendsPaid (payments of dividends)
  * CommonStockRepurchased (repurchases / buybacks)
  * EquityIssuance (proceeds from stock issuance; negative effect on investors)
  * NetCashFromFinancing (net cash provided by/used in financing activities)
  We store the raw components and expose a view that computes:
    cash_flow_to_investors = ABS(DividendsPaid) + ABS(CommonStockRepurchased) - ABS(EquityIssuance)
  (Signs in XBRL vary by company; we take ABS to represent outflow to investors.)

- Tag mapping: the same economic concept can appear under multiple GAAP tags.
  We choose from prioritized lists per concept (e.g., Revenue → ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax']).

"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import time
import pandas as pd
import yfinance as yf
from typing import Dict, Iterable, List, Optional, Tuple

import requests

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"

# --------- polite headers ---------
DEFAULT_UA = os.environ.get("SEC_USER_AGENT", "ValueResearch/1.0 (joshuaj@gmail.com)")
HEADERS = {"User-Agent": DEFAULT_UA, "Accept-Encoding": "gzip, deflate"}

# --------- tag priority maps (choose the first available) ---------
TAG_PRIORITY: Dict[str, List[str]] = {
    # Income Statement
    "Revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenuesNetOfInterestExpense",
        # Additional tags used by some companies (e.g., Apple pre-2018)
        "NetSales",
        "SalesRevenueServicesNet",
        "SalesRevenueGoodsNet",
        "TotalRevenuesAndOtherIncome",
        "RevenueNet",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "ProfitLoss",
        "OperatingIncomeLoss",
    ],
    "GrossProfit": [
        "GrossProfit",
        "GrossProfitLoss",
    ],
    "OperatingIncome": [
        "OperatingIncomeLoss",
        "IncomeFromOperations",
    ],
    "EBITDA": [
        "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    ],
    "RDExpense": [
        "ResearchAndDevelopmentExpense",
    ],
    "SGAExpense": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    # Balance Sheet
    "TotalAssets": ["Assets", "AssetsNet"],
    "TotalDebt": [
        "DebtOutstanding",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebt",
        "DebtCurrent",
        "DebtNoncurrent",
        "LiabilitiesCurrent",
    ],
    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "CashAndEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    # Cash Flow to investors (components)
    "CashFromOperations": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashProvidedByUsedInOperatingActivitiesDirect",
    ],
    "CapitalExpenditures": [
        "CapitalExpenditures",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "DividendsPaid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "DividendsPaid",
        "DividendsCommonStock",
    ],
    "Buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
        "RepurchaseOfCapitalStock",
    ],
    "EquityIssuance": [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfShares",
    ],
    "NetCashFinancing": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ],
    "Depreciation": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
    ],
}

# Acceptable units priority (we prefer USD)
UNIT_PRIORITY = ["USD", "USD / shares", "USD / shares outstanding"]

# --------- helpers ---------

def http_get_json(url: str) -> Optional[dict]:
    for i in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5 * (i + 1))
    return None


def load_sec_tickers() -> List[Tuple[int, str, str]]:
    """Return list of (cik, ticker, title)."""
    js = http_get_json(SEC_COMPANY_TICKERS_URL)
    if not js:
        raise RuntimeError("Failed to fetch SEC company_tickers.json — set SEC_USER_AGENT env and retry.")
    out: List[Tuple[int, str, str]] = []
    for rec in js.values():
        cik = int(rec["cik_str"])  # already int-like
        ticker = rec["ticker"].upper()
        title = rec.get("title", "")
        out.append((cik, ticker, title))
    return out


def choose_unit_series(fact_obj: dict) -> Optional[Tuple[str, List[dict]]]:
    """Select the preferred unit series (e.g., USD) from a CompanyFacts tag object."""
    units = fact_obj.get("units", {})
    if not units:
        return None
    # try preferred units
    for up in UNIT_PRIORITY:
        if up in units and units[up]:
            return up, units[up]
    # else arbitrary first non-empty
    for k, v in units.items():
        if v:
            return k, v
    return None


def extract_tag_series(company_facts: dict, priority_list: List[str], min_year: int) -> List[dict]:
    """Return a cleaned list of observations for the first available tag in priority_list."""
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    for tag in priority_list:
        if tag in facts:
            unit_choice = choose_unit_series(facts[tag])
            if not unit_choice:
                continue
            unit, series = unit_choice
            cleaned = []
            for obs in series:
                # Filter for annual (fy present, fp == 'FY') and year >= min_year
                fy = obs.get("fy")
                fp = obs.get("fp")
                if fy is None or fp is None:
                    continue
                try:
                    fy_int = int(fy)
                except Exception:
                    continue
                if fy_int < min_year:
                    continue
                cleaned.append({
                    "tag": tag,
                    "fy": fy_int,
                    "fp": fp,
                    "form": obs.get("form"),
                    "filed": obs.get("filed"),
                    "end": obs.get("end"),
                    "val": obs.get("val"),
                    "accn": obs.get("accn"),
                    "unit": unit,
                })
            if cleaned:
                return cleaned
    return []


# --------- SQLite ---------

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            cik INTEGER PRIMARY KEY,
            ticker TEXT,
            title TEXT,
            industry TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            cik INTEGER,
            tag TEXT,
            fy INTEGER,
            fp TEXT,
            form TEXT,
            filed TEXT,
            end TEXT,
            val REAL,
            accn TEXT,
            unit TEXT,
            PRIMARY KEY (cik, tag, fy, fp, end, unit)
        );
        """
    )
    conn.commit()

def initialize(conn: sqlite3.Connection):
    print("First dropping all tables")
    conn.execute("DROP TABLE IF EXISTS companies")
    conn.execute("DROP TABLE IF EXISTS facts")
    conn.execute("DROP TABLE IF EXISTS fundamentals")
    conn.commit()


def upsert_companies(conn: sqlite3.Connection, rows: Iterable[Tuple[int, str, str]]) -> None:
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO companies(cik, ticker, title, industry) VALUES (?,?,?,NULL)",
        [(cik, ticker, title) for (cik, ticker, title) in rows],
    )
    conn.commit()

def upsert_industries(conn: sqlite3.Connection, batch_size=50, sleep=0.25):
    cur = conn.cursor()

    rows = cur.execute("SELECT ticker FROM companies WHERE industry IS NULL").fetchall()
    tickers = [row[0] for row in rows]

    print(f"[INFO] Updating industries for {len(tickers)} tickers...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        updates = []
        try:
            tickers_obj = yf.Tickers(" ".join(batch))
            for t in batch:
                try:
                    info = tickers_obj.tickers[t].info
                    industry = info.get("industry")
                    if industry:
                        updates.append((industry, t))
                        print(f"[OK] {t}: {industry}")
                    else:
                        print(f"[WARN] {t}: No industry found")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[ERR] {t}: {e}")
        except Exception as e:
            print(f"[BATCH ERR] {batch}: {e}")
        
        if updates:
            cur.executemany(
                "UPDATE companies SET industry = ? WHERE ticker = ?",
                updates,
            )
            conn.commit()
        
        time.sleep(sleep)  # prevent rate-limiting

def insert_facts(conn: sqlite3.Connection, cik: int, rows: List[dict]) -> None:
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO facts
        (cik, tag, fy, fp, form, filed, end, val, accn, unit)
        VALUES
        (:cik, :tag, :fy, :fp, :form, :filed, :end, :val, :accn, :unit)
        """,
        [{**r, "cik": cik} for r in rows],
    )
    conn.commit()

def build_fundamentals_table(conn: sqlite3.Connection):
    TAG_MAP = {
        "income": [
            "Revenue",
            "NetIncome",
            "GrossProfit",
            "OperatingIncome",
            "EBITDA",
            "RDExpense",
            "SGAExpense",
        ],
        "balance": [
            "TotalAssets",
            "TotalDebt",
            "Equity",
            "CashAndEquivalents",
        ],
        "cashflow": [
            "CashFromOperations",
            "CapitalExpenditures",
            "DividendsPaid",
            "Buybacks",
            "EquityIssuance",
            "NetCashFinancing",
            "Depreciation"
        ]
    }
    # Drop existing
    conn.execute("DROP TABLE IF EXISTS fundamentals")
    # Create clean fundamentals table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fundamentals (
        ticker TEXT,
        statement_type TEXT,
        fy INT,
        date TEXT,
        field TEXT,
        value REAL,
        PRIMARY KEY (ticker, statement_type, date, field)
    )
    """)

    # Map CIK to ticker
    cik_map = dict(conn.execute("SELECT cik, ticker FROM companies").fetchall())

    # Load facts
    df = pd.read_sql_query("""
        SELECT cik, tag, fy, MAX(end) as end, val
        FROM facts
        WHERE form IN ('10-K','10-K/A') AND fp = 'FY'
        GROUP BY cik, tag, fy
    """, conn)

    for stmt_type, tags in TAG_MAP.items():
        stmt_df = df[df["tag"].isin(tags)].copy()
        stmt_df["field"] = stmt_df["tag"]
        stmt_df["ticker"] = stmt_df["cik"].map(cik_map)
        stmt_df["statement_type"] = stmt_type
        stmt_df.rename(columns={"end": "date", "val": "value"}, inplace=True)
        stmt_df = stmt_df.dropna(subset=["ticker"])
        stmt_df = stmt_df[["ticker", "statement_type", "fy", "date", "field", "value"]]
        stmt_df = stmt_df.drop_duplicates(subset=["ticker", "statement_type", "date", "field"])

        stmt_df.to_sql("fundamentals", conn, if_exists="append", index=False)

    conn.commit()


# --------- ingest one company ---------

def ingest_company(conn: sqlite3.Connection, cik: int, min_year: int) -> bool:
    url = SEC_COMPANY_FACTS_URL_TMPL.format(cik=cik)
    js = http_get_json(url)
    if not js:
        return False

    # Extract and insert per concept
    all_rows: List[dict] = []
    for concept, priorities in TAG_PRIORITY.items():
        rows = extract_tag_series(js, priorities, min_year=min_year)
        for r in rows:
            r["tag"] = concept  # map the raw tag to the overall concept
        all_rows.extend(rows)

    insert_facts(conn, cik, all_rows)
    return True


# --------- CLI ---------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest SEC EDGAR Company Facts → SQLite (20y fundamentals + cash to investors)")
    ap.add_argument("--db", required=True, help="SQLite database file path")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers-file", help="Path to a text file with tickers (one per line)")
    group.add_argument("--all-sec", action="store_true", help="Use the entire SEC universe (slow)")
    ap.add_argument("--max-companies", type=int, default=None, help="Limit number of companies to ingest")
    ap.add_argument("--min-year", type=int, default=2004, help="Minimum fiscal year to keep (≈20 years if 2024+)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between SEC API calls (politeness)")
    ap.add_argument("--initialize", type=bool, default=False, help="Whether to drop all tables")
    ap.add_argument("--preprocessed", type=bool, default=True, help="Whether to rewrite preprocessed table")
    return ap.parse_args()


def main() -> None:
    print("hi")
    args = parse_args()
    print(args)
    if DEFAULT_UA == "ValueResearchBot/1.0 (email@domain.com)":
        print("[WARN] Set SEC_USER_AGENT env to a real contact string to be polite to SEC.")

    conn = sqlite3.connect(args.db)
    if args.initialize:
        initialize(conn)
    init_db(conn)

    # Load SEC universe and upsert companies
    sec_list = load_sec_tickers()
    upsert_companies(conn, sec_list)
    upsert_industries(conn)

    # Figure target CIKs
    target_ciks: List[int]
    if args.all_sec:
        target_ciks = [c for c, t, ti in sec_list]
    else:
        # Read user tickers and map → CIKs
        req = set()
        with open(args.tickers_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s:
                    req.add(s)
        # map tickers to cik
        ticker_to_cik = {t.upper(): c for (c, t, _title) in sec_list}
        missing = [t for t in req if t not in ticker_to_cik]
        if missing:
            print(f"[WARN] {len(missing)} tickers not found in SEC list: {', '.join(sorted(missing)[:10])}...")
        target_ciks = [ticker_to_cik[t] for t in req if t in ticker_to_cik]

    # Trim to max_companies
    if args.max_companies is not None:
        target_ciks = target_ciks[: args.max_companies]
    print(f"Ingesting {len(target_ciks)} companies ...")

    ok = 0
    for i, cik in enumerate(target_ciks, 1):
        success = ingest_company(conn, cik=cik, min_year=args.min_year)
        if success:
            ok += 1
        if i % 25 == 0:
            print(f"  {i}/{len(target_ciks)} processed (ok={ok})")
        time.sleep(args.sleep)

    print(f"Done. {ok}/{len(target_ciks)} ingested into {args.db}")

    if args.preprocessed:
        print(f"Re-writing preprocessed fundamentals table ...")
        build_fundamentals_table(conn)
        cur = conn.cursor()
        print(f"Done. First few rows: {cur.execute('SELECT * FROM fundamentals LIMIT 10')}")


if __name__ == "__main__":
    main()
    # conn = sqlite3.connect("fundamentals.sqlite")
    # build_fundamentals_table(conn)
