"""
Point-in-time fundamentals via SEC EDGAR - the fix for two problems found
this session:

1. Reliability: HighBetaGrowthStrategy fetches fundamentals from
   yfinance's `.info` endpoint, which silently fails entire fetch batches
   under load (see PROGRESS.md Iteration 21). SEC EDGAR's Company Facts
   API is the official, free, government source for financial-statement
   data - no scraping, predictable HTTP errors instead of silent partial
   failures.

2. Lookahead bias: HighBetaGrowthStrategy calls
   self._yahoo_provider.get_fundamentals_batch() directly, with no date
   parameter - every backtest this session (2005-2025) has been scoring
   historical stock picks using TODAY's (2026) fundamentals, not what was
   actually known at the time. backtest.py already has a proper
   point-in-time query path (_get_fundamentals(ticker, date, ...), filters
   `date <= simulated_date`) reading a `fundamentals` table - it was just
   never populated with real data or wired into the strategy.

data/edgar.py already has a working SEC ingestion pipeline (ticker->CIK
mapping, company-facts fetch, tag-priority mapping covering Revenue,
NetIncome, GrossProfit, OperatingIncome, TotalAssets, TotalDebt, Equity,
CashFromOperations, CapitalExpenditures, DividendsPaid) that builds this
exact `fundamentals` table schema - it was built but never actually run
with real data, and its build_fundamentals_table() has two correctness
bugs fixed here: (a) it used the fiscal PERIOD END date as the
"knowable as of" date, not the actual SEC FILING date - a company's
FY2010 results aren't public until the 10-K is filed, typically 60-90
days after fiscal year end; (b) its GROUP BY MAX(end) selection didn't
reliably pair the aggregated end-date with the correct val for a given
(cik, tag, fy). Both fixed in build_pointintime_fundamentals_table() below.
"""
import os
import sys
import sqlite3
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
import edgar as edgar_mod  # data/edgar.py

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'edgar_fundamentals.sqlite')
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), 'edgar_universe.txt')


def extract_tag_series_merged(company_facts: dict, priority_list, min_year: int):
    """Fixes THREE real bugs found in data/edgar.py::extract_tag_series()
    and the fy-keyed dedup this function originally used:

    1. It returned as soon as it found ANY tag in priority_list with data,
       and only ever used that ONE tag's series - but companies change
       their exact XBRL tag over time (e.g. many switched revenue tags
       around the 2018 ASC 606 accounting standard update), silently
       dropping every year reported under a different-but-equally-valid
       tag. Fixed by searching ALL tags in priority_list.

    2. SEC's `fy` field labels which FILING (accession) an observation
       came from, not which fiscal period the VALUE describes. A single
       10-K includes several prior years' comparative figures AND
       quarterly breakdowns, all stamped with the filing's own `fy` - so
       grouping by `fy` silently mixes annual, quarterly, and restated-
       prior-year values under one label, picking an arbitrary one.
       Fixed by (a) filtering to only genuine annual-period observations
       (start->end spans 350-380 days) and (b) keying by the actual `end`
       (fiscal period end) date instead of the misleading `fy` label.

    3. The same (ticker, period) value reappears in every LATER filing
       too, as a comparative prior-year figure - e.g. FY2018's revenue
       shows up in the FY2018, FY2019, AND FY2020 10-Ks. For point-in-
       time correctness we want the EARLIEST filing that disclosed a
       given period (that's when it actually became knowable), not the
       latest - fixed by sorting each period's candidates by `filed`
       ascending and keeping the first.
    """
    facts = company_facts.get('facts', {}).get('us-gaap', {})
    candidates = []
    for tag in priority_list:
        if tag not in facts:
            continue
        unit_choice = edgar_mod.choose_unit_series(facts[tag])
        if not unit_choice:
            continue
        unit, series = unit_choice
        for obs in series:
            if obs.get('form') not in ('10-K', '10-K/A') or obs.get('fp') != 'FY':
                continue
            start, end, filed, val = obs.get('start'), obs.get('end'), obs.get('filed'), obs.get('val')
            if not (end and filed) or val is None:
                continue
            if start:
                # Duration concept (income statement / cash flow): only
                # keep genuine annual spans, not quarterly/partial fragments.
                try:
                    span_days = (pd_to_date(end) - pd_to_date(start)).days
                except Exception:
                    continue
                if not (350 <= span_days <= 380):
                    continue
            # else: instant concept (balance sheet - Equity, TotalAssets,
            # TotalDebt, CashAndEquivalents) - a single as-of-`end` snapshot,
            # no start/span to filter on.
            end_year = pd_to_date(end).year
            if end_year < min_year:
                continue
            candidates.append({
                'tag': tag, 'end': end, 'filed': filed, 'val': val,
                'accn': obs.get('accn'), 'unit': unit, 'fy': end_year, 'fp': 'FY',
                'form': obs.get('form'),
            })

    # For each distinct fiscal period (`end` date), keep only the
    # earliest-filed disclosure of it - that's the true point-in-time date.
    by_end = {}
    for c in candidates:
        key = c['end']
        if key not in by_end or c['filed'] < by_end[key]['filed']:
            by_end[key] = c
    return list(by_end.values())


def pd_to_date(s: str):
    import datetime as _dt
    return _dt.datetime.strptime(s, '%Y-%m-%d')


def upsert_sectors(conn: sqlite3.Connection, tickers: list, batch_size=50, sleep=0.25):
    """data/edgar.py::upsert_industries() only fetches the fine-grained
    `industry` field (e.g. "Consumer Electronics"), not the broad `sector`
    field (e.g. "Technology") that SECTOR_SCORES/DEFENSIVE_SECTOR_SCORES
    actually key on - this fetches sector too, into its own column. Still
    a one-time yfinance `.info` call (sector classification is structural/
    slow-changing, not point-in-time sensitive like financial ratios), so
    the same reliability risk as any single `.info` batch applies here,
    but only once, not per backtest run.
    """
    import yfinance as yf
    cur = conn.cursor()
    existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(companies)").fetchall()]
    if 'sector' not in existing_cols:
        conn.execute("ALTER TABLE companies ADD COLUMN sector TEXT")
        conn.commit()
    already = {r[0] for r in cur.execute(
        f"SELECT ticker FROM companies WHERE sector IS NOT NULL AND ticker IN ({','.join('?'*len(tickers))})",
        tickers).fetchall()}
    tickers = [t for t in tickers if t not in already]
    print(f"[INFO] Fetching sector for {len(tickers)} tickers (target universe only)...")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        updates = []
        try:
            tickers_obj = yf.Tickers(" ".join(batch))
            for t in batch:
                try:
                    info = tickers_obj.tickers[t].info
                    sector = info.get("sector")
                    if sector:
                        updates.append((sector, t))
                except Exception as e:
                    print(f"[ERR] {t}: {e}")
        except Exception as e:
            print(f"[BATCH ERR] {batch}: {e}")
        if updates:
            cur.executemany("UPDATE companies SET sector = ? WHERE ticker = ?", updates)
            conn.commit()
        time.sleep(sleep)
    n_missing = cur.execute("SELECT COUNT(*) FROM companies WHERE sector IS NULL AND ticker IN (%s)" %
                             ",".join("?" * len(tickers)), tickers).fetchone()[0]
    print(f"[INFO] sector fetch done, {n_missing}/{len(tickers)} still missing (will retry-fetch live if needed)")


def ingest_company_fixed(conn: sqlite3.Connection, cik: int, min_year: int) -> bool:
    """Corrected version of data/edgar.py::ingest_company() using
    extract_tag_series_merged() instead of the buggy single-tag pick."""
    url = edgar_mod.SEC_COMPANY_FACTS_URL_TMPL.format(cik=cik)
    js = edgar_mod.http_get_json(url)
    if not js:
        return False
    all_rows = []
    for concept, priorities in edgar_mod.TAG_PRIORITY.items():
        rows = extract_tag_series_merged(js, priorities, min_year=min_year)
        for r in rows:
            r['tag'] = concept
        all_rows.extend(rows)
    edgar_mod.insert_facts(conn, cik, all_rows)
    return True


def build_pointintime_fundamentals_table(conn: sqlite3.Connection):
    """Corrected version of data/edgar.py's build_fundamentals_table():
    uses the actual SEC `filed` date (not fiscal period `end`) as the
    date a fact becomes knowable, and picks the val from the LATEST
    filing per (cik, tag, fy) correctly instead of an unpaired MAX(end).
    """
    conn.execute("DROP TABLE IF EXISTS fundamentals")
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

    cik_map = dict(conn.execute("SELECT cik, ticker FROM companies").fetchall())

    import pandas as pd
    # NOTE: ingest_company() already resolves each observation to its
    # friendly concept name before storing (`r["tag"] = concept`, see
    # data/edgar.py::ingest_company) - `facts.tag` holds values like
    # 'Revenue'/'NetIncome' directly, NOT the raw XBRL tag names in
    # TAG_PRIORITY's lists. So no isin(raw_tag_list) matching is needed -
    # each row's `tag` IS the statement_type already.
    #
    # extract_tag_series_merged() already ensures one row per (cik, tag,
    # end-period) using the earliest disclosing filing - this is a
    # defensive backstop in case of any edge-case duplicates, kept
    # consistent with that same "earliest filed wins" point-in-time rule.
    df = pd.read_sql_query("""
        SELECT cik, tag, fy, filed, end, val,
               ROW_NUMBER() OVER (PARTITION BY cik, tag, fy ORDER BY filed ASC) as rn
        FROM facts
        WHERE form IN ('10-K','10-K/A') AND fp = 'FY' AND filed IS NOT NULL
    """, conn)
    df = df[df['rn'] == 1].drop(columns=['rn'])

    df['field'] = df['tag']
    df['ticker'] = df['cik'].map(cik_map)
    df['statement_type'] = df['tag']
    df.rename(columns={'filed': 'date', 'val': 'value'}, inplace=True)
    df = df.dropna(subset=['ticker'])
    df = df[['ticker', 'statement_type', 'fy', 'date', 'field', 'value']]
    df = df.drop_duplicates(subset=['ticker', 'statement_type', 'date', 'field'])
    df.to_sql('fundamentals', conn, if_exists='append', index=False)

    conn.commit()


def main():
    with open(UNIVERSE_FILE) as f:
        tickers = sorted({line.strip().upper() for line in f if line.strip()})
    print(f"Ingesting {len(tickers)} tickers from {UNIVERSE_FILE}")

    conn = sqlite3.connect(DB_PATH)
    edgar_mod.init_db(conn)

    sec_list = edgar_mod.load_sec_tickers()
    edgar_mod.upsert_companies(conn, sec_list)

    ticker_to_cik = {t.upper(): c for (c, t, _title) in sec_list}
    missing = [t for t in tickers if t not in ticker_to_cik]
    if missing:
        print(f"[WARN] {len(missing)} tickers not in SEC list: {missing[:20]}")
    target_ciks = [ticker_to_cik[t] for t in tickers if t in ticker_to_cik]
    print(f"Resolved {len(target_ciks)}/{len(tickers)} tickers to CIKs")

    resolved_tickers = [t for t in tickers if t in ticker_to_cik]
    print("Fetching sector (one-time, target universe only, cached permanently)...")
    upsert_sectors(conn, resolved_tickers)

    ok = 0
    for i, cik in enumerate(target_ciks, 1):
        try:
            success = ingest_company_fixed(conn, cik=cik, min_year=2002)
            if success:
                ok += 1
        except Exception as e:
            print(f"[ERR] cik={cik}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(target_ciks)} processed (ok={ok})")
        time.sleep(0.15)

    print(f"Ingestion done: {ok}/{len(target_ciks)} companies")
    print("Building point-in-time fundamentals table (using filing dates, not period-end dates)...")
    build_pointintime_fundamentals_table(conn)

    cur = conn.cursor()
    n_rows = cur.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
    n_tickers = cur.execute("SELECT COUNT(DISTINCT ticker) FROM fundamentals").fetchone()[0]
    print(f"fundamentals table: {n_rows} rows, {n_tickers} distinct tickers")
    conn.close()


if __name__ == '__main__':
    main()
