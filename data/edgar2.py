import requests
import pandas as pd
import sqlite3
import time
import io  # <--- Added this import
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# CRITICAL: SEC and Wikipedia require a User-Agent.
# Replace with your actual email to be polite/compliant.
HEADERS = {'User-Agent': 'joshuajang89@gmail.com'}

DB_NAME = "sp500_financials.db"
YEARS_BACK = 20

# Mapping 'Friendly Names' to US-GAAP XBRL Tags
TAG_MAPPING = {
    # Income Statement
    'Revenue': ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
    'NetIncome': ['NetIncomeLoss', 'ProfitLoss'],
    'OpIncome': ['OperatingIncomeLoss'],
    
    # Cash Flow
    'CashFromOps': ['NetCashProvidedByUsedInOperatingActivities'],
    'CashUsedInvesting': ['NetCashProvidedByUsedInInvestingActivities'],
    'CashUsedFinancing': ['NetCashProvidedByUsedInFinancingActivities'],
    
    # Balance Sheet
    'TotalAssets': ['Assets'],
    'TotalLiabilities': ['Liabilities'],
    'TotalEquity': ['StockholdersEquity', 'PartnersCapital'],
    'CashAndEquiv': ['CashAndCashEquivalentsAtCarryingValue'],
    'Debt': ['LongTermDebt', 'LongTermDebtNoncurrent']
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def setup_database():
    """Creates the SQLite database and table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financials (
            ticker TEXT,
            cik TEXT,
            year INTEGER,
            metric TEXT,
            value REAL,
            form TEXT,
            filed_date TEXT,
            PRIMARY KEY (ticker, year, metric)
        )
    ''')
    conn.commit()
    return conn

def get_sp500_tickers():
    """Scrapes Wikipedia for the current S&P 500 list using headers."""
    print("Fetching S&P 500 list...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        # FIXED: Use requests with headers to avoid 403 Forbidden
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status() # Check for errors
        
        # Use io.StringIO to treat the string like a file for pandas
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return []

def get_sec_cik_map():
    """Downloads the official Ticker -> CIK map from the SEC."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    
    # Convert dictionary of dictionaries to a clean lookup dict
    cik_map = {}
    for entry in data.values():
        cik_map[entry['ticker']] = str(entry['cik_str']).zfill(10) # CIK must be 10 digits
    return cik_map

def fetch_company_facts(cik):
    """Fetches all XBRL data for a specific CIK."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"Failed to fetch CIK {cik}: Status {r.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching CIK {cik}: {e}")
        return None

def extract_data(ticker, cik, json_data, conn):
    """Parses the massive JSON blob and saves relevant data to DB."""
    if 'facts' not in json_data or 'us-gaap' not in json_data['facts']:
        return

    us_gaap = json_data['facts']['us-gaap']
    records = []
    current_year = datetime.now().year
    cutoff_year = current_year - YEARS_BACK

    for metric_name, tags in TAG_MAPPING.items():
        found_tag = False
        for tag in tags:
            if tag in us_gaap:
                units = us_gaap[tag]['units']
                unit_key = list(units.keys())[0] 
                
                for entry in units[unit_key]:
                    # Filter for 10-K specifically
                    if entry.get('form') == '10-K':
                        try:
                            end_date = entry.get('end')
                            fy_year = int(end_date.split('-')[0])
                            
                            if fy_year >= cutoff_year:
                                records.append((
                                    ticker,
                                    cik,
                                    fy_year,
                                    metric_name,
                                    entry.get('val'),
                                    '10-K',
                                    entry.get('filed')
                                ))
                        except:
                            continue
                found_tag = True
                break 
        
    if records:
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO financials (ticker, cik, year, metric, value, form, filed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        conn.commit()
        print(f"Saved {len(records)} records for {ticker}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    conn = setup_database()
    
    # 1. Get Tickers
    tickers = get_sp500_tickers()
    
    # CRITICAL: For testing, uncomment the next line to stop it from running all 500 stocks immediately!
    # tickers = tickers[:5] 
    
    if not tickers:
        print("No tickers found. Exiting.")
        return

    # 2. Get CIK Map
    cik_map = get_sec_cik_map()
    
    print(f"Found {len(tickers)} tickers. Starting processing...")
    
    for ticker in tickers:
        clean_ticker = ticker.replace('.', '-') # Handle BRK.B -> BRK-B
        
        if clean_ticker in cik_map:
            cik = cik_map[clean_ticker]
            print(f"Processing {ticker} (CIK: {cik})...")
            
            fact_data = fetch_company_facts(cik)
            
            if fact_data:
                extract_data(ticker, cik, fact_data, conn)
            
            # Sleep to respect SEC rate limits (max 10 req/sec)
            time.sleep(0.15) 
            
        else:
            print(f"Could not find CIK for {ticker}")

    conn.close()
    print("Done! Data saved to", DB_NAME)

if __name__ == "__main__":
    main()