-- Intermediate model: int_financial_pivoted
-- Pivots canonical financial metrics from long (EAV) format to wide tabular format per filing period
-- across both US (SEC EDGAR) and Vietnamese (VN30) markets.

WITH sec_facts AS (
    SELECT
        cik,
        ticker,
        market,
        currency,
        fiscal_year,
        fiscal_period,
        duration_scope,
        period_type,
        form,
        filed_date,
        accession_number,
        start_date,
        end_date,
        canonical_metric,
        value
    FROM {{ ref('stg_sec_facts') }}
    WHERE is_as_reported = TRUE
),

vn_facts AS (
    SELECT
        cik,
        ticker,
        market,
        currency,
        fiscal_year,
        fiscal_period,
        duration_scope,
        period_type,
        form,
        filed_date,
        accession_number,
        start_date,
        end_date,
        canonical_metric,
        value
    FROM {{ ref('stg_vn_facts') }}
    WHERE is_as_reported = TRUE
),

all_facts AS (
    SELECT * FROM sec_facts
    UNION ALL
    SELECT * FROM vn_facts
)

SELECT
    cik,
    ticker,
    market,
    currency,
    fiscal_year,
    fiscal_period,
    duration_scope,
    period_type,
    MAX(form) AS form,
    MAX(filed_date) AS filed_date,
    ARG_MAX(accession_number, filed_date) AS accession_number,
    MIN(start_date) AS start_date,
    MAX(end_date) AS end_date,
    -- Core Income Statement items (Duration)
    MAX(CASE WHEN canonical_metric IN ('revenues', 'gross_revenues') THEN value END) AS revenues,
    MAX(CASE WHEN canonical_metric = 'cost_of_goods_sold' THEN value END) AS cost_of_goods_sold,
    MAX(CASE WHEN canonical_metric = 'operating_income' THEN value END) AS operating_income,
    MAX(CASE WHEN canonical_metric IN ('net_income', 'net_income_total') THEN value END) AS net_income,
    MAX(CASE WHEN canonical_metric IN ('gross_profit', 'gross_profit_alt') THEN value END) AS gross_profit,
    MAX(CASE WHEN canonical_metric IN ('eps_diluted', 'basic_eps', 'basic_eps_alt') THEN value END) AS eps_diluted,
    -- Banking Income Statement items (Duration)
    MAX(CASE WHEN canonical_metric = 'net_interest_income' THEN value END) AS net_interest_income,
    MAX(CASE WHEN canonical_metric = 'total_operating_income' THEN value END) AS total_operating_income,
    -- Core Balance Sheet line items (Instant)
    MAX(CASE WHEN canonical_metric = 'cash_and_cash_equivalents' THEN value END) AS cash_and_cash_equivalents,
    MAX(CASE WHEN canonical_metric = 'current_assets' THEN value END) AS current_assets,
    MAX(CASE WHEN canonical_metric = 'current_liabilities' THEN value END) AS current_liabilities,
    MAX(CASE WHEN canonical_metric = 'total_assets' THEN value END) AS total_assets,
    MAX(CASE WHEN canonical_metric = 'total_liabilities' THEN value END) AS total_liabilities,
    MAX(CASE WHEN canonical_metric = 'stockholders_equity_parent' THEN value END) AS stockholders_equity_parent,
    MAX(CASE WHEN canonical_metric IN ('stockholders_equity_total', 'total_equity') THEN value END) AS stockholders_equity_total,
    -- Banking Balance Sheet line items (Instant)
    MAX(CASE WHEN canonical_metric = 'customer_loans' THEN value END) AS customer_loans,
    MAX(CASE WHEN canonical_metric = 'customer_deposits' THEN value END) AS customer_deposits,
    -- Cash Flow line items (Duration)
    MAX(CASE WHEN canonical_metric = 'operating_cash_flow' THEN value END) AS operating_cash_flow
FROM all_facts
GROUP BY
    cik,
    ticker,
    market,
    currency,
    fiscal_year,
    fiscal_period,
    duration_scope,
    period_type
