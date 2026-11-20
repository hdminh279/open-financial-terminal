-- Intermediate model: int_financial_pivoted
-- Pivots 10 Group A canonical metrics from long (EAV) format to wide tabular format per filing period.

WITH stg_facts AS (
    SELECT *
    FROM {{ ref('stg_sec_facts') }}
    WHERE is_as_reported = TRUE
)

SELECT
    cik,
    ticker,
    fiscal_year,
    fiscal_period,
    duration_scope,
    period_type,
    form,
    filed_date,
    accession_number,
    MIN(start_date) AS start_date,
    MAX(end_date) AS end_date,
    -- Income Statement line items (Duration)
    MAX(CASE WHEN canonical_metric = 'revenues' THEN value END) AS revenues,
    MAX(CASE WHEN canonical_metric = 'operating_income' THEN value END) AS operating_income,
    MAX(CASE WHEN canonical_metric = 'net_income' THEN value END) AS net_income,
    MAX(CASE WHEN canonical_metric = 'gross_profit' THEN value END) AS gross_profit,
    MAX(CASE WHEN canonical_metric = 'eps_diluted' THEN value END) AS eps_diluted,
    -- Balance Sheet line items (Instant)
    MAX(CASE WHEN canonical_metric = 'cash_and_cash_equivalents' THEN value END) AS cash_and_cash_equivalents,
    MAX(CASE WHEN canonical_metric = 'current_assets' THEN value END) AS current_assets,
    MAX(CASE WHEN canonical_metric = 'current_liabilities' THEN value END) AS current_liabilities,
    MAX(CASE WHEN canonical_metric = 'total_assets' THEN value END) AS total_assets,
    MAX(CASE WHEN canonical_metric = 'total_liabilities' THEN value END) AS total_liabilities,
    MAX(CASE WHEN canonical_metric = 'stockholders_equity_parent' THEN value END) AS stockholders_equity_parent,
    MAX(CASE WHEN canonical_metric = 'stockholders_equity_total' THEN value END) AS stockholders_equity_total
FROM stg_facts
GROUP BY
    cik,
    ticker,
    fiscal_year,
    fiscal_period,
    duration_scope,
    period_type,
    form,
    filed_date,
    accession_number
