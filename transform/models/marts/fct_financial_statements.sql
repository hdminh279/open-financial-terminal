-- Gold Mart: fct_financial_statements
-- Canonical financial statements materialized as external Parquet in MinIO Gold bucket.
-- Preserves all 10 Group A core financial metrics across standard fiscal periods.

{{ config(
    materialized = 'external',
    location = 's3://sec-gold/marts/fct_financial_statements.parquet'
) }}

WITH source_data AS (
    SELECT *
    FROM {{ ref('int_quarterly_standardized') }}
)

SELECT
    -- Primary Surrogate Key
    md5(ticker || '-' || CAST(fiscal_year AS VARCHAR) || '-' || fiscal_period) AS statement_key,
    cik,
    ticker,
    fiscal_year,
    fiscal_period,
    period_type,
    start_date,
    end_date,
    
    -- Income Statement line items (Duration)
    revenues,
    operating_income,
    net_income,
    gross_profit,
    eps_diluted,
    
    -- Balance Sheet line items (Instant snapshots)
    cash_and_cash_equivalents,
    current_assets,
    current_liabilities,
    total_assets,
    total_liabilities,
    stockholders_equity_parent,
    stockholders_equity_total,
    
    -- Provenance & Audit Metadata
    is_derived,
    derivation_method,
    source_accession_number,
    source_filed_date
FROM source_data
ORDER BY
    ticker ASC,
    fiscal_year ASC,
    CASE fiscal_period
        WHEN 'Q1' THEN 1
        WHEN 'Q2' THEN 2
        WHEN 'Q3' THEN 3
        WHEN 'Q4' THEN 4
        WHEN 'FY' THEN 5
        ELSE 6
    END ASC
