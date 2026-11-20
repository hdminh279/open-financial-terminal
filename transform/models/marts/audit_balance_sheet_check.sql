-- Audit Mart: audit_balance_sheet_check
-- Verifies the fundamental accounting identity: Assets = Liabilities + Stockholders' Equity.
-- Soft audit view: identifies discrepancies without dropping valid filing observations.

{{ config(
    materialized = 'view'
) }}

WITH statements AS (
    SELECT *
    FROM {{ ref('fct_financial_statements') }}
)

SELECT
    statement_key,
    cik,
    ticker,
    fiscal_year,
    fiscal_period,
    period_type,
    end_date AS snapshot_date,
    total_assets,
    total_liabilities,
    stockholders_equity_parent,
    stockholders_equity_total,
    -- Balance sheet discrepancy using consolidated total equity (fallback to parent equity)
    (total_assets - (total_liabilities + COALESCE(stockholders_equity_total, stockholders_equity_parent))) AS discrepancy,
    ROUND(
        ABS(total_assets - (total_liabilities + COALESCE(stockholders_equity_total, stockholders_equity_parent))) 
        / NULLIF(total_assets, 0),
        6
    ) AS discrepancy_pct,
    CASE
        WHEN (total_assets - (total_liabilities + COALESCE(stockholders_equity_total, stockholders_equity_parent))) = 0
        THEN TRUE
        ELSE FALSE
    END AS is_balanced,
    CASE
        WHEN (ROUND(ABS(total_assets - (total_liabilities + COALESCE(stockholders_equity_total, stockholders_equity_parent))) / NULLIF(total_assets, 0), 6) > 0.02)
        THEN TRUE
        ELSE FALSE
    END AS audit_warning
FROM statements
WHERE total_assets IS NOT NULL 
  AND total_liabilities IS NOT NULL 
  AND (stockholders_equity_parent IS NOT NULL OR stockholders_equity_total IS NOT NULL)
