-- Gold Mart: fct_financial_ratios
-- Key financial profitability, liquidity, and growth ratios materialized as external Parquet in MinIO Gold bucket.
-- Employs NULLIF to safeguard against division by zero and strict calendar-year lag alignment for YoY growth.

{{ config(
    materialized = 'external',
    location = 's3://sec-gold/marts/fct_financial_ratios.parquet'
) }}

WITH statements AS (
    SELECT
        statement_key,
        cik,
        ticker,
        fiscal_year,
        fiscal_period,
        period_type,
        start_date,
        end_date,
        revenues,
        operating_income,
        net_income,
        gross_profit,
        current_assets,
        current_liabilities,
        total_assets,
        total_liabilities,
        stockholders_equity_parent,
        -- Prior period lookup for exact YoY comparison (same fiscal period, prior fiscal year)
        LAG(revenues, 1) OVER (
            PARTITION BY ticker, fiscal_period
            ORDER BY fiscal_year ASC
        ) AS prior_year_revenues,
        LAG(fiscal_year, 1) OVER (
            PARTITION BY ticker, fiscal_period
            ORDER BY fiscal_year ASC
        ) AS prior_fiscal_year
    FROM {{ ref('fct_financial_statements') }}
)

SELECT
    statement_key AS ratio_key,
    cik,
    ticker,
    fiscal_year,
    fiscal_period,
    period_type,
    start_date,
    end_date,
    
    -- Profitability Ratios
    ROUND(operating_income / NULLIF(revenues, 0), 4) AS operating_margin,
    ROUND(net_income / NULLIF(revenues, 0), 4) AS net_margin,
    ROUND(gross_profit / NULLIF(revenues, 0), 4) AS gross_margin,
    
    -- Liquidity & Solvency Ratios
    ROUND(current_assets / NULLIF(current_liabilities, 0), 4) AS current_ratio,
    ROUND(total_liabilities / NULLIF(stockholders_equity_parent, 0), 4) AS debt_to_equity,
    
    -- Year-over-Year Growth (valid only when strictly consecutive fiscal years)
    CASE
        WHEN prior_fiscal_year = fiscal_year - 1 AND prior_year_revenues IS NOT NULL
        THEN ROUND((revenues - prior_year_revenues) / NULLIF(prior_year_revenues, 0), 4)
        ELSE NULL
    END AS revenue_growth_yoy
FROM statements
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
