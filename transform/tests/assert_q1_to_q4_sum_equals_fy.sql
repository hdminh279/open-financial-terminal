-- Singular Test: assert_q1_to_q4_sum_equals_fy
-- Verifies that the sum of discrete quarterly revenues matches the annual reported revenue within 0.1% tolerance.
-- Test passes when this query returns 0 rows.

WITH quarterly_sums AS (
    SELECT
        ticker,
        fiscal_year,
        COUNT(DISTINCT fiscal_period) AS quarter_count,
        SUM(revenues) AS sum_revenues
    FROM {{ ref('fct_financial_statements') }}
    WHERE fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
    GROUP BY ticker, fiscal_year
),
fy_periods AS (
    SELECT
        ticker,
        fiscal_year,
        revenues AS fy_revenues
    FROM {{ ref('fct_financial_statements') }}
    WHERE fiscal_period = 'FY'
)

SELECT
    q.ticker,
    q.fiscal_year,
    q.sum_revenues,
    f.fy_revenues,
    ABS(q.sum_revenues - f.fy_revenues) AS diff_revenues,
    ROUND(ABS(q.sum_revenues - f.fy_revenues) / NULLIF(f.fy_revenues, 0), 4) AS diff_pct
FROM quarterly_sums q
JOIN fy_periods f
  ON q.ticker = f.ticker
 AND q.fiscal_year = f.fiscal_year
WHERE q.quarter_count = 4
  AND f.fy_revenues IS NOT NULL
  -- Flag discrepancy exceeding 0.1% (tolerance for minor penny rounding / tax restatements)
  AND (ROUND(ABS(q.sum_revenues - f.fy_revenues) / NULLIF(f.fy_revenues, 0), 4) > 0.001)
