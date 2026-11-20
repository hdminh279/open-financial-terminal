-- Intermediate model: int_quarterly_standardized
-- Unifies discrete quarters (Q1, Q2, Q3), derives Q4 (FY - 9M), attaches instant balance sheet snapshots,
-- and strictly excludes EPS Diluted from duration subtractions.

WITH pivoted AS (
    SELECT *
    FROM {{ ref('int_financial_pivoted') }}
),

-- Point-in-time Balance Sheet snapshots (Instant)
balance_instant AS (
    SELECT
        cik,
        ticker,
        fiscal_year,
        fiscal_period,
        end_date AS snapshot_date,
        cash_and_cash_equivalents,
        current_assets,
        current_liabilities,
        total_assets,
        total_liabilities,
        stockholders_equity_parent,
        stockholders_equity_total
    FROM pivoted
    WHERE period_type = 'instant'
),

-- As-reported discrete quarters (Q1, Q2, Q3) and annual full year (FY)
as_reported_periods AS (
    SELECT
        p.cik,
        p.ticker,
        p.fiscal_year,
        p.fiscal_period,
        CASE WHEN p.fiscal_period = 'FY' THEN 'annual' ELSE 'quarterly' END AS period_type,
        p.start_date,
        p.end_date,
        -- Income Statement items (Duration)
        p.revenues,
        p.operating_income,
        p.net_income,
        p.gross_profit,
        p.eps_diluted,
        -- Balance Sheet items (Instant)
        b.cash_and_cash_equivalents,
        b.current_assets,
        b.current_liabilities,
        b.total_assets,
        b.total_liabilities,
        b.stockholders_equity_parent,
        b.stockholders_equity_total,
        FALSE AS is_derived,
        'as_reported' AS derivation_method,
        p.accession_number AS source_accession_number,
        p.filed_date AS source_filed_date
    FROM pivoted p
    LEFT JOIN balance_instant b
      ON p.cik = b.cik
     AND p.fiscal_year = b.fiscal_year
     AND p.fiscal_period = b.fiscal_period
    WHERE (p.fiscal_period IN ('Q1', 'Q2', 'Q3') AND p.duration_scope = 'discrete_quarter')
       OR (p.fiscal_period = 'FY' AND p.duration_scope = 'annual_12m')
),

-- Derived Q4: FY (annual_12m) minus Q3 (cumulative_9m) via LEFT JOIN
derived_q4 AS (
    SELECT
        fy.cik,
        fy.ticker,
        fy.fiscal_year,
        'Q4' AS fiscal_period,
        'quarterly' AS period_type,
        CASE
            WHEN q3.end_date IS NOT NULL THEN (q3.end_date + INTERVAL 1 DAY)::DATE
            ELSE NULL
        END AS start_date,
        fy.end_date,
        -- Duration subtractions (arithmetically valid metrics only)
        (fy.revenues - q3.revenues) AS revenues,
        (fy.operating_income - q3.operating_income) AS operating_income,
        (fy.net_income - q3.net_income) AS net_income,
        (fy.gross_profit - q3.gross_profit) AS gross_profit,
        -- EPS Diluted is strictly NOT derivable via subtraction (NULL by accounting principle)
        CAST(NULL AS DECIMAL(28, 2)) AS eps_diluted,
        -- Balance Sheet is instant at FY year-end snapshot (never subtracted)
        b.cash_and_cash_equivalents,
        b.current_assets,
        b.current_liabilities,
        b.total_assets,
        b.total_liabilities,
        b.stockholders_equity_parent,
        b.stockholders_equity_total,
        TRUE AS is_derived,
        CASE
            WHEN q3.fiscal_year IS NOT NULL THEN 'fy_minus_9m'
            ELSE 'unresolvable_missing_q3'
        END AS derivation_method,
        fy.accession_number AS source_accession_number,
        fy.filed_date AS source_filed_date
    FROM pivoted fy
    LEFT JOIN pivoted q3
      ON fy.cik = q3.cik
     AND fy.fiscal_year = q3.fiscal_year
     AND fy.start_date = q3.start_date
     AND q3.fiscal_period = 'Q3'
     AND q3.duration_scope = 'cumulative_9m'
     AND q3.end_date < fy.end_date
    LEFT JOIN balance_instant b
      ON fy.cik = b.cik
     AND fy.fiscal_year = b.fiscal_year
     AND b.fiscal_period = 'FY'
    WHERE fy.fiscal_period = 'FY'
      AND fy.duration_scope = 'annual_12m'
),

combined AS (
    SELECT * FROM as_reported_periods
    UNION ALL
    SELECT * FROM derived_q4
)

SELECT *
FROM combined
