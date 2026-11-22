-- Staging model: stg_sec_facts
-- Cleanses, standardizes 10 Group A canonical metrics, supports USD and USD/shares,
-- and establishes As-reported vs Latest-known dual ranking.

WITH raw_facts AS (
    SELECT
        fact_key,
        observation_key,
        cik,
        ticker,
        taxonomy,
        tag,
        value,
        unit,
        period_type,
        start_date,
        end_date,
        fiscal_year,
        fiscal_period,
        duration_days,
        duration_scope,
        form,
        filed_date,
        accession_number,
        is_amendment,
        source_file
    FROM {{ source('sec_silver', 'facts') }}
),

standardized AS (
    SELECT
        fact_key,
        observation_key,
        cik,
        ticker,
        taxonomy,
        tag,
        -- Standardize raw GAAP tags into 10 Group A canonical metric names
        CASE
            -- Income Statement line items (Duration)
            WHEN tag IN (
                'Revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'SalesRevenueNet'
            ) THEN 'revenues'
            WHEN tag = 'OperatingIncomeLoss' THEN 'operating_income'
            WHEN tag IN ('NetIncomeLoss', 'ProfitLoss') THEN 'net_income'
            WHEN tag = 'GrossProfit' THEN 'gross_profit'
            WHEN tag = 'EarningsPerShareDiluted' THEN 'eps_diluted'

            -- Balance Sheet line items (Instant)
            WHEN tag = 'CashAndCashEquivalentsAtCarryingValue' THEN 'cash_and_cash_equivalents'
            WHEN tag = 'AssetsCurrent' THEN 'current_assets'
            WHEN tag = 'LiabilitiesCurrent' THEN 'current_liabilities'
            WHEN tag = 'Assets' THEN 'total_assets'
            WHEN tag = 'Liabilities' THEN 'total_liabilities'
            WHEN tag = 'StockholdersEquity' THEN 'stockholders_equity_parent'
            WHEN tag = 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest' THEN 'stockholders_equity_total'

            ELSE LOWER(tag)
        END AS canonical_metric,
        CAST(value AS DECIMAL(28, 2)) AS value,
        unit,
        period_type,
        start_date,
        end_date,
        fiscal_year,
        UPPER(TRIM(fiscal_period)) AS fiscal_period,
        duration_days,
        duration_scope,
        form,
        filed_date,
        accession_number,
        is_amendment,
        source_file
    FROM raw_facts
    WHERE unit IN ('USD', 'USD/shares')
      AND duration_scope != 'unclassified'
),

ranked AS (
    SELECT
        *,
        -- Rank 1 for original filing at the time period was first reported (As-Reported)
        -- Ties within the same filing are resolved by end_date DESC to pick the current period observation
        ROW_NUMBER() OVER (
            PARTITION BY
                cik,
                canonical_metric,
                fiscal_year,
                fiscal_period,
                duration_scope
            ORDER BY filed_date ASC, accession_number ASC, end_date DESC
        ) AS rank_as_reported,

        -- Rank 1 for latest known filing containing this period (Latest-Known / Restated)
        ROW_NUMBER() OVER (
            PARTITION BY
                cik,
                canonical_metric,
                fiscal_year,
                fiscal_period,
                duration_scope
            ORDER BY filed_date DESC, accession_number DESC, end_date DESC
        ) AS rank_latest_known
    FROM standardized
)

SELECT
    fact_key,
    observation_key,
    cik,
    ticker,
    taxonomy,
    tag,
    canonical_metric,
    value,
    unit,
    period_type,
    start_date,
    end_date,
    fiscal_year,
    fiscal_period,
    duration_days,
    duration_scope,
    form,
    filed_date,
    accession_number,
    is_amendment,
    -- Audit provenance flags
    CASE WHEN rank_as_reported = 1 THEN TRUE ELSE FALSE END AS is_as_reported,
    CASE WHEN rank_latest_known = 1 THEN TRUE ELSE FALSE END AS is_latest_known,
    source_file,
    'US' AS market,
    'USD' AS currency
FROM ranked
