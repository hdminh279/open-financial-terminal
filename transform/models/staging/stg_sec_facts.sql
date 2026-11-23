-- Staging model: stg_sec_facts
-- Cleanses, standardizes canonical metrics across multi-currency SEC filings
-- (US-GAAP and IFRS 20-F: USD, EUR, TWD, INR, GBP, JPY, SEK, BRL, DKK), selects one primary
-- reporting currency per issuer, and establishes As-reported vs Latest-known ranking.

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
        -- Standardize raw US-GAAP and IFRS tags into canonical metric names
        CASE
            -- Income Statement line items (Duration)
            WHEN tag IN (
                'Revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'SalesRevenueNet',
                'Revenue',
                'RevenueFromContractsWithCustomers'
            ) THEN 'revenues'
            WHEN tag IN (
                'OperatingIncomeLoss',
                'ProfitLossFromOperatingActivities',
                'OperatingProfit'
            ) THEN 'operating_income'
            WHEN tag IN ('NetIncomeLoss', 'ProfitLoss') THEN 'net_income'
            WHEN tag = 'GrossProfit' THEN 'gross_profit'
            WHEN tag = 'EarningsPerShareDiluted' THEN 'eps_diluted'

            -- Balance Sheet line items (Instant)
            WHEN tag IN (
                'CashAndCashEquivalentsAtCarryingValue',
                'CashAndCashEquivalents'
            ) THEN 'cash_and_cash_equivalents'
            WHEN tag = 'AssetsCurrent' THEN 'current_assets'
            WHEN tag = 'LiabilitiesCurrent' THEN 'current_liabilities'
            WHEN tag = 'Assets' THEN 'total_assets'
            WHEN tag = 'Liabilities' THEN 'total_liabilities'
            WHEN tag IN (
                'StockholdersEquity',
                'EquityAttributableToOwnersOfParent'
            ) THEN 'stockholders_equity_parent'
            WHEN tag IN (
                'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                'Equity'
            ) THEN 'stockholders_equity_total'

            ELSE LOWER(tag)
        END AS canonical_metric,
        CASE tag
            WHEN 'Revenues' THEN 1
            WHEN 'RevenueFromContractWithCustomerExcludingAssessedTax' THEN 2
            WHEN 'SalesRevenueNet' THEN 3
            WHEN 'Revenue' THEN 1
            WHEN 'RevenueFromContractsWithCustomers' THEN 2
            WHEN 'OperatingIncomeLoss' THEN 1
            WHEN 'ProfitLossFromOperatingActivities' THEN 2
            WHEN 'OperatingProfit' THEN 3
            ELSE 1
        END AS tag_priority,
        CAST(value AS DECIMAL(28, 2)) AS value,
        unit,
        SPLIT_PART(unit, '/', 1) AS currency,
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
    WHERE unit IN (
        'USD', 'USD/shares',
        'EUR', 'EUR/shares',
        'TWD', 'TWD/shares',
        'INR', 'INR/shares',
        'GBP', 'GBP/shares',
        'JPY', 'JPY/shares',
        'SEK', 'SEK/shares',
        'BRL', 'BRL/shares',
        'DKK', 'DKK/shares'
    )
      AND duration_scope != 'unclassified'
),

-- Select one consistent primary reporting currency per issuer (CIK)
-- so dual-currency FPI filings never produce duplicate statement keys
currency_counts AS (
    SELECT
        cik,
        currency,
        COUNT(*) AS fact_count
    FROM standardized
    GROUP BY cik, currency
),

primary_currency AS (
    SELECT
        cik,
        currency AS primary_currency,
        ROW_NUMBER() OVER (
            PARTITION BY cik
            ORDER BY fact_count DESC, CASE WHEN currency = 'USD' THEN 0 ELSE 1 END ASC, currency ASC
        ) AS rn
    FROM currency_counts
),

filtered_primary AS (
    SELECT s.*
    FROM standardized s
    INNER JOIN primary_currency pc
        ON s.cik = pc.cik
       AND s.currency = pc.primary_currency
       AND pc.rn = 1
),

ranked AS (
    SELECT
        *,
        -- Rank 1 for original filing at the time period was first reported (As-Reported)
        -- Ties within the same filing are resolved by end_date DESC, then canonical tag priority
        ROW_NUMBER() OVER (
            PARTITION BY
                cik,
                currency,
                canonical_metric,
                fiscal_year,
                fiscal_period,
                duration_scope
            ORDER BY
                filed_date ASC,
                accession_number ASC,
                end_date DESC,
                tag_priority ASC
        ) AS rank_as_reported,

        -- Rank 1 for latest known filing containing this period (Latest-Known / Restated)
        ROW_NUMBER() OVER (
            PARTITION BY
                cik,
                currency,
                canonical_metric,
                fiscal_year,
                fiscal_period,
                duration_scope
            ORDER BY
                filed_date DESC,
                accession_number DESC,
                end_date DESC,
                tag_priority ASC
        ) AS rank_latest_known
    FROM filtered_primary
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
    currency
FROM ranked
