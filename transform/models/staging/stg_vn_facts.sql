-- Staging model: stg_vn_facts
-- Cleanses, standardizes Vietnamese VAS canonical metrics, supports VND, VND/shares, and %,
-- and establishes As-reported vs Latest-known dual ranking for VN30 constituents.

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
        source_file,
        market
    FROM {{ source('sec_silver', 'vn_facts') }}
),

standardized AS (
    SELECT
        fact_key,
        observation_key,
        cik,
        ticker,
        taxonomy,
        tag,
        LOWER(TRIM(tag)) AS canonical_metric,
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
        -- Canonical statement accession number shared across all line items of the same filing
        md5(ticker || '-' || CAST(fiscal_year AS VARCHAR) || '-' || fiscal_period || '-' || form) AS accession_number,
        is_amendment,
        source_file,
        market,
        unit AS currency
    FROM raw_facts
    WHERE unit IN ('VND', 'VND/shares', '%')
      AND duration_scope != 'unclassified'
),

ranked AS (
    SELECT
        *,
        -- Rank 1 for original filing at the time period was first reported (As-Reported)
        ROW_NUMBER() OVER (
            PARTITION BY
                ticker,
                canonical_metric,
                fiscal_year,
                fiscal_period,
                duration_scope
            ORDER BY filed_date ASC, accession_number ASC, end_date DESC
        ) AS rank_as_reported,

        -- Rank 1 for latest known filing containing this period (Latest-Known / Restated)
        ROW_NUMBER() OVER (
            PARTITION BY
                ticker,
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
    market,
    currency
FROM ranked
