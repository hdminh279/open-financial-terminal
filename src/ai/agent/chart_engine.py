"""Plotly Spec-Driven Financial Visualization Engine (Refactored).

Provides deterministic chart extraction, validation, and serialization:
1. LLM-driven chart preference consumption (zero query keyword sniffing).
2. Decoupled TimeSeriesNormalizer for Annual (FY) and Quarterly (Q1-Q4) data.
3. Accounting-accurate YTD cumulative to discrete derivation for duration metrics.
4. Point-in-time preservation for instant balance sheet metrics (no subtraction).
5. Strict missing-value preservation (None is preserved; never converted to 0).
6. Multi-ticker unit consistency check and Plotly Figure JSON synthesis.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.ai.agent.contracts import ChartData, ChartType
from src.core.financial.metric_registry import MetricRegistry

logger = logging.getLogger(__name__)

# Point-in-time instant metrics that MUST NOT undergo duration subtraction
POINT_IN_TIME_METRICS: Set[str] = {
    "total_assets",
    "current_assets",
    "cash_and_cash_equivalents",
    "total_liabilities",
    "current_liabilities",
    "stockholders_equity_parent",
    "stockholders_equity_total",
}

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


@dataclass
class NormalizedPeriodPoint:
    """Standardized observation in a financial time series."""

    period_label: str  # e.g. 'FY2023' or '2023 Q1'
    fiscal_year: int
    fiscal_period: str  # 'FY', 'Q1', 'Q2', 'Q3', 'Q4'
    value: Optional[float]
    is_derived: bool = False
    provenance_url: Optional[str] = None


@dataclass
class NormalizedTimeSeries:
    """Normalized multi-period series for a single metric and company."""

    metric_name: str
    display_name: str
    ticker: str
    unit: str
    is_percentage: bool
    is_point_in_time: bool
    x_labels: List[str]
    points: List[NormalizedPeriodPoint]


class TimeSeriesNormalizer:
    """Normalizes raw SQL financial records into ordered, arithmetically sound time series."""

    def __init__(self, metric_registry: Optional[MetricRegistry] = None) -> None:
        self.metric_registry = metric_registry or MetricRegistry()

    def is_point_in_time_metric(self, metric_name: str) -> bool:
        """Check whether a metric represents an instant balance sheet snapshot."""
        norm_name = metric_name.strip().lower()
        if norm_name in POINT_IN_TIME_METRICS:
            return True
        definition = self.metric_registry.resolve(norm_name)
        if definition and definition.period_type == "instant":
            return True
        return False

    def normalize(
        self,
        records: List[Dict[str, Any]],
        metric_name: str,
        ticker: str,
        target_periods: Optional[Sequence[str]] = None,
        target_years: Optional[Sequence[int]] = None,
    ) -> Optional[NormalizedTimeSeries]:
        """Normalize records into either an Annual or Quarterly time series.

        Args:
            records: Raw financial metric records from SQL tool.
            metric_name: Canonical metric identifier.
            ticker: Company ticker symbol.
            target_periods: Optional fiscal_periods filter from query plan.
            target_years: Optional fiscal_years filter or range from query plan.

        Returns:
            NormalizedTimeSeries instance, or None if insufficient observations exist.
        """
        # Filter records for this specific metric and ticker
        m_norm = metric_name.strip().lower()
        t_norm = ticker.strip().upper()
        matched = [
            r for r in records
            if r.get("metric_name", "").strip().lower() == m_norm
            and r.get("ticker", "").strip().upper() == t_norm
        ]

        if not matched:
            return None

        # Resolve metric metadata
        definition = self.metric_registry.resolve(m_norm)
        display_name = definition.display_name if definition else m_norm.replace("_", " ").title()
        default_unit = definition.default_unit if definition else "USD"
        is_percentage = default_unit == "%" or "margin" in m_norm or "growth" in m_norm
        is_pit = self.is_point_in_time_metric(m_norm)

        # Determine granularity: check target_periods or record contents
        periods_set = {str(r.get("fiscal_period", "")).upper() for r in matched}
        has_quarterly = any(q in periods_set for q in ("Q1", "Q2", "Q3", "Q4"))
        has_annual = "FY" in periods_set

        prefer_quarterly = False
        if target_periods:
            upper_targets = [p.upper() for p in target_periods]
            if any(q in upper_targets for q in ("Q1", "Q2", "Q3", "Q4")):
                prefer_quarterly = True
        elif has_quarterly and not has_annual:
            prefer_quarterly = True

        if prefer_quarterly:
            return self._normalize_quarterly(
                matched, m_norm, display_name, t_norm, default_unit, is_percentage, is_pit, target_years=target_years
            )
        else:
            return self._normalize_annual(
                matched, m_norm, display_name, t_norm, default_unit, is_percentage, is_pit, target_years=target_years
            )

    def _normalize_annual(
        self,
        records: List[Dict[str, Any]],
        metric_name: str,
        display_name: str,
        ticker: str,
        default_unit: str,
        is_percentage: bool,
        is_pit: bool,
        target_years: Optional[Sequence[int]] = None,
    ) -> Optional[NormalizedTimeSeries]:
        """Process annual FY time series sorted by fiscal year."""
        fy_records = [
            r for r in records
            if str(r.get("fiscal_period", "")).upper() == "FY"
            and r.get("fiscal_year") is not None
        ]

        if not fy_records:
            return None

        # Group by fiscal_year to eliminate duplicates
        year_to_record: Dict[int, Dict[str, Any]] = {}
        for r in fy_records:
            fy = int(r["fiscal_year"])
            # Keep as-reported record or latest observation
            if fy not in year_to_record or (not year_to_record[fy].get("is_derived") and r.get("is_derived")):
                year_to_record[fy] = r

        if target_years:
            sorted_years = sorted(list(dict.fromkeys(target_years)))
        else:
            sorted_years = sorted(year_to_record.keys())

        if len(sorted_years) < 2:
            return None

        points: List[NormalizedPeriodPoint] = []
        x_labels: List[str] = []

        for fy in sorted_years:
            rec = year_to_record.get(fy)
            raw_val = rec.get("metric_value") if rec else None
            val_float = float(raw_val) if raw_val is not None else None
            lbl = f"FY{fy}"
            x_labels.append(lbl)
            points.append(
                NormalizedPeriodPoint(
                    period_label=lbl,
                    fiscal_year=fy,
                    fiscal_period="FY",
                    value=val_float,
                    is_derived=bool(rec.get("is_derived", False)) if rec else False,
                    provenance_url=rec.get("provenance_url") if rec else None,
                )
            )

        valid_points = [p for p in points if p.value is not None]
        if len(valid_points) < 2:
            return None

        return NormalizedTimeSeries(
            metric_name=metric_name,
            display_name=display_name,
            ticker=ticker,
            unit=default_unit,
            is_percentage=is_percentage,
            is_point_in_time=is_pit,
            x_labels=x_labels,
            points=points,
        )

    def _is_ytd_record(self, r: Dict[str, Any]) -> bool:
        """Check whether a record represents a cumulative year-to-date observation."""
        if r.get("is_cumulative") is True:
            return True
        scope = str(r.get("duration_scope", "")).lower()
        if scope in ("cumulative_6m", "cumulative_9m", "annual_12m"):
            return True
        ptype = str(r.get("period_type", "")).lower()
        if "cumulative" in ptype or "ytd" in ptype:
            return True
        fp = str(r.get("fiscal_period", "")).lower()
        if "ytd" in fp:
            return True
        return False

    def _normalize_quarterly(
        self,
        records: List[Dict[str, Any]],
        metric_name: str,
        display_name: str,
        ticker: str,
        default_unit: str,
        is_percentage: bool,
        is_pit: bool,
        target_years: Optional[Sequence[int]] = None,
    ) -> Optional[NormalizedTimeSeries]:
        """Process quarterly time series Q1->Q4 with discrete/YTD accounting derivation."""
        discrete_data: Dict[Tuple[int, str], Dict[str, Any]] = {}
        ytd_data: Dict[Tuple[int, str], Dict[str, Any]] = {}
        fy_data: Dict[int, Dict[str, Any]] = {}

        for r in records:
            fy = r.get("fiscal_year")
            if fy is None:
                continue
            fy_int = int(fy)
            fp_raw = str(r.get("fiscal_period", "")).upper()
            fp = fp_raw.replace("_YTD", "").replace("YTD_", "").replace("YTD", "").strip()

            if fp == "FY" or fp_raw == "FY":
                fy_data[fy_int] = r
                ytd_data[(fy_int, "FY")] = r
            elif fp in ("Q1", "Q2", "Q3", "Q4"):
                is_ytd = self._is_ytd_record(r)
                if fp == "Q1":
                    # Q1 is identical for discrete and YTD
                    discrete_data[(fy_int, "Q1")] = r
                    ytd_data[(fy_int, "Q1")] = r
                elif is_ytd:
                    ytd_data[(fy_int, fp)] = r
                else:
                    discrete_data[(fy_int, fp)] = r

        # Collect unique fiscal years across all observed periods
        all_years = sorted(list({fy for (fy, _) in discrete_data.keys()} | {fy for (fy, _) in ytd_data.keys()} | set(fy_data.keys())))
        if target_years:
            target_year_set = set(target_years)
            all_years = [y for y in all_years if y in target_year_set]
        if not all_years:
            return None

        points: List[NormalizedPeriodPoint] = []
        x_labels: List[str] = []

        for fy in all_years:
            for q_str in ("Q1", "Q2", "Q3", "Q4"):
                key = (fy, q_str)
                disc_rec = discrete_data.get(key)
                val_float: Optional[float] = None
                is_derived = False
                prov_url: Optional[str] = None

                # 1. Point-in-time metrics: strictly NO cumulative subtraction
                if is_pit:
                    # Look up snapshot observation (discrete or quarter-end record)
                    rec = disc_rec or ytd_data.get(key)
                    if rec and rec.get("metric_value") is not None:
                        val_float = float(rec["metric_value"])
                        is_derived = bool(rec.get("is_derived", False))
                        prov_url = rec.get("provenance_url")
                else:
                    # 2. Duration metrics: use discrete if available, else derive from YTD
                    if disc_rec and disc_rec.get("metric_value") is not None:
                        val_float = float(disc_rec["metric_value"])
                        is_derived = bool(disc_rec.get("is_derived", False))
                        prov_url = disc_rec.get("provenance_url")
                    else:
                        val_float, is_derived, prov_url = self._derive_quarterly_duration(
                            fy=fy,
                            quarter=q_str,
                            ytd_data=ytd_data,
                            fy_data=fy_data,
                            discrete_data=discrete_data,
                        )

                lbl = f"{fy} {q_str}"
                x_labels.append(lbl)
                points.append(
                    NormalizedPeriodPoint(
                        period_label=lbl,
                        fiscal_year=fy,
                        fiscal_period=q_str,
                        value=val_float,
                        is_derived=is_derived,
                        provenance_url=prov_url,
                    )
                )

        # Drop trailing/leading empty quarters if entire series is empty (< 2 non-null observations)
        valid_points = [p for p in points if p.value is not None]
        if len(valid_points) < 2:
            return None

        return NormalizedTimeSeries(
            metric_name=metric_name,
            display_name=display_name,
            ticker=ticker,
            unit=default_unit,
            is_percentage=is_percentage,
            is_point_in_time=is_pit,
            x_labels=x_labels,
            points=points,
        )

    def _derive_quarterly_duration(
        self,
        fy: int,
        quarter: str,
        ytd_data: Dict[Tuple[int, str], Dict[str, Any]],
        fy_data: Dict[int, Dict[str, Any]],
        discrete_data: Dict[Tuple[int, str], Dict[str, Any]],
    ) -> Tuple[Optional[float], bool, Optional[str]]:
        """Perform arithmetic subtraction for cumulative duration metrics:
        - Q1 = YTD_Q1
        - Q2 = YTD_Q2 - YTD_Q1
        - Q3 = YTD_Q3 - YTD_Q2
        - Q4 = FY - YTD_Q3 (or FY - (Q1 + Q2 + Q3))

        Strict rules:
        - Must have compatible fiscal year, unit, and non-null values.
        - Never fabricate values if source data is missing.
        """
        def _get_val_and_unit(rec: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
            if not rec or rec.get("metric_value") is None:
                return None, None
            try:
                return float(rec["metric_value"]), str(rec.get("unit", "")).strip().lower() or None
            except (ValueError, TypeError):
                return None, None

        if quarter == "Q1":
            q1_rec = ytd_data.get((fy, "Q1"))
            val, _ = _get_val_and_unit(q1_rec)
            if val is not None:
                return val, False, q1_rec.get("provenance_url") if q1_rec else None

        elif quarter == "Q2":
            q2_rec = ytd_data.get((fy, "Q2"))
            q1_rec = ytd_data.get((fy, "Q1"))
            v2, u2 = _get_val_and_unit(q2_rec)
            v1, u1 = _get_val_and_unit(q1_rec)
            if v2 is not None and v1 is not None:
                if u1 and u2 and u1 != u2:
                    logger.warning("YTD derivation unit mismatch for FY%d Q2: %s vs %s", fy, u1, u2)
                    return None, False, None
                return v2 - v1, True, q2_rec.get("provenance_url") if q2_rec else None

        elif quarter == "Q3":
            q3_rec = ytd_data.get((fy, "Q3"))
            q2_rec = ytd_data.get((fy, "Q2"))
            v3, u3 = _get_val_and_unit(q3_rec)
            v2, u2 = _get_val_and_unit(q2_rec)
            if v3 is not None and v2 is not None:
                if u2 and u3 and u2 != u3:
                    logger.warning("YTD derivation unit mismatch for FY%d Q3: %s vs %s", fy, u2, u3)
                    return None, False, None
                return v3 - v2, True, q3_rec.get("provenance_url") if q3_rec else None

        elif quarter == "Q4":
            fy_rec = fy_data.get(fy)
            q3_rec = ytd_data.get((fy, "Q3"))
            v_fy, u_fy = _get_val_and_unit(fy_rec)
            v_q3, u_q3 = _get_val_and_unit(q3_rec)
            if v_fy is not None and v_q3 is not None:
                if u_fy and u_q3 and u_fy != u_q3:
                    logger.warning("YTD derivation unit mismatch for FY%d Q4: %s vs %s", fy, u_fy, u_q3)
                    return None, False, None
                return v_fy - v_q3, True, fy_rec.get("provenance_url") if fy_rec else None

            # Fallback for Q4: FY - (Q1 + Q2 + Q3) if discrete Q1..Q3 are known
            q1_disc = discrete_data.get((fy, "Q1"))
            q2_disc = discrete_data.get((fy, "Q2"))
            q3_disc = discrete_data.get((fy, "Q3"))
            v1, _ = _get_val_and_unit(q1_disc)
            v2, _ = _get_val_and_unit(q2_disc)
            v3, _ = _get_val_and_unit(q3_disc)
            if v_fy is not None and v1 is not None and v2 is not None and v3 is not None:
                return v_fy - (v1 + v2 + v3), True, fy_rec.get("provenance_url") if fy_rec else None

        return None, False, None


class ChartEngine:
    """Spec-driven visualization engine synthesizing Plotly and Mermaid charts."""

    _EXCLUDED_METRIC_SUBSTRINGS: Set[str] = {"cik", "accession", "hash", "key"}

    def __init__(self, metric_registry: Optional[MetricRegistry] = None) -> None:
        """Initialize ChartEngine with in-memory MetricRegistry SSOT."""
        self.metric_registry = metric_registry or MetricRegistry()
        self.normalizer = TimeSeriesNormalizer(self.metric_registry)

    def should_render_chart(
        self,
        chart_preference: Optional[str],
        has_sql_data: bool,
    ) -> bool:
        """Check whether visualization was explicitly requested or planned by the Router LLM.

        Rules:
        - Must have valid SQL records.
        - chart_preference must NOT be 'none', empty, or unsupported.
        """
        if not has_sql_data:
            return False

        if not chart_preference:
            return False

        pref_norm = str(chart_preference).strip().lower()
        if pref_norm in ("none", "false", "no", ChartType.NONE.value):
            return False

        return pref_norm in ("line", "bar", "scatter", "area")

    @classmethod
    def extract_charts(
        cls,
        sql_data: Optional[Dict[str, Any]],
        chart_preference: Optional[str] = "none",
        ticker: Optional[str] = None,
        query_plan: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[ChartData]:
        """Convenience classmethod for extracting financial charts without manual instantiation."""
        return cls().extract_financial_charts(
            sql_data=sql_data,
            chart_preference=chart_preference,
            ticker=ticker,
            query_plan=query_plan,
            **kwargs,
        )

    def extract_financial_charts(
        self,
        sql_data: Optional[Dict[str, Any]],
        chart_preference: Optional[str] = "none",
        ticker: Optional[str] = None,
        query_plan: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[ChartData]:
        """Extract validated time series and construct Plotly-ready ChartData objects.

        Args:
            sql_data: FinancialQueryResponse dictionary from state.
            chart_preference: LLM-determined preference ('line', 'bar', 'scatter', 'area', 'none').
            ticker: Optional ticker header string.
            query_plan: Optional structured plan dictionary from Router.

        Returns:
            List of validated ChartData instances with plotly_figure_json.
        """
        charts: List[ChartData] = []
        if not sql_data or not sql_data.get("data"):
            logger.debug("ChartEngine: Skipped, no SQL data available.")
            return charts

        # 1. Router Chart Preference Guardrail (No keyword sniffing)
        if not self.should_render_chart(chart_preference, has_sql_data=True):
            logger.info("ChartEngine: Chart skipped because chart_preference is '%s'.", chart_preference)
            return charts

        records: List[Dict[str, Any]] = sql_data["data"]
        pref_norm = str(chart_preference).strip().lower()

        # Map chart_preference string to ChartType enum
        chart_type_map = {
            "line": ChartType.LINE,
            "bar": ChartType.BAR,
            "scatter": ChartType.SCATTER,
            "area": ChartType.AREA,
        }
        selected_chart_type = chart_type_map.get(pref_norm, ChartType.BAR)

        # Extract target metrics and tickers present in records
        found_metrics: List[str] = []
        found_tickers: List[str] = []
        for r in records:
            m = r.get("metric_name", "").strip().lower()
            t = r.get("ticker", "").strip().upper()
            if m and m not in found_metrics and not any(sub in m for sub in self._EXCLUDED_METRIC_SUBSTRINGS):
                found_metrics.append(m)
            if t and t not in found_tickers:
                found_tickers.append(t)

        target_periods = query_plan.get("fiscal_periods") if query_plan else None
        target_years = query_plan.get("fiscal_years") if query_plan else None
        if target_years and len(target_years) == 2 and target_years[0] < target_years[1]:
            if target_years[1] - target_years[0] <= 30:
                target_years = list(range(target_years[0], target_years[1] + 1))

        # 2. Iterate each metric, normalize series, and validate
        for m_name in found_metrics:
            series_list: List[NormalizedTimeSeries] = []
            for t_name in found_tickers:
                ts = self.normalizer.normalize(
                    records=records,
                    metric_name=m_name,
                    ticker=t_name,
                    target_periods=target_periods,
                    target_years=target_years,
                )
                if ts:
                    series_list.append(ts)

            if not series_list:
                continue

            # Verify that all companies share a common ordered x-axis
            common_x = series_list[0].x_labels
            display_name = series_list[0].display_name
            is_percentage = series_list[0].is_percentage

            # 3. Unit Scaling & Missing Value Preservation (None is never converted to 0)
            flat_non_nulls: List[float] = [
                pt.value for s in series_list for pt in s.points if pt.value is not None
            ]
            if len(flat_non_nulls) < 2:
                logger.warning("ChartEngine: Metric '%s' has insufficient data points (%d < 2).", m_name, len(flat_non_nulls))
                continue

            datasets: List[Dict[str, Any]] = []

            if is_percentage:
                unit_label = "%"
                max_abs = max(abs(v) for v in flat_non_nulls)
                scale_factor = 100.0 if max_abs <= 1.5 else 1.0

                for s in series_list:
                    scaled_values = [
                        round(pt.value * scale_factor, 2) if pt.value is not None else None
                        for pt in s.points
                    ]
                    lbl = f"{s.ticker} {display_name}" if len(series_list) > 1 else display_name
                    datasets.append({"label": f"{lbl} (%)", "values": scaled_values})

            elif series_list[0].unit == "ratio":
                unit_label = "Ratio"
                for s in series_list:
                    scaled_values = [
                        round(pt.value, 2) if pt.value is not None else None
                        for pt in s.points
                    ]
                    lbl = f"{s.ticker} {display_name}" if len(series_list) > 1 else display_name
                    datasets.append({"label": lbl, "values": scaled_values})

            else:
                max_abs = max(abs(v) for v in flat_non_nulls)
                is_billion = max_abs >= 1_000_000_000
                unit_label = "USD (Billions)" if is_billion else "USD"
                div = 1_000_000_000.0 if is_billion else 1.0

                for s in series_list:
                    scaled_values = [
                        round(pt.value / div, 2) if pt.value is not None else None
                        for pt in s.points
                    ]
                    lbl = f"{s.ticker} {display_name}" if len(series_list) > 1 else display_name
                    datasets.append({"label": lbl, "values": scaled_values})

            # 4. Construct Descriptive Title
            company_header = " vs ".join(s.ticker for s in series_list) if len(series_list) > 1 else series_list[0].ticker
            chart_title = f"{company_header} {display_name} ({unit_label})"

            # 5. Build Plotly Figure JSON
            plotly_json = self._build_plotly_figure(
                title=chart_title,
                x_labels=common_x,
                datasets=datasets,
                unit_label=unit_label,
                chart_type=selected_chart_type,
            )

            # 6. Build Backward-Compatible Mermaid and ASCII blocks
            mermaid_code = self._build_mermaid_code(
                title=chart_title,
                x_labels=common_x,
                datasets=datasets,
                unit_label=unit_label,
                chart_type=selected_chart_type,
            )
            ascii_chart = self._build_ascii_chart(
                title=chart_title,
                x_labels=common_x,
                datasets=datasets,
            )

            chart_data = ChartData(
                title=chart_title,
                chart_type=selected_chart_type,
                x_labels=common_x,
                datasets=datasets,
                unit=unit_label,
                mermaid_code=mermaid_code,
                ascii_chart=ascii_chart,
                plotly_figure_json=plotly_json,
            )
            charts.append(chart_data)

        return charts

    def _build_plotly_figure(
        self,
        title: str,
        x_labels: List[str],
        datasets: List[Dict[str, Any]],
        unit_label: str,
        chart_type: ChartType,
    ) -> Dict[str, Any]:
        """Construct Plotly Figure specification matching modern workstation schema."""
        palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
        data_traces = []

        is_line_type = chart_type in (ChartType.LINE, ChartType.SCATTER, ChartType.AREA)

        for idx, ds in enumerate(datasets):
            color = palette[idx % len(palette)]
            if is_line_type:
                trace = {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": ds["label"],
                    "x": x_labels,
                    "y": ds["values"],
                    "connectgaps": False,
                    "line": {"color": color, "width": 3, "shape": "spline"},
                    "marker": {"size": 8, "color": color},
                    "hovertemplate": f"%{{x}}<br>%{{y}} {unit_label}<extra></extra>",
                }
                if chart_type == ChartType.AREA:
                    trace["fill"] = "tozeroy"
            else:
                trace = {
                    "type": "bar",
                    "name": ds["label"],
                    "x": x_labels,
                    "y": ds["values"],
                    "marker": {"color": color, "opacity": 0.85},
                    "hovertemplate": f"%{{x}}<br>%{{y}} {unit_label}<extra></extra>",
                }
            data_traces.append(trace)

        layout = {
            "title": {"text": title, "font": {"size": 15, "weight": 600}},
            "xaxis": {"title": {"text": "Fiscal Period"}, "showgrid": True, "gridcolor": "#f1f5f9"},
            "yaxis": {"title": {"text": unit_label}, "showgrid": True, "gridcolor": "#f1f5f9"},
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": {"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
            "margin": {"l": 50, "r": 30, "t": 50, "b": 60},
            "autosize": True,
        }

        config = {
            "responsive": True,
            "displayModeBar": False,
        }

        return {"data": data_traces, "layout": layout, "config": config}

    def _build_mermaid_code(
        self,
        title: str,
        x_labels: List[str],
        datasets: List[Dict[str, Any]],
        unit_label: str,
        chart_type: ChartType,
    ) -> str:
        """Construct Mermaid xychart-beta code block."""
        chart_element = "line" if chart_type in (ChartType.LINE, ChartType.SCATTER, ChartType.AREA) else "bar"
        labels_str = ", ".join(f'"{lbl}"' for lbl in x_labels)

        all_vals = [v for ds in datasets for v in ds["values"] if v is not None]
        y_max = max(all_vals) * 1.2 if all_vals and max(all_vals) > 0 else 100
        y_min = min(all_vals) if all_vals and min(all_vals) < 0 else 0

        series_str = ""
        for ds in datasets:
            vals_str = ", ".join(str(v if v is not None else 0) for v in ds["values"])
            series_str += f"    {chart_element} [{vals_str}]\n"

        return (
            f"```mermaid\n"
            f"xychart-beta\n"
            f'    title "{title}"\n'
            f"    x-axis [{labels_str}]\n"
            f'    y-axis "{unit_label}" {int(y_min)} --> {int(y_max)}\n'
            f"{series_str}"
            f"```"
        )

    def _build_ascii_chart(
        self,
        title: str,
        x_labels: List[str],
        datasets: List[Dict[str, Any]],
    ) -> str:
        """Construct terminal-friendly ASCII representation."""
        all_vals = [v for ds in datasets for v in ds["values"] if v is not None and v > 0]
        v_max = max(all_vals) if all_vals else 100.0

        ascii_lines = [f"--- {title} ---"]
        for ds in datasets:
            if len(datasets) > 1:
                ascii_lines.append(f"[{ds['label']}]")
            for lbl, val in zip(x_labels, ds["values"]):
                if val is not None:
                    bar_len = max(1, int((val / v_max) * 20)) if v_max > 0 and val > 0 else 0
                    bars = "=" * bar_len
                    ascii_lines.append(f"{lbl}: [{bars}] {val}")
                else:
                    ascii_lines.append(f"{lbl}: [No data]")
        return "\n".join(ascii_lines)
