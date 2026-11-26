/**
 * Corporate Filing Intelligence - Financial Canvas Chart & Table Controller
 * Manages Chart.js dual-axis combo visualization, interactive legends,
 * theme updates, dynamic financial data tables, sorting, and CSV/JSON/PNG export.
 */

import { escapeHtml } from "./api.js";

export const defaultChartData = {
  categories: ["2021", "2022", "2023", "2024"],
  series: [
    {
      name: "Apple",
      data: [30000, 34500, 42000, 49200],
      color: "#00A3FF",
      type: "bar",
    },
    {
      name: "Microsoft",
      data: [18000, 21000, 27500, 31000],
      color: "#00E599",
      type: "bar",
    },
    {
      name: "Biên hoạt động",
      data: [43, 37, 47, 45],
      color: "#F5A623",
      type: "line",
    },
  ],
};

export function exportTableData(tableEl, format, filenamePrefix) {
  const headers = Array.from(tableEl.querySelectorAll("thead th")).map((th) =>
    th.textContent.replace(/[↑↓]/g, "").trim()
  );
  const rows = Array.from(tableEl.querySelectorAll("tbody tr")).map((tr) =>
    Array.from(tr.querySelectorAll("td")).map((td) => td.textContent.trim())
  );
  if (headers.length === 0 && rows.length === 0) return;

  let content = "";
  let mime = "text/plain;charset=utf-8;";
  let ext = "txt";

  if (format === "json") {
    const jsonRecords = rows.map((cells) => {
      const obj = {};
      cells.forEach((val, idx) => {
        const key = headers[idx] || `col_${idx + 1}`;
        obj[key] = val;
      });
      return obj;
    });
    content = JSON.stringify(jsonRecords, null, 2);
    mime = "application/json;charset=utf-8;";
    ext = "json";
  } else {
    const escapeCsv = (val) => {
      const str = String(val ?? "");
      return /[",\n\r]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    const csvLines = [];
    if (headers.length > 0) csvLines.push(headers.map(escapeCsv).join(","));
    rows.forEach((r) => csvLines.push(r.map(escapeCsv).join(",")));
    content = "\uFEFF" + csvLines.join("\r\n");
    mime = "text/csv;charset=utf-8;";
    ext = "csv";
  }

  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filenamePrefix}_${Date.now()}.${ext}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function getChartThemeConfig(theme) {
  const isDark = theme === "dark";
  return {
    gridColor: isDark ? "rgba(255, 255, 255, 0.07)" : "rgba(15, 23, 42, 0.06)",
    textColor: isDark ? "#94A3B8" : "#475569",
    tooltipBg: isDark ? "#161D2C" : "#FFFFFF",
    tooltipTitle: isDark ? "#F8FAFC" : "#0F172A",
    tooltipBody: isDark ? "#CBD5E1" : "#334155",
    tooltipBorder: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(15, 23, 42, 0.12)",
  };
}

function formatMetricValue(val, unit, isPercent) {
  if (val === null || val === undefined) return "—";
  if (isPercent) return `${val}%`;
  if (typeof val !== "number") return String(val);

  const unitStr = unit || "USD";
  const isBillions = /billion/i.test(unitStr);
  const suffix = isBillions ? "B" : "";
  if (unitStr.startsWith("USD")) {
    return `$${val.toLocaleString()}${suffix}`;
  }
  const currencyCode = unitStr.replace(/\s*\(Billions?\)/i, "").trim();
  return currencyCode ? `${val.toLocaleString()}${suffix} ${currencyCode}` : `${val.toLocaleString()}${suffix}`;
}

export function createChartController(els) {
  const {
    canvasChartEl,
    chartTypeToggleBtn,
    exportChartBtn,
    exportCanvasCsvBtn,
    exportCanvasJsonBtn,
    canvasPane,
    paneResizer,
    chatPane,
    toggleCanvasBtn,
    canvasEmptyState,
    canvasActiveContent,
  } = els;

  let financialChart = null;
  let chartMode = "combo"; // "combo", "bar", "line"
  const activeChartData = defaultChartData;
  let canvasSortColIdx = null;
  let canvasSortAsc = true;

  function renderCustomChartLegend() {
    const legendContainer = document.getElementById("custom-chart-legend");
    if (!legendContainer || !financialChart) return;

    legendContainer.innerHTML = financialChart.data.datasets
      .map((ds, idx) => {
        const isVisible = financialChart.isDatasetVisible(idx);
        const isLine = ds.type === "line" || ds.yAxisID === "y1";
        const color = ds.borderColor || ds.backgroundColor || "#00A3FF";
        const badgeHtml = isLine
          ? `<span class="legend-line" style="background-color: ${color};"></span>`
          : `<span class="legend-box" style="background-color: ${color};"></span>`;

        return `
          <div class="legend-item ${isVisible ? "" : "hidden"}" data-dataset-index="${idx}" title="Bấm để ẩn / hiện ${escapeHtml(ds.label)}">
            ${badgeHtml}
            <span>${escapeHtml(ds.label)}</span>
          </div>
        `;
      })
      .join("");
  }

  function updateChartThemeOptions(theme) {
    if (!financialChart) return;
    const themeCfg = getChartThemeConfig(theme);

    financialChart.options.scales.x.ticks.color = themeCfg.textColor;
    financialChart.options.scales.y.ticks.color = themeCfg.textColor;
    financialChart.options.scales.y.title.color = themeCfg.textColor;
    financialChart.options.scales.y.grid.color = themeCfg.gridColor;
    financialChart.options.scales.y1.ticks.color = themeCfg.textColor;
    financialChart.options.scales.y1.title.color = themeCfg.textColor;

    financialChart.options.plugins.tooltip.backgroundColor = themeCfg.tooltipBg;
    financialChart.options.plugins.tooltip.titleColor = themeCfg.tooltipTitle;
    financialChart.options.plugins.tooltip.bodyColor = themeCfg.tooltipBody;
    financialChart.options.plugins.tooltip.borderColor = themeCfg.tooltipBorder;
  }

  function updateFinancialGridTable(labels, seriesList, unit) {
    const tableEl = document.getElementById("financial-data-table");
    if (!tableEl) return;
    const theadTr = tableEl.querySelector("thead tr");
    const tbody = document.getElementById("financial-table-body");
    if (!theadTr || !tbody) return;

    if (!labels || !seriesList || !Array.isArray(seriesList) || seriesList.length === 0) return;

    let theadHtml = `<th class="col-metric" title="Bấm để sắp xếp">Chỉ tiêu / Doanh nghiệp <span class="canvas-sort-arrow"></span></th>`;
    labels.forEach((l) => {
      theadHtml += `<th class="col-num" title="Bấm để sắp xếp">${escapeHtml(String(l))} <span class="canvas-sort-arrow"></span></th>`;
    });
    theadTr.innerHTML = theadHtml;

    const isPercent = unit && unit.includes("%");
    tbody.innerHTML = seriesList
      .map((s) => {
        const name = s.label || s.name || "Chỉ tiêu";
        const rawValues = s.values !== undefined ? s.values : s.data;
        const sIsPercent =
          isPercent ||
          name.toLowerCase().includes("margin") ||
          name.toLowerCase().includes("biên") ||
          name.toLowerCase().includes("ratio");

        let cells = `<td class="metric-name">${escapeHtml(name)}</td>`;
        labels.forEach((_, idx) => {
          const val = rawValues && rawValues[idx] !== undefined ? rawValues[idx] : null;
          const formatted = formatMetricValue(val, unit, sIsPercent);
          cells += `<td class="num-val">${escapeHtml(String(formatted))}</td>`;
        });
        return `<tr>${cells}</tr>`;
      })
      .join("");
  }

  function initCanvasTableSorting() {
    const tableEl = document.getElementById("financial-data-table");
    if (!tableEl) return;
    const thead = tableEl.querySelector("thead");
    if (!thead) return;

    thead.addEventListener("click", (e) => {
      const th = e.target.closest("th");
      if (!th) return;
      const thList = Array.from(thead.querySelectorAll("th"));
      const colIdx = thList.indexOf(th);
      if (colIdx < 0) return;

      if (canvasSortColIdx === colIdx) {
        canvasSortAsc = !canvasSortAsc;
      } else {
        canvasSortColIdx = colIdx;
        canvasSortAsc = true;
      }

      thList.forEach((h, idx) => {
        let arrow = h.querySelector(".canvas-sort-arrow");
        if (!arrow) {
          arrow = document.createElement("span");
          arrow.className = "canvas-sort-arrow";
          h.appendChild(arrow);
        }
        if (idx === colIdx) {
          h.classList.add("active-sort");
          arrow.textContent = canvasSortAsc ? " ↑" : " ↓";
        } else {
          h.classList.remove("active-sort");
          arrow.textContent = "";
        }
      });

      const tbody = tableEl.querySelector("tbody");
      if (!tbody) return;
      const rows = Array.from(tbody.querySelectorAll("tr"));

      rows.sort((rowA, rowB) => {
        const cellA = rowA.children[colIdx];
        const cellB = rowB.children[colIdx];
        if (!cellA || !cellB) return 0;

        const textA = cellA.textContent.trim();
        const textB = cellB.textContent.trim();

        const cleanA = textA.replace(/[$%+,]/g, "").replace(/—|--/g, "NaN");
        const cleanB = textB.replace(/[$%+,]/g, "").replace(/—|--/g, "NaN");

        const numA = parseFloat(cleanA);
        const numB = parseFloat(cleanB);

        const isNumA = !isNaN(numA);
        const isNumB = !isNaN(numB);

        if (isNumA && isNumB) {
          return canvasSortAsc ? numA - numB : numB - numA;
        }
        return canvasSortAsc ? textA.localeCompare(textB) : textB.localeCompare(textA);
      });

      rows.forEach((r) => tbody.appendChild(r));
    });
  }

  function init() {
    if (!canvasChartEl) return;
    const ctx = canvasChartEl.getContext("2d");
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const themeCfg = getChartThemeConfig(currentTheme);

    const datasets = activeChartData.series.map((s) => {
      const isLine = s.type === "line" || chartMode === "line";
      const isMargin = s.name.toLowerCase().includes("margin") || s.name.toLowerCase().includes("biên");
      return {
        label: s.name,
        data: s.data,
        backgroundColor: s.color,
        borderColor: s.color,
        type: isLine ? "line" : "bar",
        yAxisID: isMargin ? "y1" : "y",
        hidden: isMargin,
        borderRadius: 4,
        borderWidth: isLine ? 2.5 : 0,
        pointRadius: isLine ? 4 : 0,
        tension: 0.3,
        barPercentage: 0.65,
        categoryPercentage: 0.7,
      };
    });

    financialChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: activeChartData.categories,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: themeCfg.tooltipBg,
            titleColor: themeCfg.tooltipTitle,
            bodyColor: themeCfg.tooltipBody,
            borderColor: themeCfg.tooltipBorder,
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label(context) {
                if (context.dataset.yAxisID === "y1") {
                  return `${context.dataset.label}: ${context.raw}%`;
                }
                return `${context.dataset.label}: $${context.raw?.toLocaleString()}M`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: themeCfg.textColor,
              font: { family: "'JetBrains Mono', monospace", size: 12, weight: 600 },
            },
          },
          y: {
            type: "linear",
            position: "left",
            title: {
              display: true,
              text: "Doanh thu (USD)",
              color: themeCfg.textColor,
              font: { size: 11, weight: 500 },
            },
            grid: { color: themeCfg.gridColor },
            ticks: {
              color: themeCfg.textColor,
              callback: (value) => (value >= 1000 ? `${value / 1000}k` : value),
              font: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
          y1: {
            type: "linear",
            position: "right",
            display: false,
            suggestedMin: 0,
            title: {
              display: true,
              text: "Biên hoạt động (%)",
              color: themeCfg.textColor,
              font: { size: 11, weight: 500 },
            },
            grid: { drawOnChartArea: false },
            ticks: {
              color: themeCfg.textColor,
              callback: (value) => `${value}%`,
              font: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
        },
      },
    });

    renderCustomChartLegend();
    initCanvasTableSorting();

    const customLegendEl = document.getElementById("custom-chart-legend");
    if (customLegendEl) {
      customLegendEl.addEventListener("click", (e) => {
        const item = e.target.closest(".legend-item");
        if (!item || !financialChart) return;
        const idx = parseInt(item.getAttribute("data-dataset-index"), 10);
        if (isNaN(idx) || !financialChart.data.datasets[idx]) return;

        const nextVisible = !financialChart.isDatasetVisible(idx);
        financialChart.setDatasetVisibility(idx, nextVisible);

        const ds = financialChart.data.datasets[idx];
        if (ds && ds.yAxisID === "y1" && financialChart.options.scales.y1) {
          financialChart.options.scales.y1.display = nextVisible;
        }

        financialChart.update();
        item.classList.toggle("hidden", !nextVisible);
      });
    }

    if (chartTypeToggleBtn) {
      chartTypeToggleBtn.addEventListener("click", () => {
        if (!financialChart) return;
        chartMode = chartMode === "combo" ? "line" : "combo";
        const isLine = chartMode === "line";

        financialChart.data.datasets.forEach((ds) => {
          if (!ds.label.toLowerCase().includes("biên") && !ds.label.toLowerCase().includes("margin")) {
            ds.type = isLine ? "line" : "bar";
            ds.borderWidth = isLine ? 2.5 : 0;
            ds.pointRadius = isLine ? 4 : 0;
          }
        });

        financialChart.update();
      });
    }

    if (exportChartBtn) {
      exportChartBtn.addEventListener("click", () => {
        if (!financialChart) return;
        const link = document.createElement("a");
        link.download = `financial_canvas_${Date.now()}.png`;
        link.href = financialChart.toBase64Image();
        link.click();
      });
    }

    if (exportCanvasCsvBtn) {
      exportCanvasCsvBtn.addEventListener("click", () => {
        const tableEl = document.getElementById("financial-data-table");
        if (tableEl) exportTableData(tableEl, "csv", "canvas_financial_metrics");
      });
    }

    if (exportCanvasJsonBtn) {
      exportCanvasJsonBtn.addEventListener("click", () => {
        const tableEl = document.getElementById("financial-data-table");
        if (tableEl) exportTableData(tableEl, "json", "canvas_financial_metrics");
      });
    }

    if (toggleCanvasBtn) {
      toggleCanvasBtn.addEventListener("click", toggle);
    }
  }

  function updateWithAgentChart(chartSpec) {
    if (!financialChart || !chartSpec) return;

    const labels = chartSpec.x_labels || chartSpec.categories;
    const seriesList = chartSpec.datasets || chartSpec.series;
    if (!labels || !seriesList || !Array.isArray(seriesList) || seriesList.length === 0) return;

    financialChart.data.labels = labels;
    const palette = ["#00A3FF", "#00E599", "#F5A623", "#8B5CF6", "#EF4444", "#EC4899", "#14B8A6"];
    const isChartPercent = chartSpec.unit === "%" || (chartSpec.title && chartSpec.title.includes("(%)"));

    financialChart.data.datasets = seriesList.map((s, idx) => {
      const name = s.label || s.name || `Series ${idx + 1}`;
      const rawValues = s.values !== undefined ? s.values : s.data;
      const isMargin =
        isChartPercent ||
        name.toLowerCase().includes("margin") ||
        name.toLowerCase().includes("biên") ||
        name.toLowerCase().includes("ratio");
      const color = s.color || palette[idx % palette.length];
      const shouldHide = !isChartPercent && isMargin;

      return {
        label: name,
        data: rawValues,
        backgroundColor: color,
        borderColor: color,
        type: isMargin
          ? chartMode === "bar" && isChartPercent
            ? "bar"
            : "line"
          : chartMode === "line"
            ? "line"
            : "bar",
        yAxisID: isChartPercent ? "y1" : isMargin ? "y1" : "y",
        hidden: shouldHide,
        borderRadius: 4,
        borderWidth: isMargin || chartMode === "line" ? 2.5 : 0,
        pointRadius: isMargin || chartMode === "line" ? 4 : 0,
        tension: 0.3,
      };
    });

    if (financialChart.options.scales.y) {
      financialChart.options.scales.y.display = !isChartPercent;
      if (financialChart.options.scales.y.title) {
        financialChart.options.scales.y.title.text = chartSpec.unit || "USD";
      }
    }
    if (financialChart.options.scales.y1) {
      financialChart.options.scales.y1.display =
        isChartPercent || financialChart.data.datasets.some((d) => d.yAxisID === "y1" && !d.hidden);
      if (financialChart.options.scales.y1.title) {
        financialChart.options.scales.y1.title.text = isChartPercent ? chartSpec.unit || "%" : "Biên hoạt động (%)";
      }
    }

    if (financialChart.options.plugins && financialChart.options.plugins.tooltip) {
      financialChart.options.plugins.tooltip.callbacks.label = function (context) {
        const val = context.raw;
        if (val === null || val === undefined) return `${context.dataset.label}: N/A`;
        const usePercent = context.dataset.yAxisID === "y1" || isChartPercent;
        const formatted = formatMetricValue(val, chartSpec.unit, usePercent);
        return `${context.dataset.label}: ${formatted}`;
      };
    }

    financialChart.update();
    renderCustomChartLegend();
    updateFinancialGridTable(labels, seriesList, chartSpec.unit);
  }

  function collapse() {
    if (canvasPane) canvasPane.classList.add("collapsed");
    if (paneResizer) paneResizer.classList.add("hidden");
    if (chatPane) chatPane.classList.add("canvas-hidden");
    if (toggleCanvasBtn) toggleCanvasBtn.classList.remove("active");
  }

  function expand() {
    if (canvasPane) canvasPane.classList.remove("collapsed");
    if (paneResizer) paneResizer.classList.remove("hidden");
    if (chatPane) {
      chatPane.classList.remove("canvas-hidden");
      chatPane.style.width = "";
      chatPane.style.flex = "1.1";
    }
    if (canvasPane) {
      canvasPane.style.width = "";
      canvasPane.style.flex = "0.9";
    }
    if (toggleCanvasBtn) toggleCanvasBtn.classList.add("active");
    setTimeout(() => {
      if (financialChart) financialChart.resize();
    }, 120);
  }

  function toggle() {
    if (canvasPane && canvasPane.classList.contains("collapsed")) {
      expand();
    } else {
      collapse();
    }
  }

  function showEmptyState(show) {
    if (canvasEmptyState && canvasActiveContent) {
      canvasEmptyState.style.display = show ? "flex" : "none";
      canvasActiveContent.style.display = show ? "none" : "block";
    }
  }

  function applyTheme(theme) {
    if (financialChart) {
      updateChartThemeOptions(theme);
      financialChart.update();
      renderCustomChartLegend();
    }
  }

  function resize() {
    if (financialChart) financialChart.resize();
  }

  return {
    init,
    resize,
    applyTheme,
    updateWithAgentChart,
    collapse,
    expand,
    toggle,
    showEmptyState,
  };
}
