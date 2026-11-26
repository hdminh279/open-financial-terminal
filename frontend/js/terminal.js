/**
 * Corporate Filing Intelligence - Modern Institutional AI Financial Terminal
 * Main ES module orchestrating the 3-pane workstation, SSE chat streaming,
 * citation preview drawer, session management, theme switcher, and watchlist handoff.
 */

import {
  deleteHistorySession,
  escapeHtml,
  fetchHistorySession,
  fetchHistorySessions,
  generateUUID,
  sanitizeUrl,
  streamAgentQuery,
} from "./api.js";
import { createChartController, defaultChartData, exportTableData } from "./chart.js";
import { initWatchlistController } from "./watchlist.js";

function formatCitationDisplay(c) {
  const isWeb = c.source_type === "web" || c.source_type === "EvidenceSourceType.WEB" || c.form === "WEB_NEWS";
  const isVn =
    c.form === "BCTC" || (c.source_url && (c.source_url.includes("vndirect") || c.source_url.includes("vietstock")));

  let domain = "";
  if (c.source_url) {
    try {
      const u = new URL(c.source_url);
      domain = u.hostname.replace(/^www\./i, "");
    } catch (_e) {
      domain = "";
    }
  }

  let company = "";
  if (isWeb) {
    company = domain || "Tin tức Web";
  } else {
    const rawTicker = c.ticker && c.ticker !== "WEB" ? c.ticker : "";
    company = c.company_name || rawTicker || (isVn ? "BCTC Việt Nam" : "Nguồn kiểm chứng");
  }

  const form = isWeb ? "Live Web" : c.form || "Filing";

  let periodLabel = "";
  if (!isWeb) {
    if (c.fiscal_period && c.fiscal_year) {
      periodLabel = c.fiscal_period === "FY" ? `FY${c.fiscal_year}` : `${c.fiscal_period} FY${c.fiscal_year}`;
    } else if (c.fiscal_year) {
      periodLabel = `FY${c.fiscal_year}`;
    } else if (c.fiscal_period) {
      periodLabel = c.fiscal_period;
    }
  }

  const url =
    c.source_url ||
    (c.cik && !String(c.cik).startsWith("VN:") ? `https://www.sec.gov/edgar/browse/?CIK=${c.cik}` : "#");
  return { company, form, periodLabel, url };
}

function translateIntentName(intent) {
  const map = {
    FINANCIAL_METRIC: "Số liệu BCTC",
    TEXTUAL_RESEARCH: "Phân tích hồ sơ",
    MIXED_RESEARCH: "Nghiên cứu tổng hợp",
    DEEP_RESEARCH_REPORT: "Báo cáo chuyên sâu",
    OUT_OF_SCOPE: "Ngoài phạm vi",
  };
  return map[intent] || intent;
}

function translateStepName(stepName) {
  if (!stepName) return "";
  const map = {
    "Understanding question": "Hiểu câu hỏi & Lập kế hoạch",
    "Resolving metrics": "Xác định chỉ tiêu tài chính",
    "Querying DuckDB": "Truy vấn DuckDB Lakehouse",
    "Searching SEC filings": "Tìm kiếm tài liệu SEC EDGAR",
    "Searching live web": "Tìm kiếm Web thời sự",
    "Validating evidence": "Xác thực bằng chứng",
    "Preparing visualization": "Khởi tạo biểu đồ trực quan",
  };
  return map[stepName] || stepName;
}

document.addEventListener("DOMContentLoaded", () => {
  // Core DOM elements
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const messagesContainer = document.getElementById("messages-container");
  const currentSessionTitle = document.getElementById("current-session-title");
  const sessionList = document.getElementById("session-list");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const sidebar = document.getElementById("sidebar");

  // Pane resizer and canvas elements
  const paneResizer = document.getElementById("pane-resizer");
  const chatPane = document.getElementById("chat-pane");
  const canvasPane = document.getElementById("canvas-pane");
  const workstationContainer = document.querySelector(".workstation-container");
  const toggleCanvasBtn = document.getElementById("toggle-canvas-btn");
  const canvasEmptyState = document.getElementById("canvas-empty-state");
  const canvasActiveContent = document.getElementById("canvas-active-content");

  // Theme elements
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const themeSunIcon = document.getElementById("theme-sun-icon");
  const themeMoonIcon = document.getElementById("theme-moon-icon");
  const themeBtnLight = document.getElementById("theme-btn-light");
  const themeBtnDark = document.getElementById("theme-btn-dark");

  // Settings elements
  const userProfileTrigger = document.getElementById("user-profile-trigger");
  const settingsModal = document.getElementById("settings-modal");
  const closeSettingsBtn = document.getElementById("close-settings-btn");
  const saveSettingsBtn = document.getElementById("save-settings-btn");
  const settingUserName = document.getElementById("setting-user-name");
  const userDisplayName = document.getElementById("user-display-name");

  // Citation Preview Drawer elements
  const citationDrawer = document.getElementById("citation-drawer");
  const citationDrawerBackdrop = document.getElementById("citation-drawer-backdrop");
  const citationDrawerClose = document.getElementById("citation-drawer-close");

  // Application state
  let currentSessionId = generateUUID();
  let isDraggingResizer = false;
  const turnCitationStore = [];

  const chartController = createChartController({
    canvasChartEl: document.getElementById("financial-canvas-chart"),
    chartTypeToggleBtn: document.getElementById("chart-type-toggle-btn"),
    exportChartBtn: document.getElementById("export-chart-btn"),
    exportCanvasCsvBtn: document.getElementById("export-canvas-csv-btn"),
    exportCanvasJsonBtn: document.getElementById("export-canvas-json-btn"),
    canvasPane,
    paneResizer,
    chatPane,
    toggleCanvasBtn,
    canvasEmptyState,
    canvasActiveContent,
  });

  function scrollToBottom() {
    setTimeout(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 50);
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("terminal_theme", theme);
    localStorage.setItem("app_theme", theme);

    if (theme === "dark") {
      themeSunIcon.style.display = "block";
      themeMoonIcon.style.display = "none";
      if (themeBtnDark) themeBtnDark.classList.add("active");
      if (themeBtnLight) themeBtnLight.classList.remove("active");
    } else {
      themeSunIcon.style.display = "none";
      themeMoonIcon.style.display = "block";
      if (themeBtnLight) themeBtnLight.classList.add("active");
      if (themeBtnDark) themeBtnDark.classList.remove("active");
    }

    chartController.applyTheme(theme);
  }

  function applyUserName(name) {
    if (userDisplayName) userDisplayName.textContent = name;
    if (settingUserName) settingUserName.value = name;
    localStorage.setItem("app_user_name", name);
  }

  function openCitationDrawer(citation) {
    if (!citationDrawer || !citationDrawerBackdrop || !citation) return;
    const citIdEl = document.getElementById("drawer-cit-id");
    const companyEl = document.getElementById("drawer-cit-company");
    const metaEl = document.getElementById("drawer-cit-meta");
    const itemEl = document.getElementById("drawer-cit-item");
    const accRow = document.getElementById("drawer-accession-row");
    const accEl = document.getElementById("drawer-cit-accession");
    const excerptEl = document.getElementById("drawer-cit-excerpt");
    const refEl = document.getElementById("drawer-cit-ref");
    const linkEl = document.getElementById("citation-drawer-link");

    const citNum = citation.citation_id || 1;
    const display = formatCitationDisplay(citation);
    const metaStr = [display.form, display.periodLabel].filter(Boolean).join(" · ");

    if (citIdEl) citIdEl.textContent = `[${citNum}]`;
    if (companyEl) companyEl.textContent = display.company;
    if (metaEl) metaEl.textContent = metaStr || "Official Source";
    if (itemEl) itemEl.textContent = citation.item_or_metric || "Tài liệu đối soát";

    if (accRow && accEl) {
      if (citation.accession_number) {
        accRow.style.display = "flex";
        accEl.textContent = citation.accession_number;
      } else {
        accRow.style.display = "none";
      }
    }

    if (excerptEl) {
      excerptEl.textContent =
        citation.excerpt ||
        citation.text ||
        citation.snippet ||
        citation.formatted_reference ||
        "Trích dẫn đã được kiểm chứng từ hệ thống.";
    }
    if (refEl) {
      refEl.textContent = citation.formatted_reference || citation.source_url || "";
    }
    if (linkEl) {
      linkEl.href = sanitizeUrl(display.url);
    }

    citationDrawerBackdrop.style.display = "block";
    citationDrawer.classList.add("open");
    citationDrawer.setAttribute("aria-hidden", "false");
  }

  function closeCitationDrawer() {
    if (!citationDrawer || !citationDrawerBackdrop) return;
    citationDrawer.classList.remove("open");
    citationDrawer.setAttribute("aria-hidden", "true");
    citationDrawerBackdrop.style.display = "none";
  }

  function renderWelcomeHero() {
    if (!messagesContainer) return;
    messagesContainer.innerHTML = `
      <div class="chat-welcome-hero" id="chat-welcome-hero">
        <div class="welcome-badge">
          <span class="welcome-sparkle">✦</span>
          <span>AI Financial Terminal</span>
        </div>
        <p class="welcome-subtitle">Tra cứu số liệu BCTC chuẩn hóa, phân tích thuyết minh 10-K/10-Q và đối soát trích dẫn gốc.</p>
        <div class="quick-suggestions-row">
          <button type="button" class="action-chip" data-query="So sánh doanh thu AAPL, MSFT và NVDA 3 năm gần nhất">So sánh AAPL, MSFT &amp; NVDA</button>
          <button type="button" class="action-chip" data-query="Phân tích rủi ro chuỗi cung ứng của Apple trong báo cáo 10-K">Rủi ro chuỗi cung ứng Apple</button>
          <button type="button" class="action-chip" data-query="So sánh doanh thu và biên lợi nhuận FPT và Vinamilk">FPT vs Vinamilk</button>
          <button type="button" class="action-chip" data-query="Biên lợi nhuận hoạt động của Microsoft từ FY2022 đến FY2025">Biên hoạt động Microsoft</button>
          <button type="button" class="action-chip" data-query="Tin tức tài chính mới nhất về NVIDIA hôm nay">Tin tức NVIDIA hôm nay</button>
        </div>
      </div>
    `;
  }

  function startNewSession() {
    currentSessionId = generateUUID();
    window.sessionCharts = [];
    turnCitationStore.length = 0;
    closeCitationDrawer();
    localStorage.removeItem("current_active_session_id");
    localStorage.setItem("current_session_id", currentSessionId);
    renderWelcomeHero();
    if (currentSessionTitle) {
      currentSessionTitle.textContent = "Đoạn chat mới";
    }
    chartController.collapse();
    chartController.showEmptyState(true);
    if (queryInput) {
      queryInput.value = "";
      queryInput.focus();
    }
    const allItems = sessionList ? sessionList.querySelectorAll(".session-item") : [];
    allItems.forEach((item) => item.classList.remove("active"));
  }

  function appendUserTurn(text) {
    const group = document.createElement("div");
    group.className = "message-group user-turn";
    group.innerHTML = `
      <div class="user-bubble">
        ${escapeHtml(text)}
      </div>
    `;
    messagesContainer.appendChild(group);
  }

  function appendAssistantSkeleton() {
    const loadingId = "loading-" + Date.now();
    const group = document.createElement("div");
    group.className = "message-group assistant-turn";
    group.id = loadingId;
    group.innerHTML = `
      <div class="avatar-col">
        <div class="avatar-icon assistant-avatar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
          </svg>
        </div>
      </div>
      <div class="content-col">
        <div class="agent-activity-card">
          <div class="activity-card-header">
            <div class="activity-header-left">
              <span class="activity-spark-icon">✦</span>
              <span class="activity-title">Đang phân tích &amp; truy vấn dữ liệu...</span>
            </div>
          </div>
        </div>
        <div class="assistant-response-text streaming-answer-preview" style="display: none;"></div>
      </div>
    `;
    messagesContainer.appendChild(group);
    return loadingId;
  }

  function renderAssistantTurn(data) {
    const group = document.createElement("div");
    group.className = "message-group assistant-turn";

    const turnIdx = turnCitationStore.length;
    turnCitationStore.push(data.citations || []);

    const citationMap = {};
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((c) => {
        if (c.citation_id) {
          citationMap[c.citation_id] = c;
        }
      });
    }

    let cleanAnswer = (data.answer || "")
      .replace(/^\*\*Abstention Notice\*\*:\s*/i, "")
      .replace(/```(?:mermaid)?\s*xychart-beta[\s\S]*?```/gi, "")
      .replace(/\n+(?:#{1,3}\s+)?(?:Financial Visualizations|Visualizations|Biểu đồ|Sources & References)[\s\S]*$/gi, "")
      .replace(/^\s*(?:---+|\*\*\*+|___+)\s*$/gm, "")
      .replace(/^(#{1,4}\s+)\[([^\]]+)\]\s*$/gm, "$1$2")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

    if (!cleanAnswer && data.abstention && data.abstention.explanation) {
      cleanAnswer = data.abstention.explanation;
    }

    // Escape tildes so marked does not misinterpret financial approximations (~383) as strikethrough
    cleanAnswer = cleanAnswer.replace(/~/g, "\\~");

    const rawHtml = marked.parse(cleanAnswer);
    let formattedAnswer =
      typeof DOMPurify !== "undefined"
        ? DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true }, ADD_ATTR: ["target", "rel"] })
        : rawHtml;

    formattedAnswer = formattedAnswer.replace(/<table>([\s\S]*?)<\/table>/gi, (match) => {
      return `
        <div class="table-card-container">
          <div class="table-export-toolbar">
            <span class="table-toolbar-label">Bảng số liệu</span>
            <div class="table-export-actions">
              <button type="button" class="btn-mini-export btn-table-export" data-format="csv" title="Xuất bảng ra CSV">CSV</button>
              <button type="button" class="btn-mini-export btn-table-export" data-format="json" title="Xuất bảng ra JSON">JSON</button>
            </div>
          </div>
          <div class="table-scroll-wrapper">${match}</div>
        </div>
      `;
    });

    formattedAnswer = formattedAnswer.replace(/\[(\d+)\]/g, (_match, p1) => {
      const cit = citationMap[p1];
      if (cit) {
        return `<button type="button" class="citation-tag" data-turn-idx="${turnIdx}" data-cit-id="${p1}" title="Xem trước trích dẫn [${p1}]">[${p1}]</button>`;
      }
      return `<span class="citation-tag">[${p1}]</span>`;
    });

    let activityStepsHtml = "";
    if (data.activity_timeline && data.activity_timeline.length > 0) {
      activityStepsHtml = data.activity_timeline
        .map((step) => {
          const icon = step.status === "completed" ? "✔" : step.status === "failed" ? "✕" : "○";
          const descHtml = step.description ? `<span class="step-desc">${escapeHtml(step.description)}</span>` : "";
          const durHtml =
            step.duration_ms !== null && step.duration_ms !== undefined && step.duration_ms > 0
              ? `<span class="step-duration">${Math.round(step.duration_ms)} ms</span>`
              : "";
          return `
            <div class="activity-step-row ${escapeHtml(step.status || "completed")}">
              <span class="step-check-badge">${icon}</span>
              <div class="step-main-info">
                <span class="step-label">${escapeHtml(translateStepName(step.step))}</span>
                ${descHtml}
              </div>
              ${durHtml}
            </div>
          `;
        })
        .join("");
    } else {
      activityStepsHtml = `
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><div class="step-main-info"><span class="step-label">Hiểu câu hỏi &amp; Lập kế hoạch</span></div></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><div class="step-main-info"><span class="step-label">Truy vấn DuckDB Lakehouse</span></div></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><div class="step-main-info"><span class="step-label">Tìm kiếm tài liệu SEC EDGAR</span></div></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><div class="step-main-info"><span class="step-label">Xác thực bằng chứng</span></div></div>
      `;
    }

    const intentBadge = data.intent
      ? `<span class="activity-intent-badge">${escapeHtml(translateIntentName(data.intent))}</span>`
      : "";
    const latencyBadge =
      data.execution_time_ms && data.execution_time_ms > 0
        ? `<span class="activity-latency-badge">${(data.execution_time_ms / 1000).toFixed(2)}s</span>`
        : "";

    let chartPreviewHtml = "";
    if (data.charts && data.charts.length > 0) {
      if (!window.sessionCharts) window.sessionCharts = [];
      const buttonsHtml = data.charts
        .map((chartSpec) => {
          const chartIdx = window.sessionCharts.length;
          window.sessionCharts.push(chartSpec);
          return `<button type="button" class="btn-view-canvas chart-preview-trigger" data-chart-idx="${chartIdx}">Xem biểu đồ</button>`;
        })
        .join("");
      chartPreviewHtml = `<div class="chart-preview-actions">${buttonsHtml}</div>`;
    }

    let sourcesAccordionHtml = "";
    if (data.citations && data.citations.length > 0) {
      const citationItems = data.citations
        .map((c, idx) => {
          const citNum = c.citation_id || idx + 1;
          const display = formatCitationDisplay(c);
          const excerpt =
            c.excerpt || c.text || c.snippet || c.formatted_reference || "Trích dẫn tài liệu kiểm chứng.";
          return `
            <div class="source-item-card">
              <div class="source-item-header">
                <span class="citation-badge">[${citNum}]</span>
                <span class="source-company">${escapeHtml(display.company)}</span>
                <span class="source-badge">${escapeHtml(display.form)}</span>
                ${display.periodLabel ? `<span class="source-badge">${escapeHtml(display.periodLabel)}</span>` : ""}
              </div>
              <div class="source-excerpt">${escapeHtml(excerpt.length > 220 ? excerpt.substring(0, 220) + "..." : excerpt)}</div>
              <div class="source-item-actions">
                <button type="button" class="btn-preview-citation" data-turn-idx="${turnIdx}" data-cit-id="${citNum}">
                  Xem trước trích dẫn
                </button>
                <a href="${escapeHtml(sanitizeUrl(display.url))}" target="_blank" rel="noopener noreferrer" class="source-link">
                  <span>Mở nguồn gốc</span>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                </a>
              </div>
            </div>
          `;
        })
        .join("");

      sourcesAccordionHtml = `
        <div class="sources-accordion-card collapsed">
          <div class="sources-card-header" title="Bấm để mở rộng / thu gọn danh sách nguồn trích dẫn">
            <div class="sources-header-left">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
              </svg>
              <span class="sources-title">Nguồn &amp; Trích dẫn kiểm chứng: ${data.citations.length} nguồn</span>
            </div>
            <span class="sources-chevron-icon" aria-label="Thu phóng nguồn">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </span>
          </div>
          <div class="sources-list-content" style="display: none;">
            ${citationItems}
          </div>
        </div>
      `;
    }

    group.innerHTML = `
      <div class="avatar-col">
        <div class="avatar-icon assistant-avatar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
          </svg>
        </div>
      </div>

      <div class="content-col">
        <div class="agent-activity-card collapsed">
          <div class="activity-card-header">
            <div class="activity-header-left">
              <span class="activity-spark-icon">✦</span>
              <span class="activity-title">Tiến trình suy luận Agent</span>
              ${intentBadge}
              ${latencyBadge}
            </div>
            <span class="activity-chevron-icon" aria-label="Thu gọn / Mở rộng">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </span>
          </div>
          <div class="activity-steps-list" style="display: none;">
            ${activityStepsHtml}
          </div>
        </div>

        <div class="assistant-response-text">
          ${formattedAnswer}
        </div>

        ${chartPreviewHtml}

        ${sourcesAccordionHtml}
      </div>
    `;

    messagesContainer.appendChild(group);
  }

  function renderErrorMessage(msg) {
    const group = document.createElement("div");
    group.className = "message-group assistant-turn";
    group.innerHTML = `
      <div class="avatar-col">
        <div class="avatar-icon assistant-avatar" style="color: var(--accent-crimson);">⚠</div>
      </div>
      <div class="content-col">
        <div class="assistant-response-text" style="color: var(--accent-crimson);">
          <strong>Lỗi hệ thống:</strong> ${escapeHtml(msg)}
        </div>
      </div>
    `;
    messagesContainer.appendChild(group);
  }

  async function submitQuery(queryText) {
    if (!queryText) return;

    queryInput.value = "";
    queryInput.style.height = "auto";
    sendBtn.disabled = true;

    appendUserTurn(queryText);

    const loadingId = appendAssistantSkeleton();
    scrollToBottom();

    const loadingEl = document.getElementById(loadingId);
    const titleEl = loadingEl ? loadingEl.querySelector(".activity-title") : null;
    const previewEl = loadingEl ? loadingEl.querySelector(".streaming-answer-preview") : null;
    let streamedText = "";

    const applyStreamPreview = () => {
      if (!previewEl || !streamedText) return;
      previewEl.style.display = "block";
      const safeText = streamedText.replace(/~/g, "\\~");
      const rawHtml = marked.parse(safeText);
      previewEl.innerHTML =
        typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true } }) : rawHtml;
      scrollToBottom();
    };

    try {
      const data = await streamAgentQuery(queryText, currentSessionId, {
        onStatus: (payload) => {
          if (titleEl && payload.step) {
            titleEl.textContent =
              payload.step === "Synthesizing answer"
                ? "Đang tổng hợp câu trả lời..."
                : translateStepName(payload.step);
          }
        },
        onToken: (delta) => {
          streamedText += delta;
          applyStreamPreview();
        },
      });

      currentSessionId = data.session_id;
      localStorage.setItem("current_active_session_id", currentSessionId);
      localStorage.setItem("current_session_id", currentSessionId);

      if (currentSessionTitle && data.query) {
        const shortTitle = data.query.length > 48 ? data.query.substring(0, 48) + "..." : data.query;
        currentSessionTitle.textContent = shortTitle;
      }

      removeElement(loadingId);
      renderAssistantTurn(data);

      if (data.charts && data.charts.length > 0) {
        chartController.updateWithAgentChart(data.charts[0]);
        chartController.expand();
        chartController.showEmptyState(false);
      } else {
        chartController.collapse();
        chartController.showEmptyState(true);
      }

      loadRecentsSessions();
    } catch (err) {
      removeElement(loadingId);
      renderErrorMessage(err.message);
    } finally {
      sendBtn.disabled = false;
      scrollToBottom();
      queryInput.focus();
    }
  }

  async function loadRecentsSessions() {
    try {
      const data = await fetchHistorySessions();
      if (!data) return;
      const sessions = data.sessions || [];

      if (sessions.length === 0) {
        sessionList.innerHTML = "";
        return;
      }

      sessionList.innerHTML = "";
      sessions.forEach((s) => {
        const item = document.createElement("button");
        item.className = `session-item ${s.session_id === currentSessionId ? "active" : ""}`;
        item.setAttribute("data-session-id", s.session_id);

        const titleSpan = document.createElement("span");
        titleSpan.className = "session-title-text";
        titleSpan.textContent = s.title || "Đoạn chat tài chính";
        item.appendChild(titleSpan);

        const trashBtn = document.createElement("span");
        trashBtn.className = "session-trash-btn";
        trashBtn.title = "Xóa phiên này";
        trashBtn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        `;
        trashBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          try {
            await deleteHistorySession(s.session_id);
          } catch (_err) {
            // Ignore network deletion errors while cleaning UI state
          }
          item.remove();
          if (currentSessionId === s.session_id) {
            localStorage.removeItem("current_active_session_id");
            startNewSession();
          }
        });
        item.appendChild(trashBtn);

        item.addEventListener("click", () => {
          localStorage.setItem("current_active_session_id", s.session_id);
          loadSession(s.session_id);
        });
        sessionList.appendChild(item);
      });
    } catch (e) {
      console.warn("Could not load sessions from storage:", e);
    }
  }

  async function loadSession(sessionId) {
    try {
      const sessionData = await fetchHistorySession(sessionId);
      if (!sessionData) {
        startNewSession();
        return;
      }
      currentSessionId = sessionId;
      localStorage.setItem("current_active_session_id", currentSessionId);
      localStorage.setItem("current_session_id", currentSessionId);

      if (currentSessionTitle) {
        currentSessionTitle.textContent = sessionData.title || "Đoạn chat";
      }

      if (sessionData.messages && sessionData.messages.length > 0) {
        messagesContainer.innerHTML = "";
        window.sessionCharts = [];
        turnCitationStore.length = 0;

        let hasAnyChart = false;
        sessionData.messages.forEach((msg) => {
          if (msg.role === "user") {
            appendUserTurn(msg.content);
          } else {
            const meta = msg.metadata || {};
            const charts = msg.charts || meta.charts || [];
            renderAssistantTurn({
              answer: msg.content,
              citations: msg.citations || meta.citations || [],
              charts,
              activity_timeline: msg.activity_timeline || meta.activity_timeline || [],
              intent: msg.intent || meta.intent || null,
              execution_time_ms: msg.execution_time_ms ?? meta.execution_time_ms ?? null,
            });
            if (charts.length > 0) {
              hasAnyChart = true;
              chartController.updateWithAgentChart(charts[0]);
            }
          }
        });

        if (hasAnyChart) {
          chartController.expand();
          chartController.showEmptyState(false);
        } else {
          chartController.collapse();
          chartController.showEmptyState(true);
        }
      } else {
        renderWelcomeHero();
        chartController.collapse();
        chartController.showEmptyState(true);
      }
      scrollToBottom();
      const allItems = sessionList ? sessionList.querySelectorAll(".session-item") : [];
      allItems.forEach((item) => {
        item.classList.toggle("active", item.getAttribute("data-session-id") === currentSessionId);
      });
    } catch (e) {
      console.warn("Failed to load session details:", e);
      startNewSession();
    }
  }

  // Sidebar collapse toggle
  if (toggleSidebarBtn && sidebar) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
      setTimeout(() => chartController.resize(), 260);
    });
  }

  // Draggable split-pane resizer
  if (paneResizer && chatPane && canvasPane && workstationContainer) {
    paneResizer.addEventListener("mousedown", () => {
      isDraggingResizer = true;
      paneResizer.classList.add("is-dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDraggingResizer) return;
      const containerRect = workstationContainer.getBoundingClientRect();
      const sidebarWidth = sidebar.classList.contains("collapsed") ? 0 : sidebar.getBoundingClientRect().width;
      const availableWidth = containerRect.width - sidebarWidth;
      const mouseOffsetInWorkspace = e.clientX - containerRect.left - sidebarWidth;

      const chatPercent = (mouseOffsetInWorkspace / availableWidth) * 100;
      if (chatPercent >= 28 && chatPercent <= 72) {
        chatPane.style.flex = "none";
        chatPane.style.width = `${chatPercent}%`;
        canvasPane.style.flex = "1";
        canvasPane.style.width = "auto";
        chartController.resize();
      }
    });

    document.addEventListener("mouseup", () => {
      if (isDraggingResizer) {
        isDraggingResizer = false;
        paneResizer.classList.remove("is-dragging");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        chartController.resize();
      }
    });
  }

  // Delegated interactions inside chat feed
  if (messagesContainer) {
    messagesContainer.addEventListener("click", (e) => {
      const header = e.target.closest(".activity-card-header");
      if (header) {
        const card = header.closest(".agent-activity-card");
        if (!card) return;
        card.classList.toggle("collapsed");
        const stepsList = card.querySelector(".activity-steps-list");
        if (stepsList) {
          stepsList.style.display = card.classList.contains("collapsed") ? "none" : "flex";
        }
        return;
      }

      const previewCard = e.target.closest(".chart-preview-card, .chart-preview-trigger");
      if (previewCard) {
        chartController.expand();
        chartController.showEmptyState(false);
        const chartIdx = previewCard.getAttribute("data-chart-idx");
        if (chartIdx !== null && window.sessionCharts && window.sessionCharts[chartIdx]) {
          chartController.updateWithAgentChart(window.sessionCharts[chartIdx]);
        } else {
          chartController.updateWithAgentChart(defaultChartData);
        }
        return;
      }

      const sourcesHeader = e.target.closest(".sources-card-header");
      if (sourcesHeader) {
        const sourcesCard = sourcesHeader.closest(".sources-accordion-card");
        if (!sourcesCard) return;
        sourcesCard.classList.toggle("collapsed");
        const listContent = sourcesCard.querySelector(".sources-list-content");
        if (listContent) {
          listContent.style.display = sourcesCard.classList.contains("collapsed") ? "none" : "flex";
        }
        return;
      }

      const citTrigger = e.target.closest(".citation-tag[data-turn-idx], .btn-preview-citation[data-turn-idx]");
      if (citTrigger) {
        const turnIdx = parseInt(citTrigger.getAttribute("data-turn-idx"), 10);
        const citId = parseInt(citTrigger.getAttribute("data-cit-id"), 10);
        const turnCitations = turnCitationStore[turnIdx] || [];
        const matched = turnCitations.find((c) => Number(c.citation_id) === citId) || turnCitations[citId - 1];
        if (matched) {
          e.preventDefault();
          openCitationDrawer(matched);
        }
        return;
      }

      const exportBtn = e.target.closest(".btn-table-export");
      if (exportBtn) {
        const wrapper = exportBtn.closest(".table-card-container");
        const tableEl = wrapper ? wrapper.querySelector("table") : null;
        const format = exportBtn.getAttribute("data-format") || "csv";
        if (tableEl) {
          exportTableData(tableEl, format, "financial_table");
        }
      }
    });
  }

  // Theme & Settings modal bindings
  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });

  if (themeBtnLight && themeBtnDark) {
    themeBtnLight.addEventListener("click", () => applyTheme("light"));
    themeBtnDark.addEventListener("click", () => applyTheme("dark"));
  }

  userProfileTrigger.addEventListener("click", () => {
    settingsModal.style.display = "flex";
  });

  closeSettingsBtn.addEventListener("click", () => {
    settingsModal.style.display = "none";
  });

  saveSettingsBtn.addEventListener("click", () => {
    if (settingUserName.value.trim()) {
      applyUserName(settingUserName.value.trim());
    }
    settingsModal.style.display = "none";
  });

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = "none";
    }
  });

  if (citationDrawerClose) citationDrawerClose.addEventListener("click", closeCitationDrawer);
  if (citationDrawerBackdrop) citationDrawerBackdrop.addEventListener("click", closeCitationDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && citationDrawer && citationDrawer.classList.contains("open")) {
      closeCitationDrawer();
    }
  });

  // Suggestion chips
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".action-chip");
    if (chip) {
      const promptQuery = chip.getAttribute("data-query");
      if (promptQuery) submitQuery(promptQuery);
    }
  });

  // Chat prompt controls
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    const scrollH = queryInput.scrollHeight;
    if (scrollH > 120) {
      queryInput.style.height = "120px";
      queryInput.style.overflowY = "auto";
    } else {
      queryInput.style.height = scrollH + "px";
      queryInput.style.overflowY = "hidden";
    }
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (queryInput.value.trim()) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (query) submitQuery(query);
  });

  newChatBtn.addEventListener("click", startNewSession);

  const clearChatBtn = document.getElementById("clear-chat-btn");
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", startNewSession);
  }

  // Boot sequence
  applyTheme(localStorage.getItem("terminal_theme") || "dark");
  applyUserName(localStorage.getItem("app_user_name") || "User");
  chartController.init();

  initWatchlistController({
    onAskAi: (promptText) => {
      if (queryInput) {
        queryInput.value = promptText;
        if (chatForm) chatForm.dispatchEvent(new Event("submit"));
      }
    },
  });

  async function initializeAppSession() {
    await loadRecentsSessions();
    const savedActiveSessionId = localStorage.getItem("current_active_session_id");
    if (savedActiveSessionId) {
      await loadSession(savedActiveSessionId);
    } else {
      startNewSession();
    }
  }
  initializeAppSession();
});
