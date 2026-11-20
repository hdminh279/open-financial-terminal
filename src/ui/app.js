/**
 * Corporate Filing Intelligence - Modern Institutional AI Financial Terminal
 * Controller handling 3-Pane Workstation, Dynamic Resizer, Agent Accordion,
 * Right-Aligned User Messages, Responsive Markdown Tables, and Vietnamese Localization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const voiceBtn = document.getElementById("voice-btn");
  const messagesContainer = document.getElementById("messages-container");
  const currentSessionTitle = document.getElementById("current-session-title");
  const sessionList = document.getElementById("session-list");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const sidebar = document.getElementById("sidebar");

  // Pane Resizer
  const paneResizer = document.getElementById("pane-resizer");
  const chatPane = document.getElementById("chat-pane");
  const canvasPane = document.getElementById("canvas-pane");
  const workstationContainer = document.querySelector(".workstation-container");

  // Dynamic Canvas Visibility Elements
  const toggleCanvasBtn = document.getElementById("toggle-canvas-btn");
  const canvasEmptyState = document.getElementById("canvas-empty-state");
  const canvasActiveContent = document.getElementById("canvas-active-content");

  // Theme Elements
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const themeSunIcon = document.getElementById("theme-sun-icon");
  const themeMoonIcon = document.getElementById("theme-moon-icon");
  const themeBtnLight = document.getElementById("theme-btn-light");
  const themeBtnDark = document.getElementById("theme-btn-dark");

  // Settings Elements
  const userProfileTrigger = document.getElementById("user-profile-trigger");
  const settingsModal = document.getElementById("settings-modal");
  const closeSettingsBtn = document.getElementById("close-settings-btn");
  const saveSettingsBtn = document.getElementById("save-settings-btn");
  const settingUserName = document.getElementById("setting-user-name");
  const userDisplayName = document.getElementById("user-display-name");

  // Canvas & Chart Elements
  const canvasChartEl = document.getElementById("financial-canvas-chart");
  const chartTypeToggleBtn = document.getElementById("chart-type-toggle-btn");
  const exportChartBtn = document.getElementById("export-chart-btn");
  const financialTableBody = document.getElementById("financial-table-body");
  const activeAccessionNum = document.getElementById("active-accession-num");

  // Application State
  let currentSessionId = generateUUID();
  let financialChart = null;
  let chartMode = "combo"; // "combo", "bar", "line"
  let isRecording = false;
  let recognition = null;
  let isDraggingResizer = false;

  // Initial Multi-Year Comparison Dataset
  const defaultChartData = {
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
      }
    ]
  };

  let activeChartData = defaultChartData;

  // Initialize Theme (Default Dark Theme per User Requirement)
  const savedTheme = localStorage.getItem("terminal_theme") || "dark";
  applyTheme(savedTheme);

  // Initialize Profile Name
  const savedUserName = localStorage.getItem("app_user_name") || "Minh";
  applyUserName(savedUserName);

  // Initialize Dual-Axis Chart Canvas
  initFinancialCanvasChart();

  // Load Past Sessions from MinIO Lakehouse
  loadRecentsSessions();

  /* ========================================================================= */
  /* Sidebar Toggle (Ẩn / Hiện Cột Bên Trái)                                  */
  /* ========================================================================= */

  if (toggleSidebarBtn && sidebar) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
      setTimeout(() => {
        if (financialChart) financialChart.resize();
      }, 260);
    });
  }

  /* ========================================================================= */
  /* Draggable Resizer (Điều chỉnh độ rộng Chat và Canvas)                     */
  /* ========================================================================= */

  if (paneResizer && chatPane && canvasPane && workstationContainer) {
    paneResizer.addEventListener("mousedown", (e) => {
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
      // Limit between 28% and 72%
      if (chatPercent >= 28 && chatPercent <= 72) {
        chatPane.style.flex = "none";
        chatPane.style.width = `${chatPercent}%`;
        canvasPane.style.flex = "1";
        canvasPane.style.width = "auto";
        if (financialChart) financialChart.resize();
      }
    });

    document.addEventListener("mouseup", () => {
      if (isDraggingResizer) {
        isDraggingResizer = false;
        paneResizer.classList.remove("is-dragging");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        if (financialChart) financialChart.resize();
      }
    });
  }

  /* ========================================================================= */
  /* Dynamic Canvas Collapsible & Visibility Management                        */
  /* ========================================================================= */

  function collapseCanvas() {
    if (canvasPane) canvasPane.classList.add("collapsed");
    if (paneResizer) paneResizer.classList.add("hidden");
    if (chatPane) chatPane.classList.add("canvas-hidden");
    if (toggleCanvasBtn) toggleCanvasBtn.classList.remove("active");
  }

  function expandCanvas() {
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

  function toggleCanvas() {
    if (canvasPane && canvasPane.classList.contains("collapsed")) {
      expandCanvas();
    } else {
      collapseCanvas();
    }
  }

  function showCanvasEmptyState(show) {
    if (canvasEmptyState && canvasActiveContent) {
      if (show) {
        canvasEmptyState.style.display = "flex";
        canvasActiveContent.style.display = "none";
      } else {
        canvasEmptyState.style.display = "none";
        canvasActiveContent.style.display = "block";
      }
    }
  }

  if (toggleCanvasBtn) {
    toggleCanvasBtn.addEventListener("click", () => {
      toggleCanvas();
    });
  }

  /* ========================================================================= */
  /* Agent Activity Accordion, Sources Toggle & Chart Preview Delegation       */
  /* ========================================================================= */

  if (messagesContainer) {
    messagesContainer.addEventListener("click", (e) => {
      // 1. Agent Activity Accordion Toggle (Smooth SVG Rotate)
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

      // 2. Click on In-Chat Chart Preview Card / View Canvas Button
      const previewCard = e.target.closest(".chart-preview-card");
      if (previewCard) {
        expandCanvas();
        showCanvasEmptyState(false);
        const chartIdx = previewCard.getAttribute("data-chart-idx");
        if (chartIdx !== null && window.sessionCharts && window.sessionCharts[chartIdx]) {
          updateCanvasWithAgentChart(window.sessionCharts[chartIdx]);
        } else {
          updateCanvasWithAgentChart(defaultChartData);
        }
        return;
      }

      // 3. Sources Accordion Toggle (Thu phóng nguồn trích dẫn SEC)
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
    });
  }

  /* ========================================================================= */
  /* Theme Switching Controller (Dark / Light Mode)                            */
  /* ========================================================================= */

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

    if (financialChart) {
      updateChartThemeOptions(theme);
      financialChart.update();
      renderCustomChartLegend();
    }
  }

  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
  });

  if (themeBtnLight && themeBtnDark) {
    themeBtnLight.addEventListener("click", () => applyTheme("light"));
    themeBtnDark.addEventListener("click", () => applyTheme("dark"));
  }

  /* ========================================================================= */
  /* User Profile & Settings Modal                                             */
  /* ========================================================================= */

  function applyUserName(name) {
    if (userDisplayName) userDisplayName.textContent = name;
    if (settingUserName) settingUserName.value = name;
    localStorage.setItem("app_user_name", name);
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

  /* ========================================================================= */
  /* Dual-Axis Financial Canvas Chart (Chart.js)                               */
  /* ========================================================================= */

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

  function initFinancialCanvasChart() {
    if (!canvasChartEl) return;
    const ctx = canvasChartEl.getContext("2d");
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const themeCfg = getChartThemeConfig(currentTheme);

    const datasets = activeChartData.series.map(s => {
      const isLine = s.type === "line" || chartMode === "line";
      const isMargin = s.name.toLowerCase().includes("margin") || s.name.toLowerCase().includes("biên");
      return {
        label: s.name,
        data: s.data,
        backgroundColor: s.color,
        borderColor: s.color,
        type: isLine ? "line" : "bar",
        yAxisID: isMargin ? "y1" : "y",
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
        datasets: datasets,
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
              label: function(context) {
                if (context.dataset.yAxisID === "y1") {
                  return `${context.dataset.label}: ${context.raw}%`;
                }
                return `${context.dataset.label}: $${context.raw?.toLocaleString()}M`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: themeCfg.textColor,
              font: { family: "'JetBrains Mono', monospace", size: 12, weight: 600 }
            }
          },
          y: {
            type: "linear",
            position: "left",
            title: {
              display: true,
              text: "Doanh thu (USD)",
              color: themeCfg.textColor,
              font: { size: 11, weight: 500 }
            },
            grid: { color: themeCfg.gridColor },
            ticks: {
              color: themeCfg.textColor,
              callback: value => value >= 1000 ? `${value / 1000}k` : value,
              font: { family: "'JetBrains Mono', monospace", size: 11 }
            }
          },
          y1: {
            type: "linear",
            position: "right",
            suggestedMin: 0,
            title: {
              display: true,
              text: "Biên hoạt động (%)",
              color: themeCfg.textColor,
              font: { size: 11, weight: 500 }
            },
            grid: { drawOnChartArea: false },
            ticks: {
              color: themeCfg.textColor,
              callback: value => `${value}%`,
              font: { family: "'JetBrains Mono', monospace", size: 11 }
            }
          }
        }
      }
    });

    renderCustomChartLegend();
  }

  /* ========================================================================= */
  /* Interactive Chart Legend Controller                                       */
  /* ========================================================================= */

  function renderCustomChartLegend() {
    const legendContainer = document.getElementById("custom-chart-legend");
    if (!legendContainer || !financialChart) return;

    legendContainer.innerHTML = financialChart.data.datasets.map((ds, idx) => {
      const isVisible = financialChart.isDatasetVisible(idx);
      const isLine = ds.type === "line" || ds.yAxisID === "y1";
      const color = ds.borderColor || ds.backgroundColor || "#00A3FF";
      const badgeHtml = isLine
        ? `<span class="legend-line" style="background-color: ${color};"></span>`
        : `<span class="legend-box" style="background-color: ${color};"></span>`;

      return `
        <div class="legend-item ${isVisible ? '' : 'hidden'}" data-dataset-index="${idx}" title="Bấm để ẩn / hiện ${escapeHtml(ds.label)}">
          ${badgeHtml}
          <span>${escapeHtml(ds.label)}</span>
        </div>
      `;
    }).join("");
  }

  const customLegendEl = document.getElementById("custom-chart-legend");
  if (customLegendEl) {
    customLegendEl.addEventListener("click", (e) => {
      const item = e.target.closest(".legend-item");
      if (!item || !financialChart) return;
      const idx = parseInt(item.getAttribute("data-dataset-index"), 10);
      if (isNaN(idx) || !financialChart.data.datasets[idx]) return;

      const isVisible = financialChart.isDatasetVisible(idx);
      const nextVisible = !isVisible;
      financialChart.setDatasetVisibility(idx, nextVisible);

      // If toggling the secondary Y-axis series (e.g. Operating Margin / y1), toggle axis visibility accordingly
      const ds = financialChart.data.datasets[idx];
      if (ds && ds.yAxisID === "y1" && financialChart.options.scales.y1) {
        financialChart.options.scales.y1.display = nextVisible;
      }

      financialChart.update();
      item.classList.toggle("hidden", !nextVisible);
    });
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

  // Toggle Line / Bar mode
  if (chartTypeToggleBtn) {
    chartTypeToggleBtn.addEventListener("click", () => {
      if (!financialChart) return;
      chartMode = chartMode === "combo" ? "line" : "combo";
      const isLine = chartMode === "line";

      financialChart.data.datasets.forEach(ds => {
        if (!ds.label.toLowerCase().includes("biên") && !ds.label.toLowerCase().includes("margin")) {
          ds.type = isLine ? "line" : "bar";
          ds.borderWidth = isLine ? 2.5 : 0;
          ds.pointRadius = isLine ? 4 : 0;
        }
      });

      financialChart.update();
    });
  }

  // Export Chart to PNG
  if (exportChartBtn) {
    exportChartBtn.addEventListener("click", () => {
      if (!financialChart) return;
      const link = document.createElement("a");
      link.download = `financial_canvas_${Date.now()}.png`;
      link.href = financialChart.toBase64Image();
      link.click();
    });
  }

  /* ========================================================================= */
  /* Watchlist & Action Chips Prompt Binding                                   */
  /* ========================================================================= */

  const watchlistCards = document.querySelectorAll(".watchlist-item");
  watchlistCards.forEach(card => {
    card.addEventListener("click", () => {
      const ticker = card.getAttribute("data-ticker");
      const sampleQuery = `Phân tích tình hình tài chính và doanh thu của ${ticker} năm 2024 theo báo cáo tài chính`;
      submitQuery(sampleQuery);
    });
  });

  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("action-chip")) {
      const promptQuery = e.target.getAttribute("data-query");
      if (promptQuery) {
        submitQuery(promptQuery);
      }
    }
  });

  /* ========================================================================= */
  /* Chat Form & Query Submission                                              */
  /* ========================================================================= */

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
    if (query) {
      submitQuery(query);
    }
  });

  newChatBtn.addEventListener("click", () => {
    startNewSession();
  });

  const clearChatBtn = document.getElementById("clear-chat-btn");
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
      startNewSession();
    });
  }

  function startNewSession() {
    currentSessionId = generateUUID();
    localStorage.setItem("current_session_id", currentSessionId);
    messagesContainer.innerHTML = "";
    currentSessionTitle.textContent = "Đoạn chat mới";
    collapseCanvas();
    showCanvasEmptyState(true);
    queryInput.value = "";
    queryInput.focus();
  }

  async function submitQuery(queryText) {
    if (!queryText) return;

    queryInput.value = "";
    queryInput.style.height = "auto";
    sendBtn.disabled = true;

    // Append User Message to the RIGHT
    appendUserTurn(queryText);

    // Append Loading Skeleton
    const loadingId = appendAssistantSkeleton();
    scrollToBottom();

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText,
          session_id: currentSessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Lỗi máy chủ: ${response.status}`);
      }

      const data = await response.json();
      currentSessionId = data.session_id;
      localStorage.setItem("current_session_id", currentSessionId);

      if (currentSessionTitle && data.query) {
        const shortTitle = data.query.length > 48 ? data.query.substring(0, 48) + "..." : data.query;
        currentSessionTitle.textContent = shortTitle;
      }

      removeElement(loadingId);
      renderAssistantTurn(data);

      // CANVAS VISIBILITY MANAGEMENT
      if (data.charts && data.charts.length > 0) {
        updateCanvasWithAgentChart(data.charts[0]);
        expandCanvas();
        showCanvasEmptyState(false);
      } else {
        // When agents answer without chart, auto-hide canvas
        collapseCanvas();
        showCanvasEmptyState(true);
      }

      // Update Provenance Accession Number
      if (data.citations && data.citations.length > 0) {
        const firstCitation = data.citations[0];
        if (firstCitation.accession_number && activeAccessionNum) {
          activeAccessionNum.textContent = firstCitation.accession_number;
        }
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

  /* ========================================================================= */
  /* UI Rendering Utilities                                                    */
  /* ========================================================================= */

  // USER TURN: Aligned to the RIGHT
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

  // ASSISTANT SKELETON: Aligned to the LEFT
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
              <span class="activity-title">Đang phân tích & truy vấn dữ liệu...</span>
            </div>
          </div>
        </div>
      </div>
    `;
    messagesContainer.appendChild(group);
    return loadingId;
  }

  // ASSISTANT TURN: Aligned to the LEFT
  function renderAssistantTurn(data) {
    const group = document.createElement("div");
    group.className = "message-group assistant-turn";

    const citationMap = {};
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((c) => {
        if (c.citation_id && c.source_url) {
          citationMap[c.citation_id] = c.source_url;
        }
      });
    }

    let cleanAnswer = (data.answer || "")
      .replace(/^\*\*Abstention Notice\*\*:\s*/i, "")
      .replace(/```(?:mermaid)?\s*xychart-beta[\s\S]*?```/gi, "")
      .replace(/\n+(?:#{1,3}\s+)?(?:Financial Visualizations|Visualizations|Biểu đồ|Sources & References)[\s\S]*$/gi, "")
      .trim();

    if (!cleanAnswer && data.abstention && data.abstention.explanation) {
      cleanAnswer = data.abstention.explanation;
    }

    let formattedAnswer = marked.parse(cleanAnswer);

    // Wrap all <table> elements in <div class="table-scroll-wrapper"> to guarantee responsive horizontal scrolling
    formattedAnswer = formattedAnswer.replace(/<table>([\s\S]*?)<\/table>/gi, (match) => {
      return `<div class="table-scroll-wrapper">${match}</div>`;
    });

    // Replace [1], [2] with seamless citation pills
    formattedAnswer = formattedAnswer.replace(/\[(\d+)\]/g, (match, p1) => {
      const url = citationMap[p1];
      if (url) {
        return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="citation-tag">[${p1}]</a>`;
      }
      return `<span class="citation-tag">[${p1}]</span>`;
    });

    // Activity Steps in Vietnamese
    let activityStepsHtml = "";
    if (data.activity_timeline && data.activity_timeline.length > 0) {
      activityStepsHtml = data.activity_timeline.map(step => `
        <div class="activity-step-row ${step.status}">
          <span class="step-check-badge">${step.status === 'completed' ? '✔' : '○'}</span>
          <span class="step-label">${escapeHtml(translateStepName(step.step))}</span>
        </div>
      `).join("");
    } else {
      activityStepsHtml = `
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><span class="step-label">Bước 1: Phân loại ý định</span></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><span class="step-label">Bước 2: Truy vấn DuckDB SQL</span></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><span class="step-label">Bước 3: Tìm kiếm ngữ nghĩa Hybrid RAG</span></div>
        <div class="activity-step-row completed"><span class="step-check-badge">✔</span><span class="step-label">Bước 4: Đối soát & Kiểm chứng</span></div>
      `;
    }

    // Interactive In-Chat Chart Preview Card
    let chartPreviewHtml = "";
    if (data.charts && data.charts.length > 0) {
      if (!window.sessionCharts) window.sessionCharts = [];
      const chartIdx = window.sessionCharts.length;
      window.sessionCharts.push(data.charts[0]);
      const chartTitle = data.charts[0].title || "Biểu đồ phân tích tài chính";
      chartPreviewHtml = `
        <div class="chart-preview-card" data-chart-idx="${chartIdx}">
          <div class="chart-preview-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
            <span>${escapeHtml(chartTitle)}</span>
          </div>
          <button type="button" class="btn-view-canvas">
            <span>Xem biểu đồ</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          </button>
        </div>
      `;
    }

    // Collapsible Sources & References Card
    let sourcesAccordionHtml = "";
    if (data.citations && data.citations.length > 0) {
      const citationItems = data.citations.map((c, idx) => {
        const citNum = c.citation_id || (idx + 1);
        const ticker = c.ticker || "";
        const form = c.form || "10-K";
        const excerpt = c.text || c.snippet || "Trích dẫn hồ sơ SEC EDGAR hợp lệ.";
        const url = c.source_url || `https://www.sec.gov/edgar/browse/?CIK=${c.cik || ''}`;
        return `
          <div class="source-item-card">
            <div class="source-item-header">
              <span class="citation-badge">[${citNum}]</span>
              <span class="source-company">${escapeHtml(c.company_name || ticker || "SEC EDGAR")}</span>
              <span class="source-badge">${escapeHtml(form)}</span>
            </div>
            <div class="source-excerpt">${escapeHtml(excerpt.length > 220 ? excerpt.substring(0, 220) + "..." : excerpt)}</div>
            <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="source-link">
              <span>Xem hồ sơ gốc trên SEC EDGAR</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
            </a>
          </div>
        `;
      }).join("");

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
              <span class="sources-title">Nguồn & Trích dẫn hồ sơ SEC EDGAR (${data.citations.length} nguồn)</span>
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
              <span class="activity-title">Tiến trình Agent</span>
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

  function translateStepName(stepName) {
    if (!stepName) return "";
    const map = {
      "Step 1: Intent Routing": "Bước 1: Phân loại ý định",
      "Step 2: DuckDB SQL Marts": "Bước 2: Truy vấn DuckDB SQL",
      "Step 3: FastEmbed Hybrid RAG": "Bước 3: Tìm kiếm ngữ nghĩa Hybrid RAG",
      "Step 4: Verification": "Bước 4: Đối soát & Kiểm chứng",
      "Understanding question": "Hiểu câu hỏi & Lập kế hoạch",
      "Resolving metrics": "Xác định chỉ tiêu tài chính",
      "Querying DuckDB": "Truy vấn DuckDB Lakehouse",
      "Searching SEC filings": "Tìm kiếm tài liệu SEC EDGAR",
      "Validating evidence": "Xác thực bằng chứng",
      "Preparing visualization": "Khởi tạo biểu đồ trực quan",
    };
    return map[stepName] || stepName;
  }

  // Update Dynamic Financial Canvas with Chart from Backend
  function updateCanvasWithAgentChart(chartSpec) {
    if (!financialChart || !chartSpec) return;

    // 1. Resolve Labels & Datasets (Support both Backend ChartData contract & legacy mock)
    const labels = chartSpec.x_labels || chartSpec.categories;
    const seriesList = chartSpec.datasets || chartSpec.series;

    if (!labels || !seriesList || !Array.isArray(seriesList) || seriesList.length === 0) return;

    financialChart.data.labels = labels;
    const palette = ["#00A3FF", "#00E599", "#F5A623", "#8B5CF6", "#EF4444", "#EC4899", "#14B8A6"];
    const isChartPercent = chartSpec.unit === "%" || (chartSpec.title && chartSpec.title.includes("(%)"));

    financialChart.data.datasets = seriesList.map((s, idx) => {
      const name = s.label || s.name || `Series ${idx + 1}`;
      const rawValues = s.values !== undefined ? s.values : s.data;
      const isMargin = isChartPercent || name.toLowerCase().includes("margin") || name.toLowerCase().includes("biên") || name.toLowerCase().includes("ratio");
      const color = s.color || palette[idx % palette.length];

      return {
        label: name,
        data: rawValues,
        backgroundColor: color,
        borderColor: color,
        type: isMargin ? (chartMode === "bar" && isChartPercent ? "bar" : "line") : (chartMode === "line" ? "line" : "bar"),
        yAxisID: isChartPercent ? "y1" : (isMargin ? "y1" : "y"),
        borderRadius: 4,
        borderWidth: isMargin || chartMode === "line" ? 2.5 : 0,
        pointRadius: isMargin || chartMode === "line" ? 4 : 0,
        tension: 0.3,
      };
    });

    // Dynamic Axis adjustments based on metric units
    if (financialChart.options.scales.y) {
      financialChart.options.scales.y.display = !isChartPercent;
      if (financialChart.options.scales.y.title) {
        financialChart.options.scales.y.title.text = chartSpec.unit || "USD";
      }
    }
    if (financialChart.options.scales.y1) {
      financialChart.options.scales.y1.display = isChartPercent || financialChart.data.datasets.some(d => d.yAxisID === "y1");
      if (financialChart.options.scales.y1.title) {
        financialChart.options.scales.y1.title.text = isChartPercent ? (chartSpec.unit || "%") : "Biên hoạt động (%)";
      }
    }

    // Dynamic Tooltip formatting
    if (financialChart.options.plugins && financialChart.options.plugins.tooltip) {
      financialChart.options.plugins.tooltip.callbacks.label = function(context) {
        const val = context.raw;
        if (val === null || val === undefined) return `${context.dataset.label}: N/A`;
        if (context.dataset.yAxisID === "y1" || isChartPercent) {
          return `${context.dataset.label}: ${val}%`;
        }
        const unitSuffix = (chartSpec.unit && chartSpec.unit.includes("Billion")) ? "B" : "M";
        return `${context.dataset.label}: $${val.toLocaleString()}${unitSuffix}`;
      };
    }

    financialChart.update();
    renderCustomChartLegend();

    // 2. Dynamically Populate Financial Table below Chart
    updateFinancialGridTable(labels, seriesList, chartSpec.unit);
  }

  // Populate Structured Financial Data Grid Table below the Chart
  function updateFinancialGridTable(labels, seriesList, unit) {
    const tableEl = document.getElementById("financial-data-table");
    if (!tableEl) return;
    const theadTr = tableEl.querySelector("thead tr");
    const tbody = document.getElementById("financial-table-body");
    if (!theadTr || !tbody) return;

    if (!labels || !seriesList || !Array.isArray(seriesList) || seriesList.length === 0) return;

    // Build Table Header: "Chỉ tiêu / Doanh nghiệp" + each year label
    let theadHtml = `<th class="col-metric">Chỉ tiêu / Doanh nghiệp</th>`;
    labels.forEach(l => {
      theadHtml += `<th class="col-num">${escapeHtml(String(l))}</th>`;
    });
    theadTr.innerHTML = theadHtml;

    // Build Table Rows: 1 row per dataset/series
    const isPercent = (unit && unit.includes("%"));
    tbody.innerHTML = seriesList.map(s => {
      const name = s.label || s.name || "Chỉ tiêu";
      const rawValues = s.values !== undefined ? s.values : s.data;
      const sIsPercent = isPercent || name.toLowerCase().includes("margin") || name.toLowerCase().includes("biên") || name.toLowerCase().includes("ratio");

      let cells = `<td class="metric-name">${escapeHtml(name)}</td>`;
      labels.forEach((_, idx) => {
        const val = rawValues && rawValues[idx] !== undefined ? rawValues[idx] : null;
        let formatted = "—";
        if (val !== null && val !== undefined) {
          if (sIsPercent) {
            formatted = `${val}%`;
          } else if (typeof val === "number") {
            formatted = val >= 1000 ? `$${val.toLocaleString()}` : (unit && unit.includes("Billion") ? `$${val.toLocaleString()}B` : `$${val.toLocaleString()}`);
          } else {
            formatted = val;
          }
        }
        cells += `<td class="num-val">${escapeHtml(String(formatted))}</td>`;
      });
      return `<tr>${cells}</tr>`;
    }).join("");
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

  /* ========================================================================= */
  /* Recents Sessions from MinIO S3 Lakehouse                                  */
  /* ========================================================================= */

  async function loadRecentsSessions() {
    try {
      const response = await fetch("/api/history");
      if (!response.ok) return;
      const data = await response.json();
      const sessions = data.sessions || [];

      if (sessions.length === 0) return;

      sessionList.innerHTML = "";
      sessions.slice(0, 8).forEach(s => {
        const item = document.createElement("button");
        item.className = `session-item ${s.session_id === currentSessionId ? 'active' : ''}`;
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
            await fetch(`/api/history/${s.session_id}`, { method: "DELETE" });
          } catch (err) {}
          item.remove();
          if (currentSessionId === s.session_id) {
            startNewSession();
          }
        });
        item.appendChild(trashBtn);

        item.addEventListener("click", () => {
          loadSession(s.session_id);
        });
        sessionList.appendChild(item);
      });
    } catch (e) {
      console.warn("Could not load sessions from MinIO Lakehouse:", e);
    }
  }

  async function loadSession(sessionId) {
    try {
      const response = await fetch(`/api/history/${sessionId}`);
      if (!response.ok) return;
      const sessionData = await response.json();
      currentSessionId = sessionId;
      localStorage.setItem("current_session_id", currentSessionId);

      if (sessionData.messages && sessionData.messages.length > 0) {
        messagesContainer.innerHTML = "";
        currentSessionTitle.textContent = sessionData.title || "Đoạn chat";

        let hasAnyChart = false;
        sessionData.messages.forEach(msg => {
          if (msg.role === "user") {
            appendUserTurn(msg.content);
          } else {
            renderAssistantTurn({
              answer: msg.content,
              citations: msg.citations || [],
              charts: msg.charts || [],
              activity_timeline: msg.activity_timeline || []
            });
            if (msg.charts && msg.charts.length > 0) {
              hasAnyChart = true;
              updateCanvasWithAgentChart(msg.charts[0]);
            }
          }
        });

        if (hasAnyChart) {
          expandCanvas();
          showCanvasEmptyState(false);
        } else {
          collapseCanvas();
          showCanvasEmptyState(true);
        }
      }
      scrollToBottom();
      loadRecentsSessions();
    } catch (e) {
      console.warn("Failed to load session details:", e);
    }
  }

  /* ========================================================================= */
  /* Voice STT Integration (Web Speech API)                                    */
  /* ========================================================================= */

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "vi-VN"; // Vietnamese speech recognition

    recognition.onstart = () => {
      isRecording = true;
      voiceBtn.classList.add("recording");
    };

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      queryInput.value = transcript;
    };

    recognition.onend = () => {
      isRecording = false;
      voiceBtn.classList.remove("recording");
      if (queryInput.value.trim()) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    };

    voiceBtn.addEventListener("click", () => {
      if (isRecording) {
        recognition.stop();
      } else {
        recognition.start();
      }
    });
  } else {
    voiceBtn.style.display = "none";
  }

  /* ========================================================================= */
  /* Helper Utilities                                                          */
  /* ========================================================================= */

  function scrollToBottom() {
    setTimeout(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 50);
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
});
