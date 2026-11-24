/**
 * Corporate Filing Intelligence - Real-Time Stock Market Terminal & Watchlist Controller
 * Handles paginated live quotes, market filtering, sorting, add/delete/reset actions,
 * and quick handoff into the AI financial research terminal.
 */

import {
  addWatchlistTickerApi,
  deleteWatchlistTickerApi,
  escapeHtml,
  fetchWatchlistQuotesApi,
  resetWatchlistApi,
} from "./api.js";

function formatStockPrice(price, currency) {
  if (price === null || price === undefined) return "N/A";
  if (currency === "VND") {
    return new Intl.NumberFormat("vi-VN").format(Math.round(price)) + " ₫";
  }
  return "$" + Number(price).toFixed(2);
}

function formatStockChange(change, currency) {
  if (change === null || change === undefined) return "--";
  const sign = change > 0 ? "+" : "";
  if (currency === "VND") {
    return sign + new Intl.NumberFormat("vi-VN").format(Math.round(change)) + " ₫";
  }
  return sign + "$" + Number(change).toFixed(2);
}

function formatStockPct(pct) {
  if (pct === null || pct === undefined) return "--";
  const sign = pct > 0 ? "+" : "";
  return sign + Number(pct).toFixed(2) + "%";
}

function formatStockVolume(vol) {
  if (!vol) return "--";
  return new Intl.NumberFormat("en-US").format(vol);
}

export function initWatchlistController({ onAskAi }) {
  const navStockTerminalBtn = document.getElementById("nav-stock-terminal-btn");
  const stockTerminalPage = document.getElementById("stock-terminal-page");
  const stockBackToChatBtn = document.getElementById("stock-back-to-chat-btn");
  const stockTableBody = document.getElementById("stock-table-body");
  const stockRefreshBtn = document.getElementById("stock-refresh-btn");
  const stockResetBtn = document.getElementById("stock-reset-btn");
  const stockAddForm = document.getElementById("stock-add-form");
  const stockTickerInput = document.getElementById("stock-ticker-input");
  const stockLoadingSpinner = document.getElementById("stock-loading-spinner");
  const statTotalSymbols = document.getElementById("stat-total-symbols");
  const statGainersCount = document.getElementById("stat-gainers-count");
  const statLosersCount = document.getElementById("stat-losers-count");
  const stockLastUpdatedClock = document.getElementById("stock-last-updated-clock");
  const marketFilterTabs = document.querySelectorAll(".market-tab-btn");
  const stockPaginationBar = document.getElementById("stock-pagination-bar");
  const stockPrevPageBtn = document.getElementById("stock-prev-page");
  const stockNextPageBtn = document.getElementById("stock-next-page");
  const stockPageInfo = document.getElementById("stock-page-info");

  const WATCHLIST_PAGE_SIZE = 20;
  let currentWatchlistQuotes = [];
  let currentMarketFilter = "ALL";
  let currentWatchlistPage = 1;
  let totalWatchlistPages = 1;
  let totalWatchlistSymbols = 0;
  let currentSortCol = null;
  let currentSortAsc = true;

  function openStockTerminal() {
    if (!stockTerminalPage) return;
    stockTerminalPage.style.display = "flex";
    document.body.style.overflow = "hidden";
    fetchWatchlistQuotes(false);
  }

  function closeStockTerminal() {
    if (!stockTerminalPage) return;
    stockTerminalPage.style.display = "none";
    document.body.style.overflow = "";
  }

  function updateWatchlistPaginationUI() {
    if (!stockPaginationBar) return;
    stockPaginationBar.style.display =
      totalWatchlistSymbols > WATCHLIST_PAGE_SIZE || totalWatchlistPages > 1 ? "flex" : "none";
    if (stockPageInfo) {
      stockPageInfo.textContent = `Trang ${currentWatchlistPage} / ${totalWatchlistPages} (${totalWatchlistSymbols} mã)`;
    }
    if (stockPrevPageBtn) stockPrevPageBtn.disabled = currentWatchlistPage <= 1;
    if (stockNextPageBtn) stockNextPageBtn.disabled = currentWatchlistPage >= totalWatchlistPages;
  }

  function updateStockSummaryStats(items, updatedAt, totalCount = items.length) {
    if (statTotalSymbols) statTotalSymbols.textContent = totalCount;

    let gainers = 0;
    let losers = 0;
    items.forEach((q) => {
      if (q.change > 0) gainers++;
      else if (q.change < 0) losers++;
    });

    if (statGainersCount) statGainersCount.textContent = `${gainers} ↗`;
    if (statLosersCount) statLosersCount.textContent = `${losers} ↘`;

    if (stockLastUpdatedClock && updatedAt) {
      const date = new Date(updatedAt);
      stockLastUpdatedClock.textContent = "Cập nhật lúc: " + date.toLocaleTimeString("vi-VN");
    }
  }

  function renderStockTable() {
    if (!stockTableBody) return;
    stockTableBody.innerHTML = "";

    let filtered = currentWatchlistQuotes.filter((item) => {
      if (currentMarketFilter === "ALL") return true;
      return item.market === currentMarketFilter;
    });

    if (currentSortCol) {
      filtered = [...filtered].sort((a, b) => {
        let valA = a[currentSortCol];
        let valB = b[currentSortCol];
        if (valA === null || valA === undefined) valA = currentSortAsc ? Infinity : -Infinity;
        if (valB === null || valB === undefined) valB = currentSortAsc ? Infinity : -Infinity;

        if (typeof valA === "string") {
          return currentSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return currentSortAsc ? valA - valB : valB - valA;
      });
    }

    if (filtered.length === 0) {
      stockTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; padding: 40px; color: var(--text-muted);">
            Không có mã cổ phiếu nào trong danh mục này. Hãy thêm mã mới ở ô bên trên.
          </td>
        </tr>
      `;
      return;
    }

    filtered.forEach((item) => {
      const tr = document.createElement("tr");
      const isUp = item.change > 0;
      const isDown = item.change < 0;
      const badgeClass = isUp ? "up" : isDown ? "down" : "flat";
      const marketClass = (item.market || "US").toLowerCase();

      let rangeHtml = `<span style="color: var(--text-muted);">--</span>`;
      if (item.day_low !== null && item.day_high !== null) {
        const lowStr = formatStockPrice(item.day_low, item.currency);
        const highStr = formatStockPrice(item.day_high, item.currency);
        rangeHtml = `<span class="day-range-val">${lowStr} — ${highStr}</span>`;
      }

      tr.innerHTML = `
        <td><span class="stock-sym-cell">${escapeHtml(item.ticker)}</span></td>
        <td><div class="stock-name-cell" title="${escapeHtml(item.company_name || item.ticker)}">${escapeHtml(item.company_name || "--")}</div></td>
        <td><span class="market-badge ${marketClass}">${item.market}</span></td>
        <td class="text-right"><span class="price-cell">${formatStockPrice(item.price, item.currency)}</span></td>
        <td class="text-right" style="color: ${isUp ? "#00E599" : isDown ? "#EF4444" : "var(--text-muted)"}; font-weight: 550; font-feature-settings: 'tnum';">
          ${formatStockChange(item.change, item.currency)}
        </td>
        <td class="text-right">
          <span class="pct-badge ${badgeClass}">${formatStockPct(item.percent_change)}</span>
        </td>
        <td class="text-center">${rangeHtml}</td>
        <td class="text-right num-val-cell">${formatStockVolume(item.volume)}</td>
        <td class="text-center">
          <div class="table-actions-cell">
            <button class="btn-ask-ai" data-ticker="${escapeHtml(item.ticker)}" title="Hỏi trợ lý AI phân tích doanh nghiệp này">
              <span>Hỏi AI</span>
            </button>
            <button class="btn-delete-symbol" data-ticker="${escapeHtml(item.ticker)}" title="Xóa khỏi danh mục theo dõi">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>
        </td>
      `;

      const askBtn = tr.querySelector(".btn-ask-ai");
      if (askBtn) {
        askBtn.addEventListener("click", () => {
          closeStockTerminal();
          if (onAskAi) {
            onAskAi(`Phân tích tình hình tài chính, kết quả kinh doanh và các điểm nổi bật gần nhất của ${item.ticker}`);
          }
        });
      }

      const delBtn = tr.querySelector(".btn-delete-symbol");
      if (delBtn) {
        delBtn.addEventListener("click", async () => {
          if (confirm(`Bạn có chắc chắn muốn xóa mã ${item.ticker} khỏi danh mục theo dõi?`)) {
            try {
              const res = await deleteWatchlistTickerApi(item.ticker);
              if (res.ok) {
                fetchWatchlistQuotes(false);
              }
            } catch (err) {
              console.error("Failed to delete ticker:", err);
            }
          }
        });
      }

      stockTableBody.appendChild(tr);
    });
  }

  async function fetchWatchlistQuotes(forceRefresh = false, page = currentWatchlistPage) {
    if (stockLoadingSpinner) stockLoadingSpinner.style.display = "flex";
    if (stockRefreshBtn) stockRefreshBtn.classList.add("loading");

    try {
      const data = await fetchWatchlistQuotesApi({
        page,
        pageSize: WATCHLIST_PAGE_SIZE,
        market: currentMarketFilter,
        forceRefresh,
      });
      currentWatchlistQuotes = data.items || [];
      currentWatchlistPage = data.page || 1;
      totalWatchlistPages = data.total_pages || 1;
      totalWatchlistSymbols = typeof data.total === "number" ? data.total : currentWatchlistQuotes.length;

      updateStockSummaryStats(currentWatchlistQuotes, data.updated_at, totalWatchlistSymbols);
      updateWatchlistPaginationUI();
      renderStockTable();
    } catch (err) {
      console.error("Watchlist fetch error:", err);
    } finally {
      if (stockLoadingSpinner) stockLoadingSpinner.style.display = "none";
      if (stockRefreshBtn) stockRefreshBtn.classList.remove("loading");
    }
  }

  if (navStockTerminalBtn) {
    navStockTerminalBtn.addEventListener("click", openStockTerminal);
  }

  if (stockBackToChatBtn) {
    stockBackToChatBtn.addEventListener("click", closeStockTerminal);
  }

  const sidebarWatchlistCard = document.querySelector(".sidebar-watchlist-card");
  if (sidebarWatchlistCard) {
    const moreBtn = sidebarWatchlistCard.querySelector(".btn-watchlist-more");
    if (moreBtn) {
      moreBtn.addEventListener("click", openStockTerminal);
    }
  }

  if (stockAddForm && stockTickerInput) {
    stockAddForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const symbol = stockTickerInput.value.trim().toUpperCase();
      if (!symbol) return;

      if (currentWatchlistQuotes.some((q) => q.ticker.toUpperCase() === symbol)) {
        alert(`Mã cổ phiếu ${symbol} đã có sẵn trong danh mục theo dõi của bạn!`);
        stockTickerInput.value = "";
        return;
      }

      const btnAdd = document.getElementById("btn-stock-add");
      if (btnAdd) {
        btnAdd.disabled = true;
        btnAdd.textContent = "Đang thêm...";
      }

      try {
        const resp = await addWatchlistTickerApi(symbol);
        if (!resp.ok) {
          const errData = await resp.json();
          alert(errData.detail || "Không thể thêm mã cổ phiếu này");
          return;
        }
        stockTickerInput.value = "";
        await fetchWatchlistQuotes(false);
      } catch (err) {
        console.error("Error adding ticker:", err);
        alert("Lỗi kết nối khi thêm mã cổ phiếu");
      } finally {
        if (btnAdd) {
          btnAdd.disabled = false;
          btnAdd.innerHTML = "<span>+ Thêm mã</span>";
        }
      }
    });
  }

  if (stockRefreshBtn) {
    stockRefreshBtn.addEventListener("click", () => fetchWatchlistQuotes(true));
  }

  if (stockResetBtn) {
    stockResetBtn.addEventListener("click", async () => {
      if (confirm("Khôi phục danh mục theo dõi về danh sách mặc định AAPL, MSFT, GOOGL, NVDA, FPT, VNM, HPG?")) {
        try {
          if (stockLoadingSpinner) stockLoadingSpinner.style.display = "flex";
          const resp = await resetWatchlistApi();
          if (resp.ok) await fetchWatchlistQuotes(true);
        } catch (err) {
          console.error("Error resetting watchlist:", err);
        } finally {
          if (stockLoadingSpinner) stockLoadingSpinner.style.display = "none";
        }
      }
    });
  }

  const allowedSortCols = ["ticker", "company_name", "price", "change", "percent_change", "volume"];
  const sortableHeaders = document.querySelectorAll(".sortable-th");
  sortableHeaders.forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-sort");
      if (!col || !allowedSortCols.includes(col)) return;

      if (currentSortCol === col) {
        currentSortAsc = !currentSortAsc;
      } else {
        currentSortCol = col;
        currentSortAsc = true;
      }

      sortableHeaders.forEach((h) => {
        const arrow = h.querySelector(".sort-arrow");
        if (h === th) {
          h.classList.add("active-sort");
          if (arrow) arrow.textContent = currentSortAsc ? " ↑" : " ↓";
        } else {
          h.classList.remove("active-sort");
          if (arrow) arrow.textContent = "";
        }
      });

      renderStockTable();
    });
  });

  if (stockPrevPageBtn) {
    stockPrevPageBtn.addEventListener("click", () => {
      if (currentWatchlistPage > 1) fetchWatchlistQuotes(false, currentWatchlistPage - 1);
    });
  }

  if (stockNextPageBtn) {
    stockNextPageBtn.addEventListener("click", () => {
      if (currentWatchlistPage < totalWatchlistPages) fetchWatchlistQuotes(false, currentWatchlistPage + 1);
    });
  }

  marketFilterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      marketFilterTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentMarketFilter = tab.getAttribute("data-filter") || "ALL";
      currentWatchlistPage = 1;
      fetchWatchlistQuotes(false, 1);
    });
  });

  // Initial background fetch for watchlist data
  fetchWatchlistQuotes(false);

  return { openStockTerminal, closeStockTerminal, fetchWatchlistQuotes };
}
