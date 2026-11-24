/**
 * Corporate Filing Intelligence - API Client & Shared Browser Utilities
 * Encapsulates SSE streaming queries, session history APIs, watchlist APIs,
 * and safe HTML/URL sanitization helpers.
 */

export function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function sanitizeUrl(url) {
  if (!url || typeof url !== "string") return "#";
  const trimmed = url.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return "#";
}

export function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Stream a research query from POST /api/query/stream using Server-Sent Events.
 */
export async function streamAgentQuery(queryText, sessionId, handlers = {}) {
  const { onStatus, onToken } = handlers;
  const response = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      query: queryText,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    let detail = `Lỗi máy chủ: ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) detail = errorData.detail;
    } catch (_e) {
      // Ignore non-JSON error payloads
    }
    throw new Error(detail);
  }

  if (!response.body || !response.body.getReader) {
    throw new Error("Trình duyệt không hỗ trợ luồng SSE.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      if (!frame.trim()) continue;
      let eventName = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }
      if (dataLines.length === 0) continue;

      let payload = null;
      try {
        payload = JSON.parse(dataLines.join("\n"));
      } catch (_e) {
        continue;
      }

      if (eventName === "status" && payload) {
        if (onStatus) onStatus(payload);
      } else if (eventName === "token" && payload && payload.delta) {
        if (onToken) onToken(payload.delta);
      } else if (eventName === "done" && payload) {
        finalData = payload;
      } else if (eventName === "error" && payload) {
        throw new Error(payload.detail || "Lỗi khi xử lý câu hỏi.");
      }
    }
  }

  if (!finalData) {
    throw new Error("Không nhận được phản hồi hoàn chỉnh từ máy chủ.");
  }

  return finalData;
}

export async function fetchHistorySessions() {
  const response = await fetch("/api/history");
  if (!response.ok) return null;
  return response.json();
}

export async function fetchHistorySession(sessionId) {
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`);
  if (!response.ok) return null;
  return response.json();
}

export async function deleteHistorySession(sessionId) {
  await fetch(`/api/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export async function fetchWatchlistQuotesApi({ page = 1, pageSize = 20, market = "ALL", forceRefresh = false } = {}) {
  const params = new URLSearchParams({
    page: String(Math.max(1, page)),
    page_size: String(pageSize),
    refresh: String(Boolean(forceRefresh)),
  });
  if (market && market !== "ALL") {
    params.set("market", market);
  }
  const resp = await fetch(`/api/watchlist?${params.toString()}`);
  if (!resp.ok) throw new Error("Failed to fetch watchlist quotes");
  return resp.json();
}

export async function addWatchlistTickerApi(symbol) {
  const resp = await fetch("/api/watchlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: symbol }),
  });
  return resp;
}

export async function deleteWatchlistTickerApi(ticker) {
  return fetch(`/api/watchlist/${encodeURIComponent(ticker)}`, { method: "DELETE" });
}

export async function resetWatchlistApi() {
  return fetch("/api/watchlist/reset", { method: "POST" });
}
