/* Shared browser helpers for the gateway UI. */
const PROMPT_OK = /^[\t\n\x20-\x7E]+$/;

async function postJSON(url, body, headers = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { /* ignore */ }
  return { ok: res.ok, status: res.status, data };
}

async function getJSON(url, headers = {}) {
  const res = await fetch(url, { headers });
  let data = null;
  try { data = await res.json(); } catch { /* ignore */ }
  return { ok: res.ok, status: res.status, data };
}

// Read a Server-Sent Events stream (fetch-based, so we can send a POST body and
// auth headers). Calls onEvent(eventName, parsedData) per event.
async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      let parsed = null;
      try { parsed = data ? JSON.parse(data) : null; } catch { /* ignore */ }
      onEvent(event, parsed);
    }
  }
}

// Animate only the real transition from the model's reversed output to its
// final readable output. Timing is driven by animation events, not fake typing.
function revealReadable(el, text, fallback) {
  const kind = el.classList.contains("story") ? "story" : "answer";
  const value = text || fallback;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.className = `${kind} done`;
    el.textContent = value;
    return;
  }
  el.className = `${kind} reversing-out`;
  el.addEventListener("animationend", () => {
    el.textContent = value;
    el.className = `${kind} done reversing-in`;
    el.addEventListener("animationend", () => {
      el.className = `${kind} done`;
    }, { once: true });
  }, { once: true });
}

function errMessage(data, fallback) {
  if (!data) return fallback;
  if (typeof data.error === "string") return data.error;
  if (data.error && data.error.message) return data.error.message;
  return fallback;
}
