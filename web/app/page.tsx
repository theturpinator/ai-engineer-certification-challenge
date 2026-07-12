"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { apiFetch, AuthButton } from "./auth";
import { DeltaChips, type Deltas } from "./chips";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = { title: string; url: string };
type Ad = {
  product: string;
  advertiser: string;
  description: string;
  image: string | null;
  link: string | null;
  sponsored: boolean;
  deltas: Deltas | null;
};
type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  ads?: Ad[];
  status?: string; // transient "Searching the archive…" while a tool runs
  error?: boolean; // stream died; show the retry affordance
  retryText?: string;
};
type ChatMeta = { chat_id: string; title: string; updated_at: string };

// The pre-multi-chat thread is "default" server-side; normalize the client's
// transient "" so per-chat buffers and the request body always agree.
const chatKey = (id: string) => id || "default";
const EMPTY: Message[] = [];

const TOOL_STATUS: Record<string, string> = {
  search_archive: "Searching the archive…",
  web_search: "Searching the web…",
  check_recalls: "Checking recalls…",
  recommend_products: "Finding products…",
  update_garage: "Updating your garage…",
  update_instructions: "Noting your preference…",
};

// The server pings every ~10s whenever nothing else is streaming, so this
// long a silence can only mean a dead connection — not a slow tool call.
const WATCHDOG_MS = 30_000;

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  ms: number
): Promise<ReadableStreamReadResult<Uint8Array>> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reader.cancel().catch(() => {});
          reject(new Error(`stream stalled: no data for ${ms / 1000}s`));
        }, ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function AdCard({ ad }: { ad: Ad }) {
  return (
    <a
      className="adcard"
      href={ad.link ?? undefined}
      target="_blank"
      rel="noopener noreferrer sponsored"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {ad.image && <img src={ad.image} alt={`${ad.advertiser} ad`} />}
      <div className="adbody">
        <span className="sponsoredtag">Sponsored · {ad.advertiser}</span>
        <strong>{ad.product}</strong>
        <p>{ad.description}</p>
        <DeltaChips deltas={ad.deltas} />
      </div>
    </a>
  );
}

const NewTabLink = (props: React.ComponentProps<"a">) => (
  <a {...props} target="_blank" rel="noopener noreferrer" />
);

function ago(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// --- Standardized car intake (issue #15) ---

const YEARS = Array.from({ length: 2027 - 1964 + 1 }, (_, i) => String(2027 - i));
const TRIMS = ["Base", "EcoBoost", "GT", "Mach 1", "Bullitt", "Boss 302", "Boss 429",
  "LX", "SVO", "Cobra", "GT350", "GT500", "Dark Horse", "Other"];
const COLORS = ["Black", "White", "Silver", "Gray", "Red", "Race Red", "Blue",
  "Grabber Blue", "Green", "Yellow", "Orange", "Burgundy", "Brown", "Purple", "Other"];

type NewCar = { year: number; trim: string; color: string; nickname?: string };

function AddCarModal({
  uid,
  onClose,
  onAdded,
}: {
  uid: string;
  onClose: () => void;
  onAdded: (car: NewCar) => void;
}) {
  const [year, setYear] = useState("");
  const [trim, setTrim] = useState("");
  const [trimOther, setTrimOther] = useState("");
  const [color, setColor] = useState("");
  const [colorOther, setColorOther] = useState("");
  const [nickname, setNickname] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  const finalTrim = trim === "Other" ? trimOther.trim() : trim;
  const finalColor = color === "Other" ? colorOther.trim() : color;
  const ready = year && finalTrim && finalColor;

  async function save() {
    setSaving(true);
    setErr("");
    try {
      const r = await apiFetch(`${API_URL}/garage/${uid}/cars`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: parseInt(year, 10),
          trim: finalTrim,
          color: finalColor,
          nickname: nickname.trim() || undefined,
        }),
      });
      if (r.status === 409) throw new Error("dup");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      onAdded(await r.json());
    } catch (e) {
      setErr(
        e instanceof Error && e.message === "dup"
          ? "That year and trim is already in your garage."
          : "Couldn’t save — please try again."
      );
      setSaving(false);
    }
  }

  return (
    <div className="modal" onClick={onClose}>
      <div
        className="sheet"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Add your car"
      >
        <h2>Add your car</h2>
        <p className="empty">Ford Mustang — pick the year, trim, and color.</p>
        <div className="editgrid">
          <label>
            <span className="fieldlabel">Year *</span>
            <select value={year} onChange={(e) => setYear(e.target.value)}>
              <option value="">Select…</option>
              {YEARS.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </label>
          <label>
            <span className="fieldlabel">Trim *</span>
            <select value={trim} onChange={(e) => setTrim(e.target.value)}>
              <option value="">Select…</option>
              {TRIMS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          {trim === "Other" && (
            <label>
              <span className="fieldlabel">Trim name *</span>
              <input
                value={trimOther}
                onChange={(e) => setTrimOther(e.target.value)}
                placeholder="e.g. Grande"
              />
            </label>
          )}
          <label>
            <span className="fieldlabel">Color *</span>
            <select value={color} onChange={(e) => setColor(e.target.value)}>
              <option value="">Select…</option>
              {COLORS.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          {color === "Other" && (
            <label>
              <span className="fieldlabel">Color name *</span>
              <input
                value={colorOther}
                onChange={(e) => setColorOther(e.target.value)}
                placeholder="e.g. Eleanor Gray"
              />
            </label>
          )}
          <label>
            <span className="fieldlabel">Nickname</span>
            <input
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder="Optional"
            />
          </label>
        </div>
        {err && <p className="formerr">{err}</p>}
        <div className="editactions">
          <button type="button" className="secondary" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button type="button" onClick={save} disabled={saving || !ready}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Chat() {
  // Per-chat buffers: an in-flight stream keeps writing into its own chat's
  // thread, so switching away and back never corrupts what's on screen.
  const [threads, setThreads] = useState<Record<string, Message[]>>({});
  const [streaming, setStreaming] = useState<Record<string, boolean>>({});
  const [input, setInput] = useState("");
  const [chats, setChats] = useState<ChatMeta[]>([]);
  const [drawer, setDrawer] = useState(false);
  const [chatId, setChatId] = useState("");
  const [picker, setPicker] = useState(false);
  const [carPrompt, setCarPrompt] = useState(false);
  const userId = useRef<string>("");
  const sessionId = useRef<string>("");
  const streamingRef = useRef<Record<string, boolean>>({});
  const bottom = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const messages = threads[chatKey(chatId)] ?? EMPTY;
  const busy = !!streaming[chatKey(chatId)];

  function updateThread(k: string, fn: (msgs: Message[]) => Message[]) {
    setThreads((t) => ({ ...t, [k]: fn(t[k] ?? []) }));
  }
  function patchLast(k: string, fn: (last: Message) => Message) {
    updateThread(k, (msgs) =>
      msgs.length ? [...msgs.slice(0, -1), fn(msgs[msgs.length - 1])] : msgs
    );
  }
  function setStreamFlag(k: string, v: boolean) {
    streamingRef.current[k] = v;
    setStreaming((s) => ({ ...s, [k]: v }));
  }

  // Auto-grow the composer with its content; CSS max-height caps it at ~5
  // lines, past which it scrolls. (field-sizing: content would be free, but
  // Safari/Firefox don't support it yet.)
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + el.offsetHeight - el.clientHeight}px`;
  }, [input]);

  useEffect(() => {
    let id = localStorage.getItem("md_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("md_user_id", id);
    }
    userId.current = id;
    // One session per visit: new per tab, survives refresh (sessionStorage).
    let sid = sessionStorage.getItem("md_session_id");
    if (!sid) {
      sid = crypto.randomUUID();
      sessionStorage.setItem("md_session_id", sid);
    }
    sessionId.current = sid;
    // "default" is the pre-multi-chat thread, so existing history shows up.
    setChatId(localStorage.getItem("md_chat_id") || "default");
    // Empty garage -> suggest the picker (once; dismissal sticks).
    if (localStorage.getItem("md_addcar_prompt") !== "dismissed") {
      apiFetch(`${API_URL}/garage/${id}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((g) => {
          if (g && !g.profile?.cars?.length) setCarPrompt(true);
        })
        .catch(() => {});
    }
  }, []);

  // The server transcript is the source of truth when (re)opening a chat —
  // unless a stream is live for it, in which case the buffer is fresher.
  useEffect(() => {
    if (!chatId || !userId.current) return;
    const k = chatKey(chatId);
    if (streamingRef.current[k]) return;
    apiFetch(`${API_URL}/chats/${userId.current}/${chatId}/messages`)
      .then((r) => (r.ok ? r.json() : []))
      .then((msgs) => setThreads((t) => ({ ...t, [k]: msgs })))
      .catch(() => {});
  }, [chatId]);

  useEffect(() => {
    bottom.current?.scrollIntoView();
  }, [messages]);

  function openDrawer() {
    setDrawer(true);
    apiFetch(`${API_URL}/chats/${userId.current}`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setChats)
      .catch(() => {});
  }

  function switchTo(id: string) {
    setDrawer(false);
    if (id === chatId) return;
    localStorage.setItem("md_chat_id", id);
    // A resumed or new chat is a fresh session for episodic memory.
    sessionId.current = crypto.randomUUID();
    sessionStorage.setItem("md_session_id", sessionId.current);
    // No buffer reset: an in-flight stream for the old chat keeps draining
    // into its own thread and is still there when the user switches back.
    setChatId(id);
  }

  function newChat() {
    switchTo(crypto.randomUUID().slice(0, 8));
  }

  function dismissCarPrompt() {
    setCarPrompt(false);
    localStorage.setItem("md_addcar_prompt", "dismissed");
  }

  function carAdded(car: NewCar) {
    setPicker(false);
    setCarPrompt(false);
    // client-side confirmation only — not an LLM turn, not in the transcript
    updateThread(chatKey(chatId), (m) => [
      ...m,
      {
        role: "assistant",
        content: `🏁 Added your ${car.year} Mustang ${car.trim} to the garage — check My Garage!`,
      },
    ]);
  }

  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    const k = chatKey(chatId); // captured: every write below targets THIS chat
    if (!text || streaming[k]) return;
    if (textArg === undefined) setInput("");
    setStreamFlag(k, true);
    updateThread(k, (m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);

    try {
      // Never abort after headers arrive: tearing the socket would cancel
      // generation server-side. The watchdog only fires on true silence,
      // which the server's ~10s pings rule out on a live connection.
      const ctrl = new AbortController();
      const headerTimer = setTimeout(() => ctrl.abort(), WATCHDOG_MS);
      let res: Response;
      try {
        res = await apiFetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            user_id: userId.current,
            session_id: sessionId.current,
            chat_id: k,
          }),
          signal: ctrl.signal,
        });
      } finally {
        clearTimeout(headerTimer);
      }
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await readWithTimeout(reader, WATCHDOG_MS);
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop()!;
        for (const event of events) {
          const data = event.replace(/^data: /, "");
          if (data === "[DONE]") continue;
          const parsed = JSON.parse(data);
          if (parsed.type === "token") {
            patchLast(k, (last) => ({
              ...last,
              status: undefined,
              content: last.content + parsed.text,
            }));
          } else if (parsed.type === "tool_start") {
            patchLast(k, (last) => ({
              ...last,
              status: TOOL_STATUS[parsed.name as string] ?? "Working…",
            }));
          } else if (parsed.type === "citations" && parsed.citations.length > 0) {
            patchLast(k, (last) => ({ ...last, citations: parsed.citations }));
          } else if (parsed.type === "ad") {
            patchLast(k, (last) => ({
              ...last,
              ads: [...(last.ads ?? []), parsed as Ad],
            }));
          } else if (parsed.type === "error") {
            throw new Error("server reported a stream error");
          }
          // "tool" and "ping" events just feed the watchdog
        }
      }
    } catch (err) {
      console.error("chat stream failed:", err); // details for debugging only
      patchLast(k, (last) => ({
        ...last,
        status: undefined,
        error: true,
        retryText: text,
      }));
    } finally {
      setStreamFlag(k, false);
    }
  }

  function retry(text: string) {
    const k = chatKey(chatId);
    // Drop the failed user+assistant pair, then resend the same message.
    updateThread(k, (msgs) =>
      msgs.length && msgs[msgs.length - 1].error ? msgs.slice(0, -2) : msgs
    );
    send(text);
  }

  return (
    <main>
      {picker && (
        <AddCarModal uid={userId.current} onClose={() => setPicker(false)} onAdded={carAdded} />
      )}
      {drawer && (
        <div className="backdrop" onClick={() => setDrawer(false)}>
          <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Chats">
            <button type="button" className="newchat" onClick={newChat}>
              + New chat
            </button>
            <ul className="chatlist">
              {chats.map((c) => (
                <li key={c.chat_id}>
                  <button
                    type="button"
                    className={c.chat_id === chatId ? "active" : ""}
                    onClick={() => switchTo(c.chat_id)}
                  >
                    <span className="title">{c.title}</span>
                    <span className="time">{ago(c.updated_at)}</span>
                  </button>
                </li>
              ))}
              {chats.length === 0 && <li className="empty">No chats yet.</li>}
            </ul>
          </aside>
        </div>
      )}
      <header>
        <button type="button" className="menubtn" aria-label="Chats" onClick={openDrawer}>
          ☰
        </button>
        Ask MustangDriver
        <nav>
          <AuthButton />
          <Link href="/garage">My Garage</Link>
        </nav>
      </header>
      <div className="messages">
        {carPrompt && (
          <div className="garageprompt">
            <span>Got a Mustang? Add it to your garage in a few taps.</span>
            <button type="button" className="addcarchip" onClick={() => setPicker(true)}>
              + Add your car
            </button>
            <button
              type="button"
              className="dismiss"
              aria-label="Dismiss"
              onClick={dismissCarPrompt}
            >
              ×
            </button>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`}>
            {msg.role === "assistant" ? (
              msg.content ? (
                <ReactMarkdown components={{ a: NewTabLink }}>
                  {msg.content}
                </ReactMarkdown>
              ) : msg.error ? null : (
                <p className="pending">{msg.status ?? "…"}</p>
              )
            ) : (
              msg.content
            )}
            {msg.error && (
              <div className="stallnote">
                <span>Connection lost — the reply may be incomplete.</span>
                {msg.retryText && (
                  <button type="button" onClick={() => retry(msg.retryText!)}>
                    Retry
                  </button>
                )}
              </div>
            )}
            {msg.ads?.map((ad, j) => (
              <AdCard key={j} ad={ad} />
            ))}
            {msg.citations && (
              <div className="sources">
                <strong>Sources</strong>
                <ul>
                  {msg.citations.map((c) => (
                    <li key={c.url}>
                      <a href={c.url} target="_blank" rel="noopener noreferrer">
                        {c.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      <div className="chiprow">
        <button type="button" className="addcarchip" onClick={() => setPicker(true)}>
          + Add your car
        </button>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          ref={box}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Desktop: Enter sends, Shift+Enter newline. Touch keyboards
            // (coarse pointer): return is a newline; the Send button sends.
            if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
            if (window.matchMedia("(pointer: coarse)").matches) return;
            e.preventDefault();
            send();
          }}
          placeholder="Ask about Mustangs…"
          aria-label="Message"
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
