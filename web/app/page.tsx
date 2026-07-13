"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { apiFetch, AuthButton, needsOnboarding } from "./auth";
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

const EXAMPLES = [
  "What oil does a 2016 GT take?",
  "Best first mods for the 5.0 Coyote?",
  "Tell me about the 1969 Mach 1",
];

const TOOL_STATUS: Record<string, string> = {
  search_archive: "Searching the archive…",
  web_search: "Searching the web…",
  check_recalls: "Checking recalls…",
  recommend_products: "Finding products…",
  search_sponsor_sites: "Searching sponsor sites…",
  update_garage: "Updating your garage…",
  update_instructions: "Noting your preference…",
  complete_onboarding: "Saving your profile…",
};

// First-run onboarding (issue #46): this hidden message opens a brand-new
// user's chat; the agent interviews them until complete_onboarding fires.
const ONBOARD_KICKOFF = "[begin onboarding]";

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

export default function Chat() {
  // Per-chat buffers: an in-flight stream keeps writing into its own chat's
  // thread, so switching away and back never corrupts what's on screen.
  const [threads, setThreads] = useState<Record<string, Message[]>>({});
  const [streaming, setStreaming] = useState<Record<string, boolean>>({});
  const [input, setInput] = useState("");
  const [chats, setChats] = useState<ChatMeta[]>([]);
  const [drawer, setDrawer] = useState(false);
  const [chatId, setChatId] = useState("");
  // First-run onboarding (issue #46): the agent interviews the user in this
  // chat; until complete_onboarding flips the server flag, all navigation
  // away from the conversation is hidden.
  const [onboarding, setOnboarding] = useState(false);
  const onboardingRef = useRef(false);
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
    // A user we know nothing about gets interviewed before anything else:
    // lock the nav and have the agent start the conversation. A missing
    // server flag means the kickoff was never sent; false = in progress
    // (mid-interview reload), so just re-lock and let them keep answering.
    apiFetch(`${API_URL}/garage/${id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((g) => {
        if (!g || !needsOnboarding(g)) return;
        onboardingRef.current = true;
        setOnboarding(true);
        if (g.profile?.onboarded === undefined) send(ONBOARD_KICKOFF);
      })
      .catch(() => {});
  }, []);

  // The server transcript is the source of truth when (re)opening a chat —
  // unless a stream is live for it, in which case the buffer is fresher.
  useEffect(() => {
    if (!chatId || !userId.current) return;
    const k = chatKey(chatId);
    if (streamingRef.current[k]) return;
    apiFetch(`${API_URL}/chats/${userId.current}/${chatId}/messages`)
      .then((r) => (r.ok ? r.json() : []))
      .then((msgs) => {
        // a stream may have started while this was in flight; its buffer
        // is fresher than the server transcript — don't wipe it
        if (streamingRef.current[k]) return;
        setThreads((t) => ({ ...t, [k]: msgs }));
      })
      .catch(() => {});
  }, [chatId]);

  useEffect(() => {
    bottom.current?.scrollIntoView();
  }, [messages]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setDrawer(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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

  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    const k = chatKey(chatId); // captured: every write below targets THIS chat
    if (!text || streaming[k]) return;
    if (textArg === undefined) setInput("");
    // The onboarding kickoff is machine-sent: no user bubble, and the server
    // hides it from transcript replays too.
    const hidden = text === ONBOARD_KICKOFF;
    setStreamFlag(k, true);
    updateThread(k, (m) => [
      ...m,
      ...(hidden ? [] : [{ role: "user" as const, content: text }]),
      { role: "assistant" as const, content: "" },
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
      // Text streamed before a tool event is search narration ("Let me
      // search…"), not answer — drop it when the tool fires (issue #45).
      // Only the post-last-tool segment survives as the answer bubble —
      // unless nothing follows the last tool (the model replied first,
      // then called e.g. complete_onboarding): then the "narration" was
      // the reply, and it comes back at stream end (issue #46).
      let bubbleText = ""; // what has streamed into the current bubble
      let wiped = ""; // last pre-tool segment, restored if the stream ends bare
      // Ads arrive mid-stream, right after their tool event. Hold every one
      // until the stream completes, then attach to the trailing (answer)
      // bubble — below the finished text — so cards never interleave with
      // tokens still streaming in (issues #39, #41).
      let heldAds: Ad[] = [];
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
            bubbleText += parsed.text;
            patchLast(k, (last) => ({
              ...last,
              status: undefined,
              content: last.content + parsed.text,
            }));
          } else if (parsed.type === "tool") {
            if (bubbleText) {
              // wipe the narration that streamed ahead of this tool call
              wiped = bubbleText;
              patchLast(k, (last) => ({ ...last, content: "" }));
              bubbleText = "";
            }
          } else if (parsed.type === "tool_start") {
            patchLast(k, (last) => ({
              ...last,
              status: TOOL_STATUS[parsed.name as string] ?? "Working…",
            }));
          } else if (parsed.type === "citations" && parsed.citations.length > 0) {
            patchLast(k, (last) => ({ ...last, citations: parsed.citations }));
          } else if (parsed.type === "ad") {
            heldAds.push(parsed as Ad); // attached after the stream completes
          } else if (parsed.type === "error") {
            throw new Error("server reported a stream error");
          }
          // "ping" events just feed the watchdog
        }
      }
      if (!bubbleText && wiped) {
        // the turn ended on a tool call — the wiped text was the reply
        const text = wiped;
        patchLast(k, (last) => ({ ...last, status: undefined, content: text }));
      }
      if (heldAds.length) {
        // stream done: cards land at the end of the answer bubble in one go
        const ads = heldAds;
        heldAds = [];
        patchLast(k, (last) => ({ ...last, ads: [...(last.ads ?? []), ...ads] }));
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
      if (onboardingRef.current) {
        // unlock the moment the turn's complete_onboarding call landed
        apiFetch(`${API_URL}/garage/${userId.current}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((g) => {
            if (g && !needsOnboarding(g)) {
              onboardingRef.current = false;
              setOnboarding(false);
            }
          })
          .catch(() => {});
      }
    }
  }

  function retry(text: string) {
    const k = chatKey(chatId);
    // Drop the whole failed turn — every trailing assistant bubble plus its
    // user message — then resend.
    updateThread(k, (msgs) => {
      if (!msgs.length || !msgs[msgs.length - 1].error) return msgs;
      let i = msgs.length;
      while (i > 0 && msgs[i - 1].role === "assistant") i--;
      return msgs.slice(0, Math.max(i - 1, 0));
    });
    send(text);
  }

  return (
    <main>
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
        {/* onboarding locks the site to this conversation: no drawer, no nav */}
        {!onboarding && (
          <button type="button" className="menubtn" aria-label="Chats" onClick={openDrawer}>
            ☰
          </button>
        )}
        Ask MustangDriver
        {!onboarding && (
          <nav>
            <AuthButton />
            <Link href="/profile">Profile</Link>
            <Link href="/garage">My Garage</Link>
          </nav>
        )}
      </header>
      <div className="messages">
        {/* only after the transcript fetch lands, so history never flashes it */}
        {!onboarding && threads[chatKey(chatId)] !== undefined && messages.length === 0 && (
          <div className="welcome">
            <p>
              Your Mustang copilot — maintenance, mods, recalls, history, and
              what to buy next.
            </p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} type="button" onClick={() => send(q)}>
                  {q}
                </button>
              ))}
            </div>
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
          placeholder={onboarding ? "Type your answer…" : "Ask about Mustangs…"}
          aria-label="Message"
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
