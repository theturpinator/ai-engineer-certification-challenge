"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = { title: string; url: string };
type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };
type ChatMeta = { chat_id: string; title: string; updated_at: string };

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [chats, setChats] = useState<ChatMeta[]>([]);
  const [drawer, setDrawer] = useState(false);
  const [chatId, setChatId] = useState("");
  const userId = useRef<string>("");
  const sessionId = useRef<string>("");
  const bottom = useRef<HTMLDivElement>(null);

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
  }, []);

  // The server transcript is the source of truth when (re)opening a chat.
  useEffect(() => {
    if (!chatId || !userId.current) return;
    fetch(`${API_URL}/chats/${userId.current}/${chatId}/messages`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setMessages)
      .catch(() => {});
  }, [chatId]);

  useEffect(() => {
    bottom.current?.scrollIntoView();
  }, [messages]);

  function openDrawer() {
    setDrawer(true);
    fetch(`${API_URL}/chats/${userId.current}`)
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
    setMessages([]);
    setChatId(id);
  }

  function newChat() {
    switchTo(crypto.randomUUID().slice(0, 8));
  }

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          user_id: userId.current,
          session_id: sessionId.current,
          chat_id: chatId || "default",
        }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop()!;
        for (const event of events) {
          const data = event.replace(/^data: /, "");
          if (data === "[DONE]") continue;
          const parsed = JSON.parse(data);
          if (parsed.type === "token") {
            setMessages((m) => {
              const last = m[m.length - 1];
              return [...m.slice(0, -1), { ...last, content: last.content + parsed.text }];
            });
          } else if (parsed.type === "citations" && parsed.citations.length > 0) {
            setMessages((m) => {
              const last = m[m.length - 1];
              return [...m.slice(0, -1), { ...last, citations: parsed.citations }];
            });
          }
        }
      }
    } catch (err) {
      setMessages((m) => {
        const last = m[m.length - 1];
        return [
          ...m.slice(0, -1),
          { ...last, content: last.content || `Something went wrong (${err}). Please try again.` },
        ];
      });
    } finally {
      setBusy(false);
    }
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
        <button type="button" className="menubtn" aria-label="Chats" onClick={openDrawer}>
          ☰
        </button>
        Ask MustangDriver
        <nav>
          <Link href="/garage">My Garage</Link>
        </nav>
      </header>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`}>
            {msg.role === "assistant" ? (
              <ReactMarkdown components={{ a: NewTabLink }}>
                {msg.content || "…"}
              </ReactMarkdown>
            ) : (
              msg.content
            )}
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
      <form onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
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
