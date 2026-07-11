"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = { title: string; url: string };
type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };

const NewTabLink = (props: React.ComponentProps<"a">) => (
  <a {...props} target="_blank" rel="noopener noreferrer" />
);

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const userId = useRef<string>("");
  const loaded = useRef(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let id = localStorage.getItem("md_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("md_user_id", id);
    }
    userId.current = id;
    const saved = localStorage.getItem(`md_chat_${id}`);
    if (saved) setMessages(JSON.parse(saved));
    loaded.current = true;
  }, []);

  useEffect(() => {
    if (loaded.current && userId.current) {
      localStorage.setItem(`md_chat_${userId.current}`, JSON.stringify(messages));
    }
    bottom.current?.scrollIntoView();
  }, [messages]);

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
        body: JSON.stringify({ message: text, user_id: userId.current }),
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
      <header>Ask MustangDriver</header>
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
