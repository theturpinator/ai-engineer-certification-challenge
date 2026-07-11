"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Profile = {
  year?: number | string;
  trim?: string;
  generation?: string;
  mods?: string[];
  wishlist?: string[];
  goals?: string[];
};
type Garage = {
  profile?: Profile;
  instructions?: string[];
  summaries?: { summary: string; date: string }[];
};

const EMPTY = "Nothing here yet — just mention it in chat.";

function ListSection({ title, items }: { title: string; items?: string[] }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {items?.length ? (
        <ul>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="empty">{EMPTY}</p>
      )}
    </section>
  );
}

export default function GaragePage() {
  const [garage, setGarage] = useState<Garage | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let id = localStorage.getItem("md_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("md_user_id", id);
    }
    fetch(`${API_URL}/garage/${id}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setGarage)
      .catch(() => setError(true));
  }, []);

  const p = garage?.profile ?? {};
  const hasCar = Boolean(p.year || p.trim || p.generation);
  const isEmpty =
    garage !== null &&
    !hasCar &&
    !p.mods?.length &&
    !p.wishlist?.length &&
    !p.goals?.length &&
    !garage.instructions?.length &&
    !garage.summaries?.length;

  return (
    <main>
      <header>
        My Garage
        <nav>
          <Link href="/">Chat</Link>
        </nav>
      </header>
      <div className="garage">
        {!garage && !error && <p className="empty">Loading…</p>}
        {error && (
          <p className="card empty">Couldn’t load your garage. Please try again in a moment.</p>
        )}
        {garage && (
          <>
            {isEmpty && (
              <p className="card">
                Your garage is empty for now. It fills itself in as you chat — tell the
                assistant about your Mustang, your mods, or your plans, and they’ll show up
                here automatically. No forms, ever.
              </p>
            )}
            <section className="card">
              <h2>My Car</h2>
              {hasCar ? (
                <p>
                  {[p.year, "Mustang", p.trim].filter(Boolean).join(" ")}
                  {p.generation ? ` (${p.generation})` : ""}
                </p>
              ) : (
                <p className="empty">{EMPTY}</p>
              )}
            </section>
            <ListSection title="Installed Mods" items={p.mods} />
            <ListSection title="Wishlist" items={p.wishlist} />
            <ListSection title="Goals" items={p.goals} />
            <ListSection title="Preferences" items={garage.instructions} />
            <section className="card">
              <h2>Recent Conversations</h2>
              {garage.summaries?.length ? (
                <ul>
                  {garage.summaries.map((s, i) => (
                    <li key={i}>
                      <strong>{s.date}</strong> — {s.summary}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty">No past conversations yet.</p>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
