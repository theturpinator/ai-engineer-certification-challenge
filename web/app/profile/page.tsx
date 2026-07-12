"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiFetch, AuthButton } from "../auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Car = {
  year?: number | string;
  trim?: string;
  generation?: string;
  color?: string;
  nickname?: string;
  mods?: string[];
};
type Garage = {
  profile?: { cars?: Car[]; goals?: string[] };
  instructions?: string[];
  summaries?: { summary: string; date: string }[];
};

const EMPTY = "Nothing here yet — just mention it in chat.";

function carLine(c: Car) {
  const name = [c.year, "Mustang", c.trim].filter(Boolean).join(" ");
  const details = [
    c.nickname && `“${c.nickname}”`,
    c.color,
    c.mods?.length ? `${c.mods.length} mod${c.mods.length > 1 ? "s" : ""}` : null,
  ].filter(Boolean);
  return details.length ? `${name} — ${details.join(" · ")}` : name;
}

function MemGroup({
  title,
  kind,
  items,
}: {
  title: string;
  kind: string;
  items: React.ReactNode[];
}) {
  return (
    <div className="statgroup">
      <p className="statshead">{title}</p>
      <p className="memkind">{kind}</p>
      {items.length ? (
        <ul>
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="empty">{EMPTY}</p>
      )}
    </div>
  );
}

export default function ProfilePage() {
  const [garage, setGarage] = useState<Garage | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    // ?uid=<uuid> override for demos/tests; otherwise the browser identity
    const override = new URLSearchParams(window.location.search).get("uid");
    let id = override || localStorage.getItem("md_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("md_user_id", id);
    }
    apiFetch(`${API_URL}/garage/${id}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setGarage)
      .catch(() => setError(true));
  }, []);

  const goals = garage?.profile?.goals ?? [];
  const cars = garage?.profile?.cars ?? [];
  const instructions = garage?.instructions ?? [];
  const summaries = garage?.summaries ?? [];

  return (
    <main>
      <header>
        My Profile
        <nav>
          <AuthButton />
          <Link href="/">Chat</Link>
          <Link href="/garage">My Garage</Link>
        </nav>
      </header>
      <div className="garage">
        {!garage && !error && <p className="empty">Loading…</p>}
        {error && (
          <p className="card empty">
            Couldn’t load your profile. Please try again in a moment.
          </p>
        )}
        {garage && (
          <>
            <section className="card">
              <h2>Goals</h2>
              {goals.length ? (
                <ul>
                  {goals.map((g) => (
                    <li key={g}>{g}</li>
                  ))}
                </ul>
              ) : (
                <p className="empty">{EMPTY}</p>
              )}
            </section>
            <section className="card">
              <h2>What I know about you</h2>
              <MemGroup
                title="Your garage"
                kind="Semantic memory — facts about your cars"
                items={cars.map((c) => carLine(c))}
              />
              <MemGroup
                title="Preferences"
                kind="Procedural memory — how you like your answers"
                items={instructions}
              />
              <MemGroup
                title="Recent conversations"
                kind="Episodic memory — what we talked about lately"
                items={summaries.map((s) => (
                  <>
                    {s.summary} <span className="memdate">{s.date}</span>
                  </>
                ))}
              />
            </section>
          </>
        )}
      </div>
    </main>
  );
}
