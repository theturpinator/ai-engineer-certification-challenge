"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiFetch, AuthButton } from "../auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Stats = {
  power?: number;
  acceleration?: number;
  top_speed?: number;
  handling?: number;
  braking?: number;
  hp?: number;
  zero_to_sixty?: number;
  top_speed_mph?: number;
  modified?: boolean;
};
type Car = {
  id: string;
  year?: number | string;
  trim?: string;
  generation?: string;
  color?: string;
  nickname?: string;
  mods?: string[];
  wishlist?: string[];
  stats?: Stats | null;
};
type Profile = { cars?: Car[]; goals?: string[] };
type Garage = {
  profile?: Profile;
  instructions?: string[];
  summaries?: { summary: string; date: string }[];
};

const EMPTY = "Nothing here yet — just mention it in chat.";
const BARS: [keyof Stats, string][] = [
  ["power", "Power"],
  ["acceleration", "Accel"],
  ["top_speed", "Top Speed"],
  ["handling", "Handling"],
  ["braking", "Braking"],
];

function carLabel(c: Car) {
  return c.nickname || [c.year, c.trim].filter(Boolean).join(" ") || "Mustang";
}

function carTitle(c: Car) {
  const name = [c.year, "Mustang", c.trim].filter(Boolean).join(" ");
  return c.generation ? `${name} (${c.generation})` : name;
}

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

const RETRY_MS = [8000, 20000, 40000]; // portrait generation takes ~30-60s

function Portrait({ url }: { url: string }) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<"loading" | "ok" | "pending">("loading");

  useEffect(() => {
    setAttempt(0);
    setState("loading");
  }, [url]);

  return (
    <div className="portraitbox">
      {state !== "ok" && (
        <div className="portrait placeholder">
          {state === "pending" ? "Portrait generating…" : "…"}
        </div>
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        className="portrait"
        style={state === "ok" ? undefined : { display: "none" }}
        src={`${url}&r=${attempt}`}
        alt="AI-generated portrait of this car"
        // cached images can finish before React attaches onLoad
        ref={(el) => {
          if (el?.complete && el.naturalWidth > 0) setState("ok");
        }}
        onLoad={() => setState("ok")}
        onError={() => {
          setState("pending");
          const delay = RETRY_MS[attempt];
          if (delay)
            setTimeout(() => {
              setAttempt((a) => a + 1);
              setState("loading");
            }, delay);
        }}
      />
    </div>
  );
}

function StatBars({ stats }: { stats: Stats }) {
  const figures = [
    stats.hp != null && `${stats.hp} hp`,
    stats.zero_to_sixty != null && `0–60 in ${stats.zero_to_sixty}s`,
    stats.top_speed_mph != null && `${stats.top_speed_mph} mph top`,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <>
      {figures && <p className="figures">Stock: {figures}</p>}
      <div className="stats">
        {BARS.map(([key, label]) => {
          const v = Math.max(0, Math.min(100, Number(stats[key]) || 0));
          return (
            <div className="stat" key={key}>
              <span className="label">{label}</span>
              <div className="bar">
                <div style={{ width: `${v}%` }} />
              </div>
              <span className="num">{v}</span>
            </div>
          );
        })}
      </div>
      {stats.modified && (
        <p className="empty">Ratings reflect this car’s installed mods.</p>
      )}
    </>
  );
}

function ChipEditor({
  label,
  items,
  onChange,
}: {
  label: string;
  items: string[];
  onChange: (items: string[]) => void;
}) {
  const [text, setText] = useState("");
  const add = () => {
    const t = text.trim();
    if (t && !items.includes(t)) onChange([...items, t]);
    setText("");
  };
  return (
    <div className="chipedit">
      <span className="fieldlabel">{label}</span>
      {items.length > 0 && (
        <div className="chips">
          {items.map((it) => (
            <span className="chip" key={it}>
              {it}
              <button
                type="button"
                aria-label={`Remove ${it}`}
                onClick={() => onChange(items.filter((x) => x !== it))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="chipadd">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              add();
            }
          }}
          placeholder="Add one…"
          aria-label={`Add to ${label}`}
        />
        <button type="button" onClick={add}>
          Add
        </button>
      </div>
    </div>
  );
}

function EditCar({
  car,
  uid,
  onSaved,
  onCancel,
}: {
  car: Car;
  uid: string;
  onSaved: (car: Car) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState({
    year: car.year?.toString() ?? "",
    trim: car.trim ?? "",
    generation: car.generation ?? "",
    color: car.color ?? "",
    nickname: car.nickname ?? "",
    mods: car.mods ?? [],
    wishlist: car.wishlist ?? [],
  });
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setDraft((d) => ({ ...d, [k]: e.target.value }));

  async function save() {
    setSaving(true);
    setErr("");
    const year = parseInt(draft.year, 10);
    try {
      const r = await apiFetch(`${API_URL}/garage/${uid}/cars/${car.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          // null clears a field; an unparseable year is simply left unchanged
          year: draft.year.trim() ? (Number.isFinite(year) ? year : undefined) : null,
          trim: draft.trim.trim() || null,
          generation: draft.generation.trim() || null,
          color: draft.color.trim() || null,
          nickname: draft.nickname.trim() || null,
          mods: draft.mods,
          wishlist: draft.wishlist,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      onSaved(await r.json());
    } catch {
      setErr("Couldn’t save — check the values (year 1964–2027) and try again.");
      setSaving(false);
    }
  }

  return (
    <section className="card editform">
      <h2>Edit car</h2>
      <div className="editgrid">
        {(
          [
            ["year", "Year"],
            ["trim", "Trim"],
            ["generation", "Generation"],
            ["color", "Color"],
            ["nickname", "Nickname"],
          ] as const
        ).map(([k, label]) => (
          <label key={k}>
            <span className="fieldlabel">{label}</span>
            <input
              value={draft[k]}
              onChange={set(k)}
              inputMode={k === "year" ? "numeric" : undefined}
            />
          </label>
        ))}
      </div>
      <ChipEditor
        label="Installed Mods"
        items={draft.mods}
        onChange={(mods) => setDraft((d) => ({ ...d, mods }))}
      />
      <ChipEditor
        label="Wishlist"
        items={draft.wishlist}
        onChange={(wishlist) => setDraft((d) => ({ ...d, wishlist }))}
      />
      {err && <p className="formerr">{err}</p>}
      <div className="editactions">
        <button type="button" className="secondary" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
        <button type="button" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </section>
  );
}

export default function GaragePage() {
  const [garage, setGarage] = useState<Garage | null>(null);
  const [error, setError] = useState(false);
  const [uid, setUid] = useState("");
  const [active, setActive] = useState(0);
  const [editing, setEditing] = useState(false);
  const [imgVer, setImgVer] = useState(0);
  const [tick, setTick] = useState(0); // stats arrive async -> a few refetches

  useEffect(() => {
    // ?uid=<uuid> override for demos/tests; otherwise the browser identity
    const override = new URLSearchParams(window.location.search).get("uid");
    let id = override || localStorage.getItem("md_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("md_user_id", id);
    }
    setUid(id);
  }, []);

  useEffect(() => {
    if (!uid) return;
    apiFetch(`${API_URL}/garage/${uid}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((g: Garage) => {
        setGarage(g);
        if (tick < 3 && g.profile?.cars?.some((c) => !c.stats)) {
          setTimeout(() => setTick((t) => t + 1), 10000);
        }
      })
      .catch(() => setError(true));
  }, [uid, tick]);

  const profile = garage?.profile ?? {};
  const cars = profile.cars ?? [];
  const activeIdx = Math.min(active, Math.max(cars.length - 1, 0));
  const car = cars[activeIdx];
  const isEmpty =
    garage !== null &&
    !cars.length &&
    !profile.goals?.length &&
    !garage.instructions?.length &&
    !garage.summaries?.length;

  function onSaved(updated: Car) {
    setGarage((g) =>
      g
        ? {
            ...g,
            profile: {
              ...g.profile,
              cars: cars.map((c, i) => (i === activeIdx ? updated : c)),
            },
          }
        : g
    );
    setEditing(false);
    setImgVer((v) => v + 1); // identity/color edits regenerate the portrait
    setTick((t) => (t < 3 ? t : t - 1)); // re-arm the stats refetch
  }

  return (
    <main>
      <header>
        My Garage
        <nav>
          <AuthButton />
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
                assistant about your Mustangs, your mods, or your plans, and they’ll show
                up here automatically. You can fine-tune any detail here afterwards.
              </p>
            )}
            {cars.length > 1 && (
              <div className="cartabs">
                {cars.map((c, i) => (
                  <button
                    key={c.id}
                    className={i === activeIdx ? "active" : ""}
                    aria-pressed={i === activeIdx}
                    onClick={() => {
                      setActive(i);
                      setEditing(false);
                    }}
                  >
                    {carLabel(c)}
                  </button>
                ))}
              </div>
            )}
            {car ? (
              <>
                <section className="card carcard">
                  <div className="carhead">
                    <h2>{carTitle(car)}</h2>
                    {!editing && (
                      <button
                        type="button"
                        className="editbtn"
                        onClick={() => setEditing(true)}
                      >
                        ✎ Edit
                      </button>
                    )}
                  </div>
                  {(car.nickname || car.color) && (
                    <p className="carmeta">
                      {[car.nickname && `“${car.nickname}”`, car.color]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                  )}
                  <Portrait
                    url={`${API_URL}/garage/${uid}/cars/${car.id}/image?v=${imgVer}`}
                  />
                  {car.stats ? (
                    <StatBars stats={car.stats} />
                  ) : (
                    <p className="empty">Crunching the numbers for this car…</p>
                  )}
                </section>
                {editing ? (
                  <EditCar
                    key={car.id}
                    car={car}
                    uid={uid}
                    onSaved={onSaved}
                    onCancel={() => setEditing(false)}
                  />
                ) : (
                  <>
                    <ListSection title="Installed Mods" items={car.mods} />
                    <ListSection title="Wishlist" items={car.wishlist} />
                  </>
                )}
              </>
            ) : (
              <section className="card">
                <h2>My Cars</h2>
                <p className="empty">{EMPTY}</p>
              </section>
            )}
            <ListSection title="Goals" items={profile.goals} />
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
