"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiFetch, AuthButton } from "../auth";
import { DeltaChips, type Deltas } from "../chips";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Stats = {
  power?: number;
  acceleration?: number;
  top_speed?: number;
  handling?: number;
  braking?: number;
  style?: number;
  comfort?: number;
  safety?: number;
  reliability?: number;
  hp?: number;
  zero_to_sixty?: number;
  top_speed_mph?: number;
  nhtsa?: { stars: number; vehicle?: string; url: string };
};
type Bars = { current: Record<string, number>; dream: Record<string, number> };
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
  bars?: Bars | null;
  photo_uploaded?: boolean; // the owner's own photo is stored -> no upload pill
};
type Profile = { cars?: Car[]; goals?: string[] };
type Garage = {
  profile?: Profile;
  instructions?: string[];
};

const EMPTY = "Nothing here yet — just mention it in chat.";
const BAR_GROUPS: [string, [keyof Stats, string][]][] = [
  ["Performance", [
    ["power", "Power"],
    ["acceleration", "Accel"],
    ["top_speed", "Top Speed"],
    ["handling", "Handling"],
    ["braking", "Braking"],
  ]],
  ["Ownership", [
    ["style", "Style"],
    ["comfort", "Comfort"],
    ["safety", "Safety"],
    ["reliability", "Reliability"],
  ]],
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

function Portrait({ url, children }: { url: string; children?: React.ReactNode }) {
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
        alt="Portrait of this car"
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
      {children}
    </div>
  );
}

function UploadPhoto({
  uid,
  carId,
  onUploaded,
}: {
  uid: string;
  carId: string;
  onUploaded: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  // Always-visible pill overlaid on the portrait (no hover-reveal — mobile
  // has no hover); the parent hides it entirely once a photo is uploaded.
  return (
    <div className="uploadoverlay">
      {err && <span className="formerr">{err}</span>}
      <label className="uploadpill">
        {busy ? "Uploading…" : "Use my own photo"}
        <input
          type="file"
          accept="image/*"
          disabled={busy}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (!file) return;
            setBusy(true);
            setErr("");
            try {
              const r = await apiFetch(`${API_URL}/garage/${uid}/cars/${carId}/image`, {
                method: "PUT",
                headers: { "Content-Type": file.type || "image/jpeg" },
                body: file,
                signal: AbortSignal.timeout(120_000), // big photos beat the 30s default
              });
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
              onUploaded();
            } catch {
              setErr("Upload failed — try a smaller image (under 8 MB).");
            } finally {
              setBusy(false);
            }
          }}
        />
      </label>
    </div>
  );
}

function StatBars({ stats, bars }: { stats: Stats; bars?: Bars | null }) {
  const figures = [
    stats.hp != null && `${stats.hp} hp`,
    stats.zero_to_sixty != null && `0–60 in ${stats.zero_to_sixty}s`,
    stats.top_speed_mph != null && `${stats.top_speed_mph} mph top`,
  ]
    .filter(Boolean)
    .join(" · ");
  let anyDream = false;
  const groups = BAR_GROUPS.map(([title, group]) => {
    // Old cached baselines lack the ownership stats until the background
    // enrichment regenerates them; hide the group rather than show zeros.
    if (group.every(([key]) => stats[key] == null)) return null;
    const rows = group.map(([key, label]) => {
      const stock = Math.max(0, Math.min(100, Number(stats[key]) || 0));
      const current = bars?.current?.[key] ?? stock;
      const dream = Math.max(bars?.dream?.[key] ?? current, current);
      if (dream > current) anyDream = true;
      return (
        <div className="stat" key={key}>
          <span className="label">{label}</span>
          <div className="bar">
            {dream > current && (
              <div className="dreamfill" style={{ width: `${dream}%` }} />
            )}
            <div className="fill" style={{ width: `${current}%` }} />
          </div>
          <span className="num">
            {current}
            {dream > current && <span className="dreamnum"> →{dream}</span>}
          </span>
        </div>
      );
    });
    return (
      <div key={title} className="statgroup">
        <p className="statshead">{title}</p>
        <div className="stats">{rows}</div>
      </div>
    );
  });
  return (
    <>
      {figures && <p className="figures">Stock: {figures}</p>}
      {groups}
      {stats.nhtsa && (
        <p className="empty">
          Safety grounded in the{" "}
          <a href={stats.nhtsa.url} target="_blank" rel="noopener noreferrer">
            NHTSA {stats.nhtsa.stars}-star overall safety rating
          </a>
          {stats.nhtsa.vehicle ? ` (${stats.nhtsa.vehicle})` : ""}.
        </p>
      )}
      {anyDream && (
        <p className="empty">
          Solid bar: current build · light extension: with your wishlist.
        </p>
      )}
    </>
  );
}

type ShopRow = {
  id: string;
  name: string;
  advertiser: string | null;
  sponsored: boolean;
  description: string;
  categories: string[];
  image: string | null;
  link: string | null;
  deltas: Deltas | null;
  installed: boolean;
  wishlisted: boolean;
};

function ShopRowView({
  row,
  busy,
  onAct,
}: {
  row: ShopRow;
  busy: boolean;
  onAct: (row: ShopRow, field: "mods" | "wishlist") => void;
}) {
  return (
    <div className="shoprow">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {row.image && <img src={row.image} alt={`${row.advertiser} ad`} />}
      <div className="shopinfo">
        {row.sponsored && (
          <span className="sponsoredtag">Sponsored · {row.advertiser}</span>
        )}
        {row.link ? (
          <a href={row.link} target="_blank" rel="noopener noreferrer sponsored">
            <strong>{row.name}</strong>
          </a>
        ) : (
          <strong>{row.name}</strong>
        )}
        <p>{row.description}</p>
        <DeltaChips deltas={row.deltas} />
        <div className="shopactions">
          <button
            type="button"
            disabled={row.installed || busy}
            onClick={() => onAct(row, "mods")}
          >
            {row.installed ? "✓ Installed" : "I have this"}
          </button>
          <button
            type="button"
            className="secondary"
            disabled={row.wishlisted || row.installed || busy}
            onClick={() => onAct(row, "wishlist")}
          >
            {row.wishlisted ? "✓ On wishlist" : "Add to wishlist"}
          </button>
        </div>
      </div>
    </div>
  );
}

function UpgradeShop({
  uid,
  car,
  onCarUpdated,
}: {
  uid: string;
  car: Car;
  onCarUpdated: (car: Car) => void;
}) {
  const [shop, setShop] = useState<{ recommended: ShopRow[]; catalog: ShopRow[] } | null>(
    null
  );
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    apiFetch(`${API_URL}/garage/${uid}/cars/${car.id}/shop`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setShop)
      .catch(() => {});
  }, [uid, car]);

  async function act(row: ShopRow, field: "mods" | "wishlist") {
    setBusy(row.id + field);
    try {
      const list = (field === "mods" ? car.mods : car.wishlist) ?? [];
      const r = await apiFetch(`${API_URL}/garage/${uid}/cars/${car.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: [...list, row.name] }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      onCarUpdated(await r.json()); // bars + lists update immediately
    } catch {
      /* row stays actionable; the user can retry */
    } finally {
      setBusy("");
    }
  }

  if (!shop) return null;
  const q = query.trim().toLowerCase();
  const filtered = shop.catalog.filter(
    (r) =>
      !q ||
      [r.name, r.advertiser ?? "", r.description, ...r.categories]
        .join(" ")
        .toLowerCase()
        .includes(q)
  );
  return (
    <section className="card shop">
      <h2>Upgrade Shop</h2>
      {shop.recommended.length > 0 && (
        <>
          <p className="empty">Recommended for this car:</p>
          {shop.recommended.map((row) => (
            <ShopRowView key={row.id} row={row} busy={busy !== ""} onAct={act} />
          ))}
        </>
      )}
      <button type="button" className="shoptoggle" onClick={() => setOpen(!open)}>
        {open ? "Hide the full catalog" : `Browse all upgrades (${shop.catalog.length})`}
      </button>
      {open && (
        <>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search upgrades…"
            aria-label="Search upgrades"
          />
          {filtered.map((row) => (
            <ShopRowView key={row.id} row={row} busy={busy !== ""} onAct={act} />
          ))}
          {filtered.length === 0 && <p className="empty">No upgrades match.</p>}
        </>
      )}
    </section>
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
  onDeleted,
  onCancel,
}: {
  car: Car;
  uid: string;
  onSaved: (car: Car) => void;
  onDeleted: () => void;
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

  async function del() {
    if (!confirm("Delete this car? Its photo and stats go with it.")) return;
    setSaving(true);
    setErr("");
    try {
      const r = await apiFetch(`${API_URL}/garage/${uid}/cars/${car.id}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      onDeleted();
    } catch {
      setErr("Couldn’t delete — please try again.");
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
        <button type="button" className="danger" onClick={del} disabled={saving}>
          Delete car
        </button>
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
    !garage.instructions?.length;

  function patchCar(id: string, fn: (c: Car) => Car) {
    setGarage((g) =>
      g
        ? {
            ...g,
            profile: {
              ...g.profile,
              cars: (g.profile?.cars ?? []).map((c) => (c.id === id ? fn(c) : c)),
            },
          }
        : g
    );
  }

  function onSaved(updated: Car) {
    // the PATCH response has no photo_uploaded; keep what we know
    patchCar(updated.id, (c) => ({ ...updated, photo_uploaded: c.photo_uploaded }));
    setEditing(false);
    setTick((t) => (t < 3 ? t : t - 1)); // re-arm the stats refetch
  }

  function onDeleted(id: string) {
    setGarage((g) =>
      g
        ? {
            ...g,
            profile: {
              ...g.profile,
              cars: (g.profile?.cars ?? []).filter((c) => c.id !== id),
            },
          }
        : g
    );
    setEditing(false);
    setActive(0);
  }

  return (
    <main>
      <header>
        My Garage
        <nav>
          <AuthButton />
          <Link href="/profile">Profile</Link>
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
            {cars.length >= 10 && (
              <p className="card empty">
                Garage is full — max 10 cars. Delete a car to free a slot.
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
                  >
                    {!car.photo_uploaded && (
                      <UploadPhoto
                        uid={uid}
                        carId={car.id}
                        onUploaded={() => {
                          setImgVer((v) => v + 1);
                          patchCar(car.id, (c) => ({ ...c, photo_uploaded: true }));
                        }}
                      />
                    )}
                  </Portrait>
                  {car.stats ? (
                    <StatBars stats={car.stats} bars={car.bars} />
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
                    onDeleted={() => onDeleted(car.id)}
                    onCancel={() => setEditing(false)}
                  />
                ) : (
                  <>
                    <UpgradeShop uid={uid} car={car} onCarUpdated={onSaved} />
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
          </>
        )}
      </div>
    </main>
  );
}
