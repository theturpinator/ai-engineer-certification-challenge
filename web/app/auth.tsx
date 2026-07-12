"use client";

// Optional Google sign-in (issue #17). Everything here is inert unless
// NEXT_PUBLIC_GOOGLE_CLIENT_ID is set at build time: apiFetch degrades to
// plain fetch and AuthButton renders nothing, so the app builds and runs
// cleanly before the owner creates the OAuth client.

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
const GSI_SRC = "https://accounts.google.com/gsi/client";

type Auth = { token: string; user_id: string; name?: string; picture?: string };

type GoogleId = {
  initialize: (cfg: {
    client_id: string;
    callback: (resp: { credential: string }) => void;
  }) => void;
  renderButton: (el: HTMLElement, opts: Record<string, unknown>) => void;
};

declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

function getAuth(): Auth | null {
  try {
    const raw = localStorage.getItem("md_auth");
    return raw ? (JSON.parse(raw) as Auth) : null;
  } catch {
    return null;
  }
}

/** fetch that attaches the app JWT when signed in. On a 401 the token is
 * dead (expired/revoked): clear it and retry once anonymously, so a stale
 * login never blocks the app. */
export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  const auth = typeof window === "undefined" ? null : getAuth();
  if (!auth) return fetch(input, init);
  const r = await fetch(input, {
    ...init,
    headers: {
      ...(init?.headers as Record<string, string>),
      Authorization: `Bearer ${auth.token}`,
    },
  });
  if (r.status !== 401) return r;
  localStorage.removeItem("md_auth");
  return fetch(input, init);
}

/** After login/restore the server names the canonical user_id. A change means
 * this browser now acts as the signed-in user; reload so chats/garage refetch
 * under it (md_user_id is read on mount everywhere). */
function adoptUserId(userId: string) {
  if (localStorage.getItem("md_user_id") !== userId) {
    localStorage.setItem("md_user_id", userId);
    window.location.reload();
  }
}

export function AuthButton() {
  const [auth, setAuth] = useState<Auth | null>(null);
  const [ready, setReady] = useState(false); // localStorage read (post-hydration)
  const slot = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) return;
    const stored = getAuth();
    setAuth(stored);
    setReady(true);
    if (!stored) return;
    // validate/restore the session; a dead token falls back to anonymous
    fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${stored.token}` },
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((me) => {
        const next = { ...stored, user_id: me.user_id, name: me.name, picture: me.picture };
        localStorage.setItem("md_auth", JSON.stringify(next));
        setAuth(next);
        adoptUserId(me.user_id);
      })
      .catch(() => {
        localStorage.removeItem("md_auth");
        setAuth(null);
      });
  }, []);

  // Signed out: load GIS (once) and render the official Google button.
  useEffect(() => {
    if (!GOOGLE_CLIENT_ID || !ready || auth || !slot.current) return;
    const el = slot.current;

    async function onCredential(resp: { credential: string }) {
      let anon = localStorage.getItem("md_user_id");
      if (!anon) {
        anon = crypto.randomUUID();
        localStorage.setItem("md_user_id", anon);
      }
      const r = await fetch(`${API_URL}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: resp.credential, anon_user_id: anon }),
      });
      if (!r.ok) return; // leave the button; the user can retry
      const d = await r.json();
      const next: Auth = { token: d.token, user_id: d.user_id, name: d.name, picture: d.picture };
      localStorage.setItem("md_auth", JSON.stringify(next));
      setAuth(next);
      adoptUserId(d.user_id); // reloads when the canonical id differs
    }

    const render = () => {
      window.google?.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID!,
        callback: onCredential,
      });
      window.google?.accounts.id.renderButton(el, {
        theme: "filled_black",
        size: "small",
        shape: "pill",
        text: "signin",
      });
    };
    let script = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`);
    if (!script) {
      script = document.createElement("script");
      script.src = GSI_SRC;
      script.async = true;
      document.head.appendChild(script);
    }
    if (window.google?.accounts?.id) render();
    else script.addEventListener("load", render);
    return () => script?.removeEventListener("load", render);
  }, [ready, auth]);

  if (!GOOGLE_CLIENT_ID || !ready) return null;
  if (auth) {
    return (
      <span className="authbox">
        {auth.picture && (
          // eslint-disable-next-line @next/next/no-img-element
          <img className="avatar" src={auth.picture} alt="" referrerPolicy="no-referrer" />
        )}
        <span className="authname">{(auth.name || "").split(" ")[0]}</span>
        <button
          type="button"
          className="signout"
          onClick={() => {
            // md_user_id stays: this browser keeps its current garage
            localStorage.removeItem("md_auth");
            setAuth(null);
          }}
        >
          Sign out
        </button>
      </span>
    );
  }
  return <span className="authbox"><span ref={slot} className="gsislot" /></span>;
}
