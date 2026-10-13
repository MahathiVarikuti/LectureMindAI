"use client";

const SESSION_KEY = "lecturemind_session_id";

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

// Session isolation is per-browser via UUID in localStorage.
// Clearing storage or switching browsers loses access to prior data.
// This is NOT real authentication — see docs/security.md.
