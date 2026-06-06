"use client";
import { useState, useRef } from "react";

// Vapi SDK needs two separate values:
// PUBLIC_KEY  → Dashboard → API Keys → "Public Key" (client-safe, starts with different prefix)
// ASSISTANT_ID → Dashboard → Assistants → your assistant's ID (UUID)
const PUBLIC_KEY   = process.env.NEXT_PUBLIC_VAPI_PUBLIC_KEY   || "";
const ASSISTANT_ID = process.env.NEXT_PUBLIC_VAPI_ASSISTANT_ID || "";

export default function VoiceButton() {
  const [status, setStatus] = useState<"idle" | "connecting" | "active">("idle");
  const vapiRef = useRef<unknown>(null);

  async function startCall() {
    if (!PUBLIC_KEY || !ASSISTANT_ID) {
      alert("Set NEXT_PUBLIC_VAPI_PUBLIC_KEY and NEXT_PUBLIC_VAPI_ASSISTANT_ID in .env.local");
      return;
    }
    setStatus("connecting");
    const { default: Vapi } = await import("@vapi-ai/web");
    const vapi = new Vapi(PUBLIC_KEY);   // constructor takes the PUBLIC key
    vapiRef.current = vapi;

    vapi.on("call-start", () => setStatus("active"));
    vapi.on("call-end",   () => { setStatus("idle"); vapiRef.current = null; });
    vapi.on("error",      () => { setStatus("idle"); vapiRef.current = null; });

    await vapi.start(ASSISTANT_ID);      // start() takes the ASSISTANT ID
  }

  async function endCall() {
    if (vapiRef.current) {
      (vapiRef.current as { stop: () => void }).stop();
    }
    setStatus("idle");
  }

  if (status === "active") {
    return (
      <button
        onClick={endCall}
        className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-full text-sm font-medium transition-colors"
      >
        <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
        End Call
      </button>
    );
  }

  return (
    <button
      onClick={startCall}
      disabled={status === "connecting"}
      className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white rounded-full text-sm font-medium transition-colors"
    >
      {status === "connecting" ? (
        <>
          <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
          Connecting…
        </>
      ) : (
        <>
          <span>🎤</span> Talk to Vansh
        </>
      )}
    </button>
  );
}
