"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import Vapi from "@vapi-ai/web";

type CallStatus = "idle" | "connecting" | "active" | "ending";

const PUBLIC_KEY   = process.env.NEXT_PUBLIC_VAPI_PUBLIC_KEY   ?? "";
const ASSISTANT_ID = process.env.NEXT_PUBLIC_VAPI_ASSISTANT_ID ?? "";

let vapiInstance: Vapi | null = null;
function getVapi(): Vapi {
  if (!vapiInstance) {
    // alwaysIncludeMicInPermissionPrompt ensures Daily.co registers the mic
    // in the WebRTC audio track even when permissions were already granted
    vapiInstance = new Vapi(PUBLIC_KEY, undefined, {
      alwaysIncludeMicInPermissionPrompt: true,
    });
  }
  return vapiInstance;
}

// ── Raw browser mic test (bypasses Vapi entirely) ─────────────────────────────
function useMicTest() {
  const [micLevel, setMicLevel]   = useState(0);
  const [testing,  setTesting]    = useState(false);
  const [micError, setMicError]   = useState<string | null>(null);
  const rafRef    = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef    = useRef<AudioContext | null>(null);

  const startTest = useCallback(async () => {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      ctxRef.current = ctx;
      const source   = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const buf = new Uint8Array(analyser.frequencyBinCount);

      setTesting(true);
      const tick = () => {
        analyser.getByteFrequencyData(buf);
        const avg = buf.reduce((a, b) => a + b, 0) / buf.length;
        setMicLevel(Math.min(100, Math.round(avg * 3)));
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      setMicError((e as Error)?.message ?? "Mic access failed");
    }
  }, []);

  const stopTest = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    ctxRef.current?.close();
    streamRef.current = null;
    ctxRef.current    = null;
    setTesting(false);
    setMicLevel(0);
  }, []);

  useEffect(() => () => { cancelAnimationFrame(rafRef.current); streamRef.current?.getTracks().forEach(t => t.stop()); }, []);

  return { micLevel, testing, micError, startTest, stopTest };
}

// ── Main component ────────────────────────────────────────────────────────────
export default function VoiceButton() {
  const [status,     setStatus]     = useState<CallStatus>("idle");
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [error,      setError]      = useState<string | null>(null);

  const { micLevel, testing, micError, startTest, stopTest } = useMicTest();

  useEffect(() => {
    if (!PUBLIC_KEY || !ASSISTANT_ID) return;
    const vapi = getVapi();

    const onStart      = () => { setStatus("active"); setError(null); };
    const onEnd        = () => { setStatus("idle"); setAiSpeaking(false); };
    const onError      = (e: unknown) => {
      console.error("[Vapi error]", e);
      setStatus("idle");
      const msg = (e as Error)?.message ?? String(e);
      setError(msg.slice(0, 100));
    };
    const onSpeechStart = () => setAiSpeaking(true);
    const onSpeechEnd   = () => setAiSpeaking(false);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const onMessage = (msg: any) => {
      // Only log meaningful events — skip high-frequency noise
      const noisy = ["volume-level", "model-output", "conversation-update", "voice-input"];
      if (!noisy.includes(msg?.type)) {
        console.log("[Vapi]", msg?.type, msg);
      }
    };

    vapi.on("call-start",   onStart);
    vapi.on("call-end",     onEnd);
    vapi.on("error",        onError);
    vapi.on("speech-start", onSpeechStart);
    vapi.on("speech-end",   onSpeechEnd);
    vapi.on("message",      onMessage);

    return () => {
      vapi.off("call-start",   onStart);
      vapi.off("call-end",     onEnd);
      vapi.off("error",        onError);
      vapi.off("speech-start", onSpeechStart);
      vapi.off("speech-end",   onSpeechEnd);
      vapi.off("message",      onMessage);
    };
  }, []);

  const toggle = useCallback(async () => {
    if (!PUBLIC_KEY || !ASSISTANT_ID) { setError("Voice not configured."); return; }
    const vapi = getVapi();
    if (status === "active") { setStatus("ending"); vapi.stop(); return; }
    if (status !== "idle") return;
    if (testing) stopTest(); // release mic test stream so Vapi can acquire the mic

    // Check permission via query API — never opens a stream so Vapi gets clean mic access
    try {
      const perm = await navigator.permissions.query({ name: "microphone" as PermissionName });
      if (perm.state === "denied") {
        setError("Mic blocked — allow microphone in browser site settings.");
        return;
      }
    } catch {
      // permissions.query not supported in all browsers — proceed anyway
    }

    setStatus("connecting");
    setError(null);
    try {
      await vapi.start(ASSISTANT_ID);
    } catch (e) {
      setError((e as Error)?.message ?? "Failed to start");
      setStatus("idle");
    }
  }, [status]);

  if (!PUBLIC_KEY || !ASSISTANT_ID) return null;

  const isActive = status === "active";
  const isBusy   = status === "connecting" || status === "ending";
  const callLabel: Record<CallStatus, string> = {
    idle: "Talk to Vansh", connecting: "Connecting…",
    active: aiSpeaking ? "Speaking…" : "Listening…", ending: "Ending…",
  };

  return (
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex items-center gap-2">

        {/* Mic test button — shows raw browser mic level, bypasses Vapi */}
        {!isActive && (
          <button
            onClick={testing ? stopTest : startTest}
            className={`px-2 py-1.5 rounded-xl text-xs font-medium transition-all ${
              testing ? "bg-amber-600 hover:bg-amber-700 text-white"
                      : "bg-slate-700 hover:bg-slate-600 text-slate-300"
            }`}
            title="Test your microphone"
          >
            {testing ? "Stop mic test" : "Test mic"}
          </button>
        )}

        {/* Main call button */}
        <button
          onClick={toggle}
          disabled={isBusy}
          className={[
            "flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium transition-all",
            isActive ? "bg-red-600 hover:bg-red-700 text-white"
                     : "bg-emerald-600 hover:bg-emerald-700 text-white",
            isBusy ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
          ].join(" ")}
        >
          {isActive ? (
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <rect x="4" y="4" width="12" height="12" rx="2" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 1a3 3 0 00-3 3v6a3 3 0 006 0V4a3 3 0 00-3-3z" />
              <path d="M5 9a1 1 0 00-2 0 7 7 0 0014 0 1 1 0 00-2 0 5 5 0 01-10 0z" />
              <path d="M9 17v2a1 1 0 002 0v-2a7.01 7.01 0 01-2 0z" />
            </svg>
          )}
          {isActive && aiSpeaking && (
            <span className="flex gap-0.5 items-end h-3">
              {[0,150,300].map(d => (
                <span key={d} className="w-0.5 bg-white rounded-full animate-bounce"
                  style={{ height: "100%", animationDelay: `${d}ms` }} />
              ))}
            </span>
          )}
          <span>{callLabel[status]}</span>
        </button>
      </div>

      {/* Mic test level bar */}
      {testing && (
        <div className="w-40 flex flex-col gap-0.5">
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-amber-400 rounded-full transition-all duration-75"
              style={{ width: `${micLevel}%` }} />
          </div>
          <p className="text-xs text-slate-400 text-right">
            {micLevel > 5 ? "✓ Mic working" : "Speak to test…"}
          </p>
        </div>
      )}

      {/* Status indicator during active call */}
      {isActive && (
        <p className="text-xs text-slate-500 text-right">
          {aiSpeaking ? "AI speaking…" : "Your turn — speak now"}
        </p>
      )}

      {(error || micError) && (
        <p className="text-xs text-red-400 max-w-[200px] text-right">{error ?? micError}</p>
      )}
    </div>
  );
}
