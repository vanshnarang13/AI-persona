"use client";
import dynamic from "next/dynamic";
import VoiceButton from "@/components/voice/VoiceButton";

const ChatWindow = dynamic(() => import("@/components/chat/ChatWindow"), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
      Loading…
    </div>
  ),
});

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl h-[85vh] flex flex-col bg-slate-900 rounded-2xl shadow-2xl border border-slate-800 overflow-hidden">
        {/* Header */}
        <div className="flex items-center px-5 py-4 border-b border-slate-800 bg-slate-900 gap-3">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            V
          </div>
          <div className="flex-1">
            <p className="text-sm font-semibold text-slate-100">Vansh Narang</p>
            <p className="text-xs text-slate-400">AI Engineer · IIT Roorkee · Available for interviews</p>
          </div>
          <VoiceButton />
        </div>

        {/* Chat */}
        <ChatWindow />
      </div>

      <p className="mt-3 text-xs text-slate-600">
        Answers grounded in Vansh&apos;s resume, projects, and experience.
      </p>
    </div>
  );
}
