const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

/**
 * Stream a chat message. The server manages session history — the client
 * only needs to persist and forward session_id across requests.
 * The server emits a `{ session_id }` event before [DONE] so the client
 * can store it on first message.
 */
export async function streamChat(
  message: string,
  sessionId: string | null,
  onToken: (token: string) => void,
  onSessionId: (id: string) => void,
  onDone: () => void,
  onError: (err: string) => void
): Promise<void> {
  const body: Record<string, unknown> = { message, mode: "chat" };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    onError(`Server error: ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") { onDone(); return; }
      try {
        const parsed = JSON.parse(raw);
        if (parsed.token) onToken(parsed.token);
        if (parsed.session_id) onSessionId(parsed.session_id);
        if (parsed.error) onError(parsed.error);
      } catch {
        // ignore malformed chunk
      }
    }
  }
  onDone();
}
