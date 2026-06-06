# Vapi Voice Assistant Setup

## Steps to configure

1. Go to [Vapi Dashboard](https://dashboard.vapi.ai) → Assistants → Create Assistant
2. Paste the JSON from `vapi_assistant_config.json`
3. Replace `REPLACE_WITH_API_URL` with your deployed API URL (e.g. `https://persona-api.onrender.com`)
   - For local testing: use ngrok → `ngrok http 8000` → copy the HTTPS URL
4. In the assistant config, under Phone Numbers → assign the imported Twilio number
5. Copy the **Assistant ID** from the dashboard → paste into `client/.env.local` as `NEXT_PUBLIC_VAPI_ASSISTANT_ID`

## Local testing with ngrok

```bash
# Terminal 1 — start API
cd server && make dev

# Terminal 2 — expose to internet
ngrok http 8000

# Copy the ngrok HTTPS URL and update the tool server URLs in Vapi dashboard
# e.g. https://abc123.ngrok.io/retrieve
```

## Latency tuning notes

- Voice path uses TOP_K=3 (vs 8 for chat) — set in tuning.py
- No cross-encoder reranking on voice path (saves ~200-300ms)
- Redis cache eliminates embed+search on repeated queries
- Target: < 2s first response (Vapi STT → tool call → LLM → TTS)
- ElevenLabs voice ID `onwK4e9ZLuTAKqWW03F9` = Daniel (neutral, professional)
  Change to any 11labs voice ID you prefer
