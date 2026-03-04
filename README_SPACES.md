---
title: hydradeck-webui
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# hydradeck WebUI (Hugging Face Spaces - Docker)

Set these Secrets in Space settings if needed:

- `GROK_API_KEY`
- `GROK_BASE_URL` (optional, defaults to `https://api.example.com`)
- `GROK_MODEL` (optional, defaults to `grok-4`)

The app entrypoint is `custom_web.py` and listens on `$PORT`.

If build logs stop after cache export, trigger a rebuild by committing any change; this Space uses Docker + uvicorn startup.
