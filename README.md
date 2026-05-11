# Northwind Mobile — On-Device AI Seller Assistant

A Copilot+ PC demo that helps telco retail sellers maximize **total customer value** during in-store conversations — recommending the right plan, accessories, trade-ins, and promos based on the live conversation. **100% on-device** via Foundry Local on the NPU.

## Why this matters

- **Privacy:** the customer conversation never leaves the device — no cloud, no recording in someone else's data center.
- **Reliability:** retail Wi-Fi is unreliable. On-device means consistent latency, even in airplane mode.
- **Total value:** sellers often miss adjacent value (plan tier, accessories, trade-in, promos). This assistant surfaces them in real time.

## Architecture

```
Browser (localhost:5000)
   │
   ├── Live Conversation Panel
   │     ├── Web Speech API mic + scripted demo transcripts
   │     ├── Entity extraction (Phi-4 Mini on NPU, ~9s)
   │     ├── Recommendation cards (device / plan / accessories / trade-in / promos)
   │     └── Cart summary with 24-month total + total-value uplift
   │
   └── Seller Coach Panel
         └── Private chat (objection handling + [TOOL_CALL] shim for catalog lookups)
   │
Flask backend (northwind_assistant.py)
   │
   └── OpenAI-compatible HTTP API → Foundry Local service (random port)
            │
            ├── Intel Core Ultra → phi-4-mini-instruct-openvino-npu (OpenVINO EP)
            └── Qualcomm Snapdragon X → phi-3.5-mini on QNN
```

Silicon is auto-detected via WMI CPU name (authoritative on ARM64 where `platform.machine()` may report AMD64 under emulation). The Foundry Local endpoint is discovered at runtime by parsing `foundry service status`; the model id is resolved by querying `/v1/models` and preferring NPU-loaded OpenVINO/QNN variants.

## Features

- 🎙️ **Live conversation capture** via Web Speech API; plus three scripted demo conversations for reliable demo playback (budget family, premium upgrader, business line add).
- 🧠 **Customer profile extraction** — segment, budget band, lines needed, monthly data usage, current carrier/device, interests, pain points, intl-travel, insurance intent.
- 📱 **Device recommendation** with a sensible alternative.
- 📶 **Plan recommendation** with ARPU uplift vs Essentials baseline.
- 🎧 **Accessory bundle** matched to customer interests + segment.
- 🔄 **Trade-in estimate** by model lookup.
- 🎁 **Promo eligibility** — trade-in bonus, BOGO, switcher credit, business 365, bundle discount.
- 🛒 **One-click cart** — itemized monthly + upfront + 24-month total + total-value uplift.
- 🌎 **EN/ES toggle** — translates customer-facing cart summary on-device.
- 💬 **Seller Coach** — private chat with `[TOOL_CALL]` shim for objection handling and catalog lookups.
- 📊 **Audit trail + tokenomics** — every tool call logged; cumulative local-AI savings widget.
- ✈️ **Offline toggle** — simulate airplane mode; everything still works.

## Quick Start

### Prerequisites
- Windows 11 24H2 on a Copilot+ PC (Intel Core Ultra or Snapdragon X)
- Python 3.10+
- Foundry Local (`winget install Microsoft.FoundryLocal`)

### Install + Run
```powershell
.\setup.ps1     # one-time: installs Foundry Local + Python deps
.\run.bat       # starts Flask on http://127.0.0.1:5000
```

Or manual:
```powershell
pip install -r requirements.txt
python northwind_assistant.py
```

Then open **http://127.0.0.1:5000**.

## Demo Script (5 minutes)

1. Click **"Load demo conversation"** → choose **"Premium upgrader"**.
2. Click **Analyze ▸**. Watch the Customer Profile populate (Aurora X Pro intent, Verizon, intl travel, Galaxy S22 trade-in).
3. Recommendations cascade in: Aurora X Pro + Premium plan + Pro accessories + $350 trade-in + flagship promo.
4. Cart summary shows monthly + 24-month total + **total-value uplift vs baseline**.
5. Click **ES** → cart translates to Spanish on-device.
6. In Seller Coach (right panel), type: *"Customer says the Pro is too expensive — what do I say?"* — coach responds with specific talk-track using cart numbers.
7. Toggle **Go Offline** → re-run analyze. Still works.

## Project Layout

```
northwind-assistant/
├── northwind_assistant.py    # Flask app (HTML/CSS/JS inline)
├── catalog/                  # Fictional Northwind Mobile catalog
│   ├── devices.json
│   ├── plans.json
│   ├── accessories.json
│   ├── promos.json
│   └── trade_in.json
├── demo_data/sample_conversations/
│   ├── 01_budget_family.txt
│   ├── 02_premium_upgrader.txt
│   └── 03_business_line_add.txt
├── requirements.txt
├── setup.ps1
├── run.bat
└── README.md
```

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | UI |
| `GET /health` | Silicon + Foundry status |
| `GET /catalog/{devices,plans,accessories,promos,trade_in}` | Raw catalog |
| `GET /sample-conversation/{key}` | Scripted transcripts |
| `POST /transcribe-turn` | Extract customer profile from transcript |
| `POST /recommend` | Build recommendation from profile |
| `POST /cart` | Itemized cart + totals |
| `POST /coach` | Seller chat with tool-calling shim |
| `POST /translate` | EN ↔ ES |
| `GET /audit` | Recent tool calls + cumulative tokenomics |

## Notes

- All carrier, device, and OEM names are **fictional** to avoid trademark concerns.
- The app gracefully degrades to **Demo Mode** with heuristic-based fallbacks if Foundry Local isn't running, so the UI still demos end-to-end on any Windows machine.
- Tool-calling uses the `[TOOL_CALL]{...}[/TOOL_CALL]` marker shim (same pattern as surface-npu-demo) since some Phi variants lack native tool-calling.
- The Foundry Local OpenAI endpoint binds to a **random port** per service start; the app discovers it by parsing `foundry service status` so no hard-coding required.
- NPU context budget on Phi-4 Mini OpenVINO is **3696 input / 528 output tokens** — outputs are clamped to 480.
- Model TTL is set to **7200s** (2h) when loaded by the app so the NPU model stays hot through a demo session.

## Verified on Intel Core Ultra

```
silicon=intel  model=phi-4-mini-instruct-openvino-npu:3  endpoint=http://127.0.0.1:60326/v1
```

End-to-end smoke test (all 3 scenarios + coach + translate + audit) passes against the live NPU:

| Step | Latency |
|------|---------|
| Entity extraction | ~9-11s |
| Recommendation rationale | ~6-8s |
| Coach with tool calls | ~10-22s |
| Spanish translation | ~4s |
