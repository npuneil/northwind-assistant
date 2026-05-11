# Northwind Mobile — On-Device AI Seller Assistant 📱

A showcase application demonstrating on-device AI for telco retail sellers, running entirely on the NPU (Neural Processing Unit) via **Microsoft Foundry Local**. Listens to the in-store conversation between a seller and a customer, extracts a structured customer profile, and surfaces the right **device + plan + accessory bundle + trade-in + promo** in real time — maximizing **total value per customer** without sending a single byte of conversation audio to the cloud. Optimized for **Intel Core Ultra (AI Boost NPU)** with Phi-4 Mini; Snapdragon X (QNN) is also auto-detected. Works in airplane mode.

## On-Device AI Prototypes & Sample Code

### Overview

This repository contains prototypes, demos, and sample code that illustrate patterns for building on-device AI solutions. The content is provided for educational and demonstration purposes only to help developers explore ideas and implementation approaches.

This repository does not contain Microsoft products and is not a supported or production-ready offering.

### Prototype & Sample Code Disclosure

- All code and demos are experimental prototypes or samples.
- They may be incomplete, change without notice, or be removed at any time.
- The contents are provided "as-is," without warranties or guarantees of any kind.

### No Product, Performance, or Business Claims

- This repository makes no claims about performance, accuracy, productivity, efficiency, cost savings, reliability, or security.
- Any example outputs, screenshots, or logs are illustrative only and should not be interpreted as typical or expected results.

### AI Output Variability

- AI and machine-learning outputs may be non-deterministic, incomplete, or incorrect.
- Example outputs shown here are not guaranteed and may vary across runs, devices, or environments.

### Responsible AI Considerations

- These samples are intended to demonstrate technical patterns, not validated AI systems.
- Developers are responsible for evaluating fairness, reliability, privacy, accessibility, and safety before using similar approaches in real applications.
- Do not deploy AI solutions based on this code without appropriate testing, human oversight, and safeguards.

### Data & Fictitious Content

- Any names, data, or scenarios used in examples are fictitious and for illustration only.
- All carrier, device, and OEM names ("Northwind Mobile", "Aurora", "Lumen") are invented for this demo and do not represent real products.
- Do not use real personal, customer, or confidential data without proper authorization and protections.

### Third-Party Components

- The repository may reference third-party libraries or tools.
- Use of those components is subject to their respective licenses and terms.

### No Support

Microsoft does not provide support, SLAs, or warranties for the contents of this repository.

### Summary

By using this repository, you acknowledge that it contains illustrative prototypes and sample code only, not supported or production-ready software.

---

## Quick Start

```powershell
# First time:
winget install Microsoft.FoundryLocal
foundry model run phi-4-mini --device NPU
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Every time:
python northwind_assistant.py     # opens at http://localhost:5000
```

Or just double-click `run.bat`.

## Prerequisites

- **Windows 11 Copilot+ PC** with Intel Core Ultra (AI Boost NPU) or Snapdragon X NPU
- **Python 3.10+**
- **Foundry Local** installed (`winget install Microsoft.FoundryLocal`)

## NPU Optimization

This app is silicon-aware and probes Foundry Local at startup:

- **NPU-first model preference**:
  - Intel → `phi-4-mini-instruct-openvino-npu` → `-openvino-gpu` → generic-cpu fallback
  - Qualcomm → `phi-3.5-mini-instruct-qnn` → generic fallback
- **Auto-discovery** of the running Foundry Local port (parses `foundry service status`) and best available NPU model from `/v1/models`
- **Auto-load** the preferred model on NPU with a **2-hour TTL** so it stays hot across a demo session
- **OpenAI-compatible client** against `http://127.0.0.1:<port>/v1` — bypasses the SDK to avoid version churn
- **Max generation clamped to 480 tokens** to stay inside the OpenVINO NPU output cache window
- **CPU/GPU fallback** when no NPU model is available; **demo-mode heuristic fallback** when Foundry Local isn't running so the UI still works end-to-end

## Features

| Panel | Description |
|-------|-------------|
| **Live Conversation** | Web Speech API mic capture (or load one of three scripted demo conversations) with editable transcript pane |
| **Customer Profile** | One-shot Phi-4 Mini extraction of segment, budget band, lines needed, monthly data usage, current carrier/device, interests, pain points, intl-travel, insurance intent |
| **Device Recommendation** | Primary pick + 1 alternative with rationale, financing options, and highlight bullets |
| **Plan Recommendation** | Right-sized plan (Essentials / Plus / Premium / Family 4-line / Business Pro) with perks |
| **Accessory Bundle** | Case, charger, screen protector, earbuds, watch, insurance — matched to interests + segment |
| **Trade-In Estimate** | Catalog lookup against the customer's current device (iPhone / Galaxy / Pixel / Lumen) |
| **Promo Eligibility** | Trade-in flagship bonus, BOGO, switcher credit, Business 365 free, accessory bundle discount |
| **Cart Summary** | Itemized device + plan + accessories with monthly total, upfront, 24-month total, and **total-value uplift vs Essentials baseline** |
| **EN/ES Toggle** | On-device Spanish translation of the customer-facing cart summary |
| **Seller Coach** | Private right-panel chat with `[TOOL_CALL]` shim (`lookup_plan`, `compare_plans`, `lookup_promo`, `list_all`) for objection handling and policy lookups |
| **Audit Trail** | Live trace of every tool call with elapsed time + cumulative tokenomics surfaced in the footer (local-AI cost saved, CO₂ avoided) |
| **Offline Toggle** | Simulate airplane mode; everything still works because everything is local |

## Architecture

```
Browser (localhost:5000)
   │
   ├── Live Conversation Panel
   │     ├── Web Speech API → streaming transcript
   │     ├── /transcribe-turn  → Phi entity extraction (JSON)
   │     ├── /recommend        → device / plan / accessories / trade-in / promos
   │     └── /cart             → itemized totals + uplift
   │
   └── Seller Coach Panel
         └── /coach            → [TOOL_CALL] shim over catalog tools
   │
Flask backend (northwind_assistant.py, single file)
   │
   └── OpenAI-compatible HTTP API → Foundry Local service (random port)
            ├── Intel Core Ultra → phi-4-mini-instruct-openvino-npu  (OpenVINO EP)
            └── Snapdragon X     → phi-3.5-mini-instruct-qnn         (QNN EP)
```

Silicon is detected via WMI CPU name (authoritative on ARM64 where `platform.machine()` may report AMD64 under x64 emulation). The Foundry Local endpoint is discovered at runtime by parsing `foundry service status` — no hard-coded ports.

## Sample Data

The fictional **Northwind Mobile** catalog lives in `catalog/*.json`:

- **5 devices** across Aurora (flagship/flip) and Lumen (mid/budget) lines with full pricing, financing, specs, and target-customer hints
- **5 plans** — Essentials ($35), Plus ($55), Premium ($75), Family 4-line ($160 bundle), Business Pro ($65)
- **12 accessories** across case, charging, audio, wearable, connectivity, insurance categories
- **5 promos** — trade-in flagship bonus, BOGO line, switcher credit, Business 365 free, accessory bundle 20% off
- **12 trade-in models** with good/fair condition values (iPhone 11-15, Galaxy S22-24, Pixel 7-8, etc.)
- **3 scripted demo conversations** in `demo_data/sample_conversations/` for reliable mic-free playback (budget family, premium upgrader, business line-add)

All carrier, device, and OEM names are invented. Edit any JSON file and reload — no rebuild needed.

## Demo Experience

**The key demo moment:** Load the "**Premium upgrader**" sample conversation, click **Analyze ▸**, and watch Phi-4 Mini on the NPU pull a structured customer profile out of a 6-line conversation in ~10 seconds — international traveler, Verizon, Galaxy S22 ready for trade-in, photography hobbyist — then cascade into Aurora X Pro + Northwind Premium + Pro accessories + $230 trade-in + flagship promo. Click **ES** and the cart translates to Spanish on-device in ~4 seconds. Toggle **Go Offline** and run it again. Then ask the Seller Coach *"Customer says it's too expensive — what do I say?"* and watch it pull live numbers from the catalog tools to build a talk-track. None of the conversation ever touches the cloud.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | UI |
| `GET /health` | Silicon, model, and Foundry status |
| `GET /catalog/{devices,plans,accessories,promos,trade_in}` | Raw catalog |
| `GET /sample-conversation/{key}` | Scripted demo transcripts |
| `POST /transcribe-turn` | Extract customer profile from transcript |
| `POST /recommend` | Build recommendation from profile |
| `POST /cart` | Itemized cart + 24-month total + uplift |
| `POST /coach` | Seller chat (with tool-calling shim) |
| `POST /translate` | EN ↔ ES |
| `GET /audit` | Recent tool calls + cumulative tokenomics |

## License

MIT.
