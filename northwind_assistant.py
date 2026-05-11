"""
Northwind Mobile - On-Device AI Seller Assistant
=================================================
A Copilot+ PC demo showcasing on-device AI for telco retail sellers.
100% local processing via Foundry Local (Phi-4 Mini on Intel / Phi-3.5 Mini on Qualcomm NPU).

Architecture mirrors surface-npu-demo:
  - Single-file Flask app with inline HTML/CSS/JS
  - Foundry Local SDK auto-selects model based on detected silicon
  - OpenAI-compatible chat completions
  - Dedicated single-step endpoints (avoid two-step agent-loop hang)
  - Tool-calling shim via [TOOL_CALL] markers for catalog lookups
"""

import json
import os
import platform
import re
import subprocess
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).parent.resolve()
CATALOG_DIR = APP_DIR / "catalog"
DEMO_DIR = APP_DIR / "demo_data"
STATIC_DIR = APP_DIR / "static"

# ------------------------------------------------------------
# Silicon detection (Intel Core Ultra vs Qualcomm Snapdragon X)
# ------------------------------------------------------------
def detect_silicon():
    """Return ('intel'|'qualcomm'|'other', cpu_name). Uses WMI on Windows since
    platform.machine() may report AMD64 under x64 emulation on ARM64."""
    cpu_name = ""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in out.stdout.splitlines() if l.strip() and "Name" not in l]
            if lines:
                cpu_name = lines[0]
    except Exception:
        pass
    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"
    low = cpu_name.lower()
    if "snapdragon" in low or "qualcomm" in low or "oryon" in low:
        return ("qualcomm", cpu_name)
    if "intel" in low or "core(tm) ultra" in low or "core ultra" in low:
        return ("intel", cpu_name)
    return ("other", cpu_name)

SILICON, CPU_NAME = detect_silicon()

# ------------------------------------------------------------
# Foundry Local bootstrap
# ------------------------------------------------------------
# Pick model alias by silicon (matches surface-npu-demo conventions).
MODEL_ALIAS = "phi-3.5-mini" if SILICON == "qualcomm" else "phi-4-mini"

FOUNDRY_OK = False
FOUNDRY_ENDPOINT = None
FOUNDRY_MODEL = None
FOUNDRY_ERROR = None
_openai_client = None

# Model id preferences per silicon (substring match against /v1/models response).
# We pick the highest-priority *loaded* model that contains the substring.
MODEL_PREFERENCES = {
    "intel":    ["phi-4-mini-instruct-openvino-npu", "phi-4-mini-instruct-openvino-gpu", "phi-3.5-mini"],
    "qualcopm": ["phi-3.5-mini-instruct-qnn", "phi-3.5-mini"],
    "qualcomm": ["phi-3.5-mini-instruct-qnn", "phi-3.5-mini"],
    "other":    ["phi-4-mini-instruct", "phi-3.5-mini"],
}

def _discover_foundry_endpoint():
    """Parse 'foundry service status' for the bound URL. Returns base url like http://127.0.0.1:60326 or None."""
    try:
        out = subprocess.run(["foundry", "service", "status"], capture_output=True, text=True, timeout=10)
        m = re.search(r"https?://[0-9a-zA-Z\.\-:]+", (out.stdout or "") + (out.stderr or ""))
        if not m:
            return None
        url = m.group(0)
        # Strip path (e.g. /openai/status) -> base
        parsed = url.split("/openai")[0].rstrip("/")
        return parsed
    except Exception:
        return None

def _ensure_model_loaded(model_alias):
    """Best-effort: ask foundry CLI to load the alias on NPU if it isn't already.
    Also extends TTL so the model stays hot for the demo session."""
    try:
        listed = subprocess.run(["foundry", "service", "list"], capture_output=True, text=True, timeout=10)
        already_loaded = model_alias.lower() in (listed.stdout or "").lower()
        if not already_loaded:
            subprocess.run(
                ["foundry", "model", "load", model_alias, "--device", "NPU", "--ttl", "7200"],
                capture_output=True, text=True, timeout=180,
            )
        else:
            # Touch the TTL by running a no-op request later via OpenAI client.
            pass
        return True
    except Exception:
        return False

def _pick_model_id(base_url, prefs):
    """Hit /v1/models and pick the first id matching one of our preference substrings."""
    try:
        import urllib.request
        with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        ids = [m["id"] for m in data.get("data", [])]
        for pref in prefs:
            for mid in ids:
                if pref.lower() in mid.lower():
                    return mid, ids
        return (ids[0] if ids else None), ids
    except Exception:
        return None, []

def init_foundry():
    """Discover Foundry Local endpoint + ensure model loaded + create OpenAI client.
    Bypasses the SDK (which has churned across versions) and talks straight to the
    OpenAI-compatible HTTP API exposed by `foundry service`."""
    global FOUNDRY_OK, FOUNDRY_ENDPOINT, FOUNDRY_MODEL, FOUNDRY_ERROR, _openai_client
    try:
        from openai import OpenAI

        base = _discover_foundry_endpoint()
        if not base:
            raise RuntimeError("Could not discover Foundry Local endpoint. Is `foundry service` running? Try: foundry service start")
        # Ensure the preferred alias is loaded into the service.
        _ensure_model_loaded(MODEL_ALIAS)

        prefs = MODEL_PREFERENCES.get(SILICON, MODEL_PREFERENCES["other"])
        model_id, available = _pick_model_id(base, prefs)
        if not model_id:
            raise RuntimeError(f"No suitable model loaded. Available: {available}. Try: foundry model run {MODEL_ALIAS} --device NPU")

        FOUNDRY_ENDPOINT = base + "/v1"
        FOUNDRY_MODEL = model_id
        _openai_client = OpenAI(base_url=FOUNDRY_ENDPOINT, api_key="foundry", timeout=120.0)
        FOUNDRY_OK = True
        print(f"[Foundry] silicon={SILICON} cpu={CPU_NAME!r}")
        print(f"[Foundry] endpoint={FOUNDRY_ENDPOINT}")
        print(f"[Foundry] model={FOUNDRY_MODEL}")
    except Exception as e:
        FOUNDRY_ERROR = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"[Foundry] init failed: {FOUNDRY_ERROR}")
        print("[Foundry] App will run in DEMO MODE (heuristic fallbacks).")

# NPU model context budget is tight (Phi-4 Mini openvino-npu: 3696 in / 528 out).
# Clamp output token requests below the cap.
MAX_OUTPUT_TOKENS = 480

def chat(messages, max_tokens=400, temperature=0.3, timeout=120.0):
    """Single non-streaming chat completion with timeout. Returns text or raises."""
    if not FOUNDRY_OK:
        raise RuntimeError("Foundry Local not available")
    mt = min(int(max_tokens), MAX_OUTPUT_TOKENS)
    resp = _openai_client.with_options(timeout=timeout).chat.completions.create(
        model=FOUNDRY_MODEL,
        messages=messages,
        max_tokens=mt,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

# ------------------------------------------------------------
# Catalog loader
# ------------------------------------------------------------
def _load_json(name):
    with open(CATALOG_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)

CATALOG = {
    "devices": _load_json("devices.json")["devices"],
    "plans": _load_json("plans.json")["plans"],
    "accessories": _load_json("accessories.json")["accessories"],
    "promos": _load_json("promos.json")["promos"],
    "trade_in": _load_json("trade_in.json"),
}

def find_device(device_id):
    return next((d for d in CATALOG["devices"] if d["id"] == device_id), None)

def find_plan(plan_id):
    return next((p for p in CATALOG["plans"] if p["id"] == plan_id), None)

def find_accessory(acc_id):
    return next((a for a in CATALOG["accessories"] if a["id"] == acc_id), None)

def estimate_trade_in(model_text, condition="good"):
    if not model_text:
        return None
    low = model_text.lower()
    for entry in CATALOG["trade_in"]["trade_in_values"]:
        if entry["model_pattern"] in low:
            return entry["condition_good" if condition == "good" else "condition_fair"]
    return None

# ------------------------------------------------------------
# Audit trail (in-memory, demo-scope)
# ------------------------------------------------------------
AUDIT_LOG = []
TOKENOMICS = {"local_tasks": 0, "approx_tokens": 0}

def audit(tool, args, success, elapsed_ms, extra=None):
    AUDIT_LOG.append({
        "ts": time.time(),
        "tool": tool,
        "args": args,
        "success": success,
        "elapsed_ms": elapsed_ms,
        "extra": extra or {},
    })
    if len(AUDIT_LOG) > 200:
        del AUDIT_LOG[:50]

def bump_tokenomics(tokens):
    TOKENOMICS["local_tasks"] += 1
    TOKENOMICS["approx_tokens"] += int(tokens or 0)

# ------------------------------------------------------------
# JSON extraction helper
# ------------------------------------------------------------
def extract_json(text):
    """Try to extract the first JSON object/array from model output."""
    if not text:
        return None
    # Code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Bare object
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

# ------------------------------------------------------------
# Flask app
# ------------------------------------------------------------
app = Flask(__name__, static_folder=None)

# Inline HTML/CSS/JS lives in templates/index.html string at bottom of file.

@app.route("/")
def index():
    return INDEX_HTML

@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "silicon": SILICON,
        "cpu": CPU_NAME,
        "foundry_ok": FOUNDRY_OK,
        "model": FOUNDRY_MODEL,
        "foundry_error": FOUNDRY_ERROR,
    })

@app.route("/catalog/<which>")
def catalog_route(which):
    if which not in CATALOG:
        return jsonify({"error": "not_found"}), 404
    return jsonify(CATALOG[which])

@app.route("/audit")
def audit_route():
    return jsonify({"log": AUDIT_LOG[-50:], "tokenomics": TOKENOMICS})

# ------------------------------------------------------------
# /sample-conversation - list/return scripted transcripts
# ------------------------------------------------------------
@app.route("/sample-conversation/<key>")
def sample_conversation(key):
    fname = {
        "budget_family": "01_budget_family.txt",
        "premium_upgrader": "02_premium_upgrader.txt",
        "business_line_add": "03_business_line_add.txt",
    }.get(key)
    if not fname:
        return jsonify({"error": "not_found"}), 404
    path = DEMO_DIR / "sample_conversations" / fname
    if not path.exists():
        return jsonify({"error": "missing_file"}), 404
    return jsonify({"transcript": path.read_text(encoding="utf-8")})

# ------------------------------------------------------------
# /transcribe-turn - extract entities from transcript
# ------------------------------------------------------------
ENTITY_SYSTEM_PROMPT = """You are an information extractor for a telco retail conversation.
Read the customer/seller transcript and return a STRICT JSON object with these fields:
{
  "intended_device": string or null  (the device or device type the customer mentioned wanting),
  "budget_band": "budget" | "mid" | "premium" | null,
  "monthly_usage_gb": integer or null  (per line if mentioned),
  "lines_needed": integer or null,
  "segment": "individual" | "family" | "business" | null,
  "current_carrier": string or null,
  "current_device": string or null  (their current phone, for trade-in),
  "pain_points": [string]  (short phrases, e.g. "battery dies by lunch"),
  "interests": [string]  (e.g. "photography", "gaming", "travel", "fitness"),
  "intl_travel": true | false | null,
  "wants_insurance": true | false | null
}
Return ONLY the JSON object. No prose. Use null when not stated."""

@app.route("/transcribe-turn", methods=["POST"])
def transcribe_turn():
    data = request.get_json(force=True, silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "empty transcript"}), 400
    t0 = time.time()
    try:
        text = chat(
            [
                {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nReturn the JSON object."},
            ],
            max_tokens=400, temperature=0.1,
        )
        entities = extract_json(text) or {}
    except Exception as e:
        traceback.print_exc()
        entities = _fallback_entities(transcript)
        audit("transcribe-turn", {"len": len(transcript)}, False, int((time.time()-t0)*1000), {"err": str(e)})
        bump_tokenomics(len(transcript)//4)
        return jsonify({"entities": entities, "source": "fallback"})
    audit("transcribe-turn", {"len": len(transcript)}, True, int((time.time()-t0)*1000))
    bump_tokenomics(len(transcript)//4 + 100)
    return jsonify({"entities": entities, "source": "model"})

def _fallback_entities(transcript):
    """Heuristic fallback when model is unavailable (so demo still works)."""
    t = transcript.lower()
    ent = {
        "intended_device": None, "budget_band": None, "monthly_usage_gb": None,
        "lines_needed": None, "segment": None, "current_carrier": None,
        "current_device": None, "pain_points": [], "interests": [],
        "intl_travel": None, "wants_insurance": None,
    }
    def has_word(w): return re.search(r"\b" + re.escape(w) + r"\b", t) is not None
    if "aurora x pro" in t: ent["intended_device"] = "Aurora X Pro"
    elif "aurora x" in t: ent["intended_device"] = "Aurora X"
    elif "lumen" in t: ent["intended_device"] = "Lumen 7"
    if any(has_word(w) for w in ("family", "daughter", "son", "husband", "wife", "kids", "household")):
        ent["segment"] = "family"
    if any(has_word(w) for w in ("business", "company", "technician", "technicians", "employees", "field team", "hvac")):
        ent["segment"] = "business"
    if has_word("single") and has_word("line"):
        ent["segment"] = "individual"
    if "international" in t or "travel" in t or "tokyo" in t or "europe" in t:
        ent["intl_travel"] = True
    if "drop" in t or "crack" in t: ent["wants_insurance"] = True
    for carrier in ["verizon", "at&t", "att", "t-mobile", "tmobile", "northwind"]:
        if carrier in t:
            ent["current_carrier"] = carrier.replace("att", "at&t").title()
            break
    for dev in ["iphone 15", "iphone 14", "iphone 13", "iphone 12", "iphone 11",
                "galaxy s24", "galaxy s23", "galaxy s22", "pixel 8", "pixel 7", "lumen 6"]:
        if dev in t:
            ent["current_device"] = dev.title()
            break
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*gig", t)
    if m: ent["monthly_usage_gb"] = (int(m.group(1)) + int(m.group(2))) // 2
    else:
        m = re.search(r"(\d+)\s*(?:gig|gb)", t)
        if m: ent["monthly_usage_gb"] = int(m.group(1))
    m = re.search(r"(\d+)\s*lines?", t)
    if m: ent["lines_needed"] = int(m.group(1))
    if "budget" in t or "reasonable" in t or "affordable" in t: ent["budget_band"] = "budget"
    elif "premium" in t or "pro" in t or "flagship" in t: ent["budget_band"] = "premium"
    if "photo" in t or "camera" in t: ent["interests"].append("photography")
    if "game" in t or "gaming" in t: ent["interests"].append("gaming")
    if "gym" in t or "fitness" in t or "run" in t: ent["interests"].append("fitness")
    if "netflix" in t or "stream" in t: ent["interests"].append("streaming")
    if "travel" in t: ent["interests"].append("travel")
    return ent

# ------------------------------------------------------------
# /recommend - build recommendation from entities
# ------------------------------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True, silent=True) or {}
    entities = data.get("entities") or {}
    t0 = time.time()
    rec = _build_recommendation(entities)
    # Optionally enrich rationale with model
    if FOUNDRY_OK:
        try:
            rationale_text = chat(
                [
                    {"role": "system", "content":
                        "You are a Northwind Mobile seller's AI assistant. Given a customer profile and a proposed "
                        "recommendation, write a SHORT (under 60 words) friendly rationale the seller can read aloud. "
                        "Mention specifically why each piece fits the customer. No bullet lists, just two sentences."},
                    {"role": "user", "content": json.dumps({"profile": entities, "recommendation": rec})},
                ],
                max_tokens=160, temperature=0.4,
            )
            rec["rationale"] = rationale_text.strip()
        except Exception:
            pass
    audit("recommend", {"segment": entities.get("segment")}, True, int((time.time()-t0)*1000))
    bump_tokenomics(300)
    return jsonify(rec)

def _build_recommendation(ent):
    segment = ent.get("segment") or "individual"
    budget = ent.get("budget_band")
    interests = set(ent.get("interests") or [])
    usage = ent.get("monthly_usage_gb")
    intl = ent.get("intl_travel")
    wants_ins = ent.get("wants_insurance")

    # Device pick
    if budget == "premium" or "photography" in interests or (ent.get("intended_device") or "").lower().startswith("aurora x pro"):
        device = find_device("aurora-x-pro")
        alt = find_device("aurora-x")
    elif segment == "business":
        device = find_device("lumen-7")
        alt = find_device("aurora-x")
    elif budget == "budget" or segment == "family":
        device = find_device("lumen-7")
        alt = find_device("lumen-7-lite")
    else:
        device = find_device("aurora-x")
        alt = find_device("lumen-7")

    # Plan pick
    if segment == "business":
        plan = find_plan("business-pro")
    elif segment == "family" and (ent.get("lines_needed") or 0) >= 3 or segment == "family":
        plan = find_plan("family-4")
    elif intl or (usage and usage > 30) or budget == "premium":
        plan = find_plan("premium")
    elif usage and usage <= 10 and budget == "budget":
        plan = find_plan("essentials")
    else:
        plan = find_plan("plus")

    # Accessories
    accessories = []
    if wants_ins or (device and device["tier"] == "flagship"):
        accessories.append(find_accessory("insurance-premium"))
    else:
        accessories.append(find_accessory("insurance-basic"))
    accessories.append(find_accessory("case-rugged" if "fitness" in interests or segment in ("family", "business") else "case-slim"))
    accessories.append(find_accessory("screen-protector"))
    accessories.append(find_accessory("charger-fast"))
    if "fitness" in interests or "gaming" in interests or "streaming" in interests or segment == "individual":
        accessories.append(find_accessory("earbuds-pro" if budget == "premium" else "earbuds-basic"))
    if "fitness" in interests:
        accessories.append(find_accessory("watch-pro" if budget == "premium" else "watch-active"))

    # Trade-in
    trade_in = None
    current = ent.get("current_device")
    if current:
        val = estimate_trade_in(current, "good")
        if val:
            trade_in = {"device": current, "estimated_value": val, "condition_assumed": "good"}

    # Promos
    eligible_promos = []
    if trade_in and device and device["id"] in ("aurora-x-pro", "aurora-x", "aurora-flip"):
        eligible_promos.append(next(p for p in CATALOG["promos"] if p["id"] == "trade-in-bonus-500"))
    if segment == "family" and (ent.get("lines_needed") or 0) >= 3 and ent.get("current_carrier") and "northwind" not in (ent.get("current_carrier") or "").lower():
        eligible_promos.append(next(p for p in CATALOG["promos"] if p["id"] == "family-switcher-300"))
    if segment == "business":
        eligible_promos.append(next(p for p in CATALOG["promos"] if p["id"] == "business-365-free"))
    if len(accessories) >= 3:
        eligible_promos.append(next(p for p in CATALOG["promos"] if p["id"] == "accessory-bundle-20"))

    return {
        "device": device, "device_alternative": alt,
        "plan": plan,
        "accessories": [a for a in accessories if a],
        "trade_in": trade_in,
        "promos": eligible_promos,
        "rationale": _default_rationale(ent, device, plan),
    }

def _default_rationale(ent, device, plan):
    bits = []
    if device: bits.append(f"{device['name']} fits because " + ", ".join(device.get("best_for", [])[:2]) + ".")
    if plan:   bits.append(f"{plan['name']} matches their usage and segment.")
    return " ".join(bits)

# ------------------------------------------------------------
# /cart - assemble itemized cart + totals
# ------------------------------------------------------------
@app.route("/cart", methods=["POST"])
def cart():
    data = request.get_json(force=True, silent=True) or {}
    rec = data.get("recommendation") or {}
    items = []
    monthly = 0.0
    upfront = 0.0
    device = rec.get("device") or {}
    plan = rec.get("plan") or {}
    accs = rec.get("accessories") or []
    trade_in = rec.get("trade_in")
    promos = rec.get("promos") or []

    if device:
        items.append({"label": f"{device['name']} (24-mo financing)",
                      "monthly": device.get("price_24mo", 0), "upfront": 0})
        monthly += device.get("price_24mo", 0) or 0
    if plan:
        plan_monthly = plan.get("total_price") or plan.get("price_per_line") or 0
        lines_label = f" ({plan.get('lines')} lines)" if plan.get("lines") else ""
        items.append({"label": f"{plan['name']}{lines_label}",
                      "monthly": plan_monthly, "upfront": 0})
        monthly += plan_monthly

    accessory_subtotal = 0
    for a in accs:
        if a.get("price_monthly"):
            items.append({"label": a["name"], "monthly": a["price_monthly"], "upfront": 0})
            monthly += a["price_monthly"]
        elif a.get("monthly_24mo"):
            items.append({"label": f"{a['name']} (24-mo)", "monthly": a["monthly_24mo"], "upfront": 0})
            monthly += a["monthly_24mo"]
        else:
            items.append({"label": a["name"], "monthly": 0, "upfront": a.get("price", 0)})
            upfront += a.get("price", 0) or 0
            accessory_subtotal += a.get("price", 0) or 0

    # Promos
    promo_lines = []
    if any(p.get("id") == "accessory-bundle-20" for p in promos) and accessory_subtotal:
        disc = round(accessory_subtotal * 0.20, 2)
        upfront -= disc
        promo_lines.append({"label": "Accessory bundle 20% off", "amount": -disc})
    if any(p.get("id") == "trade-in-bonus-500" for p in promos):
        promo_lines.append({"label": "Trade-in flagship bonus eligible", "amount": 0, "note": "Applied at point of sale"})
    if trade_in and trade_in.get("estimated_value"):
        promo_lines.append({"label": f"Trade-in credit ({trade_in['device']})",
                            "amount": -trade_in["estimated_value"]})
        upfront -= trade_in["estimated_value"]

    baseline_plan = 35  # Essentials per-line baseline
    plan_uplift_monthly = max(0, (plan.get("price_per_line") or 0) - baseline_plan) if plan else 0
    total_value_uplift_24mo = round(plan_uplift_monthly * 24 + accessory_subtotal, 2)

    return jsonify({
        "items": items,
        "promo_lines": promo_lines,
        "monthly_total": round(monthly, 2),
        "upfront_total": round(upfront, 2),
        "total_24mo": round(monthly * 24 + upfront, 2),
        "total_value_uplift_24mo": total_value_uplift_24mo,
    })

# ------------------------------------------------------------
# /coach - seller chat sidebar with tool-calling shim
# ------------------------------------------------------------
COACH_SYSTEM_PROMPT = """You are the seller's private AI coach at Northwind Mobile. Help the seller handle
customer objections, compare plans/devices, and check promo eligibility. You have access to TOOLS.

When you need data, emit a tool call on its own line:
[TOOL_CALL]{"name":"<tool_name>","arguments":{...}}[/TOOL_CALL]

Available tools:
- lookup_plan(plan_id)            -> returns plan details
- lookup_device(device_id)        -> returns device details
- compare_plans(plan_ids)         -> returns list of plan details
- lookup_promo(promo_id)          -> returns promo details
- list_all(kind)                  -> kind in {"plans","devices","accessories","promos"}

After receiving tool results, give a short, confident answer (under 80 words) the seller can paraphrase to the customer.
Be specific: cite numbers, plan names, and reasons. Never invent products or prices."""

def run_tool(name, args):
    args = args or {}
    if name == "lookup_plan":
        return find_plan(args.get("plan_id"))
    if name == "lookup_device":
        return find_device(args.get("device_id"))
    if name == "lookup_accessory":
        return find_accessory(args.get("accessory_id"))
    if name == "compare_plans":
        ids = args.get("plan_ids") or []
        return [find_plan(i) for i in ids if find_plan(i)]
    if name == "lookup_promo":
        return next((p for p in CATALOG["promos"] if p["id"] == args.get("promo_id")), None)
    if name == "list_all":
        kind = args.get("kind")
        return CATALOG.get(kind, [])
    return {"error": "unknown_tool", "name": name}

TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", re.DOTALL)

@app.route("/coach", methods=["POST"])
def coach():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    t0 = time.time()
    messages = [{"role": "system", "content": COACH_SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": user_msg})

    if not FOUNDRY_OK:
        reply = _coach_fallback(user_msg)
        audit("coach", {"q": user_msg[:80]}, True, int((time.time()-t0)*1000), {"source": "fallback"})
        bump_tokenomics(150)
        return jsonify({"reply": reply, "tool_calls": []})

    try:
        raw = chat(messages, max_tokens=500, temperature=0.3)
        tool_calls_log = []
        # Up to 2 tool iterations
        for _ in range(2):
            calls = TOOL_CALL_RE.findall(raw)
            if not calls:
                break
            tool_outputs = []
            for c in calls:
                try:
                    call = json.loads(c.strip())
                except Exception:
                    continue
                result = run_tool(call.get("name"), call.get("arguments"))
                tool_calls_log.append({"name": call.get("name"), "args": call.get("arguments"), "result_preview": str(result)[:200]})
                tool_outputs.append(f"TOOL_RESULT[{call.get('name')}]: {json.dumps(result)[:1200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "\n".join(tool_outputs) + "\n\nNow give the final answer for the seller."})
            raw = chat(messages, max_tokens=400, temperature=0.3)
        # Strip any leftover tool-call markers
        reply = TOOL_CALL_RE.sub("", raw).strip()
        audit("coach", {"q": user_msg[:80]}, True, int((time.time()-t0)*1000), {"tools": [t["name"] for t in tool_calls_log]})
        bump_tokenomics(400)
        return jsonify({"reply": reply, "tool_calls": tool_calls_log})
    except Exception as e:
        traceback.print_exc()
        reply = _coach_fallback(user_msg)
        audit("coach", {"q": user_msg[:80]}, False, int((time.time()-t0)*1000), {"err": str(e)})
        return jsonify({"reply": reply, "tool_calls": [], "error": str(e)})

def _coach_fallback(q):
    ql = q.lower()
    if "expensive" in ql or "too much" in ql or "cost" in ql:
        return ("Lead with total value, not sticker price. Walk through the 24-month breakdown: trade-in credit, "
                "switcher promo, and bundled accessory discount. Then show monthly cost — usually it's lower than "
                "their current bill on a competing carrier once you factor in the included perks.")
    if "premium" in ql or "unlimited" in ql:
        return ("Premium ($75/line) is right for travelers and heavy streamers — unlimited data, 4K video, "
                "40GB hotspot, and international roaming in 215+ countries. Plus ($55/line) is for mainstream "
                "users who don't travel and stream in HD.")
    return "Tell me the customer's objection and I'll give you specific talking points and numbers."

# ------------------------------------------------------------
# /translate - EN/ES translation
# ------------------------------------------------------------
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    target = (data.get("target") or "es").lower()
    if not text:
        return jsonify({"error": "empty"}), 400
    t0 = time.time()
    if not FOUNDRY_OK:
        audit("translate", {"target": target}, False, 0, {"source": "fallback"})
        return jsonify({"translated": "[Translation unavailable - Foundry Local not running]"})
    target_name = "Spanish" if target == "es" else "English"
    try:
        out = chat(
            [
                {"role": "system", "content": f"Translate the user's text to {target_name}. Return only the translation."},
                {"role": "user", "content": text},
            ],
            max_tokens=600, temperature=0.2,
        )
        audit("translate", {"target": target, "len": len(text)}, True, int((time.time()-t0)*1000))
        bump_tokenomics(len(text)//4 + 200)
        return jsonify({"translated": out.strip()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"translated": f"[Translation error: {e}]"})

# ------------------------------------------------------------
# Inline HTML / CSS / JS
# ------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Northwind Mobile — Seller Assistant</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='30' fill='%230a3d7a'/%3E%3Ctext x='32' y='40' font-family='Arial' font-size='28' font-weight='bold' text-anchor='middle' fill='white'%3EN%3C/text%3E%3C/svg%3E" />
<style>
* { box-sizing: border-box; }
:root {
  --nw-blue:#0a3d7a; --nw-blue-dark:#072a55; --nw-accent:#00b3d6;
  --nw-bg:#f4f6fb; --nw-panel:#ffffff; --nw-text:#1a2231; --nw-muted:#6a7385;
  --nw-border:#e3e7ef; --nw-good:#1e9d5a; --nw-warn:#d68900; --nw-bad:#c43c3c;
}
html,body { margin:0; padding:0; height:100%; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--nw-bg); color:var(--nw-text); font-size:14px; }
header {
  display:flex; align-items:center; gap:14px; padding:10px 18px;
  background:linear-gradient(90deg,var(--nw-blue),var(--nw-blue-dark));
  color:#fff; border-bottom:3px solid var(--nw-accent);
}
header .logo {
  width:36px;height:36px;border-radius:50%;background:#fff;color:var(--nw-blue);
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;
}
header h1 { margin:0; font-size:17px; font-weight:600; letter-spacing:.2px; }
header .tag { font-size:11px; opacity:.85; margin-top:2px; }
header .spacer { flex:1; }
.status-pill {
  display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px;
  background:rgba(255,255,255,.12); font-size:12px;
}
.status-pill .dot { width:8px;height:8px;border-radius:50%;background:#7dd3a4; }
.status-pill.off .dot { background:#ffb060; }
.poc-banner {
  background:#fff7e0; color:#7a5b00; font-size:12px; text-align:center; padding:6px 12px;
  border-bottom:1px solid #f0e3b8;
}
.layout { display:grid; grid-template-columns:1.4fr 1fr; gap:14px; padding:14px; height:calc(100vh - 95px); }
@media (max-width: 1100px) { .layout { grid-template-columns:1fr; height:auto; } }
.panel {
  background:var(--nw-panel); border:1px solid var(--nw-border); border-radius:12px;
  box-shadow:0 1px 2px rgba(0,0,0,.03); display:flex; flex-direction:column; overflow:hidden;
}
.panel h2 { margin:0; padding:12px 16px; font-size:14px; background:#f8fafd; border-bottom:1px solid var(--nw-border); color:var(--nw-blue); }
.panel-body { padding:14px 16px; overflow-y:auto; flex:1; }
button {
  font:inherit; padding:7px 14px; border-radius:8px; border:1px solid var(--nw-blue);
  background:var(--nw-blue); color:#fff; cursor:pointer; transition:.15s;
}
button:hover { background:var(--nw-blue-dark); }
button.ghost { background:#fff; color:var(--nw-blue); }
button.ghost:hover { background:#eef3fb; }
button.accent { background:var(--nw-accent); border-color:var(--nw-accent); color:#003646; }
button:disabled { opacity:.5; cursor:not-allowed; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.transcript {
  background:#f8fafd; border:1px solid var(--nw-border); border-radius:8px;
  padding:10px 12px; min-height:140px; max-height:260px; overflow-y:auto;
  font-size:13px; line-height:1.5; white-space:pre-wrap;
}
.transcript .seller { color:var(--nw-blue); font-weight:600; }
.transcript .customer { color:#444; }
.muted { color:var(--nw-muted); font-size:12px; }
.card {
  border:1px solid var(--nw-border); border-radius:10px; padding:12px; margin-bottom:10px;
  background:#fff; animation:fade .25s ease-out;
}
@keyframes fade { from{opacity:0;transform:translateY(4px);} to{opacity:1;transform:none;} }
.card h3 { margin:0 0 6px 0; font-size:13px; color:var(--nw-blue); text-transform:uppercase; letter-spacing:.5px; }
.card .title { font-size:15px; font-weight:600; }
.card .price { color:var(--nw-blue-dark); font-weight:600; }
.card ul { margin:6px 0 0 0; padding-left:18px; font-size:13px; }
.tag-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.tag {
  background:#eef3fb; color:var(--nw-blue); padding:2px 8px; border-radius:999px;
  font-size:11px; font-weight:500;
}
.tag.warn { background:#fff3df; color:#7a5b00; }
.tag.good { background:#e3f6ec; color:var(--nw-good); }
.chat-log { display:flex; flex-direction:column; gap:8px; }
.chat-msg { padding:8px 12px; border-radius:10px; max-width:90%; font-size:13px; line-height:1.45; }
.chat-msg.user { background:var(--nw-blue); color:#fff; align-self:flex-end; }
.chat-msg.bot  { background:#eef3fb; color:var(--nw-text); align-self:flex-start; white-space:pre-wrap; }
.chat-input { display:flex; gap:6px; padding:10px; border-top:1px solid var(--nw-border); background:#f8fafd; }
.chat-input input { flex:1; padding:8px 12px; border:1px solid var(--nw-border); border-radius:8px; font:inherit; }
.cart-totals { font-size:13px; border-top:1px dashed var(--nw-border); margin-top:10px; padding-top:10px; }
.cart-totals .big { font-size:18px; font-weight:700; color:var(--nw-blue-dark); }
.uplift { background:#e3f6ec; color:var(--nw-good); padding:6px 10px; border-radius:8px; font-size:13px; margin-top:8px; }
.profile-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:6px; font-size:12px; }
.profile-grid div { background:#f8fafd; padding:6px 8px; border-radius:6px; border:1px solid var(--nw-border); }
.profile-grid b { color:var(--nw-blue); display:block; font-size:10px; text-transform:uppercase; letter-spacing:.4px; }
footer.savings {
  font-size:11px; color:var(--nw-muted); padding:8px 14px; text-align:center;
  border-top:1px solid var(--nw-border); background:#fff;
}
.section-actions { display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }
select, input[type=text] { font:inherit; padding:6px 10px; border:1px solid var(--nw-border); border-radius:6px; }
.lang-toggle { display:inline-flex; border:1px solid var(--nw-border); border-radius:999px; overflow:hidden; }
.lang-toggle button { border:none; border-radius:0; padding:4px 14px; background:#fff; color:var(--nw-blue); font-size:12px; }
.lang-toggle button.active { background:var(--nw-blue); color:#fff; }
</style>
</head>
<body>
<header>
  <div class="logo">N</div>
  <div>
    <h1>Northwind Mobile — Seller Assistant</h1>
    <div class="tag">On-Device AI Copilot • <span id="silicon-tag">detecting…</span></div>
  </div>
  <div class="spacer"></div>
  <div class="status-pill" id="status-pill"><span class="dot"></span><span id="status-text">Connecting…</span></div>
  <button class="ghost" id="offline-toggle" title="Toggle simulated offline mode">Go Offline</button>
</header>
<div class="poc-banner">📌 POC Demo — 100% on-device inference. Customer conversation never leaves this Copilot+ PC.</div>

<div class="layout">
  <!-- LEFT: Live Conversation -->
  <div class="panel">
    <h2>🎙️ Live Conversation</h2>
    <div class="panel-body">
      <div class="section-actions">
        <button id="btn-mic" class="accent">🎤 Start Listening</button>
        <select id="sample-select">
          <option value="">— Load demo conversation —</option>
          <option value="budget_family">Budget family (mom + daughter)</option>
          <option value="premium_upgrader">Premium upgrader (traveler/photographer)</option>
          <option value="business_line_add">Business line add (HVAC owner)</option>
        </select>
        <button id="btn-analyze" class="ghost">Analyze ▸</button>
        <button id="btn-clear" class="ghost">Clear</button>
      </div>
      <div class="transcript" id="transcript-pane" contenteditable="true" spellcheck="false"></div>

      <h3 style="margin:14px 0 6px 0; font-size:13px; color:var(--nw-blue);">CUSTOMER PROFILE</h3>
      <div class="profile-grid" id="profile-grid"><div class="muted">Run analyze to extract profile…</div></div>

      <h3 style="margin:14px 0 6px 0; font-size:13px; color:var(--nw-blue);">RECOMMENDATIONS</h3>
      <div id="recs"><div class="muted">Recommendations will appear here.</div></div>

      <h3 style="margin:14px 0 6px 0; font-size:13px; color:var(--nw-blue);">CART SUMMARY
        <span style="float:right;" class="lang-toggle">
          <button data-lang="en" class="active">EN</button>
          <button data-lang="es">ES</button>
        </span>
      </h3>
      <div id="cart"><div class="muted">Click "Build Cart" after recommendations appear.</div></div>
    </div>
    <footer class="savings" id="savings-footer">Local AI tasks: 0 • Estimated cloud cost saved: $0.00 • CO₂ avoided: 0g</footer>
  </div>

  <!-- RIGHT: Seller Coach -->
  <div class="panel">
    <h2>🧠 Seller Coach (private)</h2>
    <div class="panel-body" id="chat-body">
      <div class="muted" style="margin-bottom:8px;">
        Ask anything: "Why is Plus better than Essentials for them?", "Customer says it's too expensive", "Compare Aurora X Pro vs Galaxy S24 trade-in math".
      </div>
      <div class="chat-log" id="chat-log"></div>
    </div>
    <div class="chat-input">
      <input id="chat-text" type="text" placeholder="Ask the coach…" />
      <button id="chat-send">Send</button>
    </div>
  </div>
</div>

<script>
// ────────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────────
let LAST_ENTITIES = null;
let LAST_REC = null;
let LAST_CART = null;
let CART_LANG = "en";
let OFFLINE = false;
const CHAT_HIST = [];

// ────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  if (OFFLINE) throw new Error("Simulated offline mode is on.");
  const opts = body ? { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error("HTTP "+r.status);
  return r.json();
}
function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
function esc(s) { return (s ?? "").toString().replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }

// ────────────────────────────────────────────────────────────────────
// Init: silicon + foundry status
// ────────────────────────────────────────────────────────────────────
(async function init() {
  try {
    const h = await fetch("/health").then(r => r.json());
    $("silicon-tag").textContent = `${h.silicon || "?"} • ${h.model || "model unavailable"}`;
    if (h.foundry_ok) {
      $("status-pill").classList.remove("off");
      $("status-text").textContent = "NPU ready";
    } else {
      $("status-pill").classList.add("off");
      $("status-text").textContent = "Demo mode (Foundry Local not running)";
    }
  } catch (e) {
    $("status-text").textContent = "Offline";
  }
  refreshSavings();
})();

// ────────────────────────────────────────────────────────────────────
// Mic (Web Speech API) + sample transcripts
// ────────────────────────────────────────────────────────────────────
let recog = null;
const btnMic = $("btn-mic");
function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { btnMic.disabled = true; btnMic.textContent = "🎤 Mic n/a"; return; }
  recog = new SR();
  recog.continuous = true; recog.interimResults = true; recog.lang = "en-US";
  recog.onresult = (ev) => {
    let finalText = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      if (ev.results[i].isFinal) finalText += ev.results[i][0].transcript + " ";
    }
    if (finalText) {
      const t = $("transcript-pane");
      t.textContent = (t.textContent + " " + finalText).trim();
      t.scrollTop = t.scrollHeight;
    }
  };
  recog.onend = () => { btnMic.textContent = "🎤 Start Listening"; btnMic.classList.add("accent"); };
}
setupMic();
btnMic.addEventListener("click", () => {
  if (!recog) return;
  if (btnMic.textContent.includes("Stop")) { recog.stop(); return; }
  try { recog.start(); btnMic.textContent = "⏹ Stop Listening"; btnMic.classList.remove("accent"); } catch(e) {}
});

$("sample-select").addEventListener("change", async (e) => {
  const key = e.target.value; if (!key) return;
  try {
    const r = await fetch(`/sample-conversation/${key}`).then(r => r.json());
    $("transcript-pane").textContent = r.transcript || "";
  } catch (err) { alert("Could not load sample: " + err.message); }
});

$("btn-clear").addEventListener("click", () => {
  $("transcript-pane").textContent = "";
  $("profile-grid").innerHTML = '<div class="muted">Run analyze to extract profile…</div>';
  $("recs").innerHTML = '<div class="muted">Recommendations will appear here.</div>';
  $("cart").innerHTML = '<div class="muted">Click "Build Cart" after recommendations appear.</div>';
  LAST_ENTITIES = LAST_REC = LAST_CART = null;
});

// ────────────────────────────────────────────────────────────────────
// Analyze flow: transcript → entities → recommendation → cart-ready
// ────────────────────────────────────────────────────────────────────
$("btn-analyze").addEventListener("click", async () => {
  const transcript = $("transcript-pane").textContent.trim();
  if (!transcript) { alert("Please load or speak a conversation first."); return; }
  $("profile-grid").innerHTML = '<div class="muted">Extracting profile on-device…</div>';
  $("recs").innerHTML = '<div class="muted">Building recommendations…</div>';
  try {
    const e = await api("/transcribe-turn", { transcript });
    LAST_ENTITIES = e.entities || {};
    renderProfile(LAST_ENTITIES);
    const rec = await api("/recommend", { entities: LAST_ENTITIES });
    LAST_REC = rec;
    renderRecs(rec);
    // Auto-build cart
    const c = await api("/cart", { recommendation: rec });
    LAST_CART = c;
    renderCart(c);
    refreshSavings();
  } catch (err) {
    $("recs").innerHTML = `<div class="muted">Error: ${esc(err.message)}</div>`;
  }
});

function renderProfile(ent) {
  const fields = [
    ["Segment", ent.segment], ["Budget", ent.budget_band],
    ["Lines", ent.lines_needed], ["Usage", ent.monthly_usage_gb ? ent.monthly_usage_gb + " GB" : null],
    ["Current carrier", ent.current_carrier], ["Current device", ent.current_device],
    ["Intl travel", ent.intl_travel === true ? "Yes" : ent.intl_travel === false ? "No" : null],
    ["Wants insurance", ent.wants_insurance ? "Yes" : null],
    ["Intended device", ent.intended_device],
  ].filter(([_, v]) => v !== null && v !== undefined && v !== "");
  const interests = (ent.interests || []).join(", ");
  const pains = (ent.pain_points || []).join(" • ");
  const grid = fields.map(([k, v]) => `<div><b>${esc(k)}</b>${esc(v)}</div>`).join("");
  const extras = (interests ? `<div style="grid-column:1/-1;"><b>Interests</b>${esc(interests)}</div>` : "")
               + (pains ? `<div style="grid-column:1/-1;"><b>Pain points</b>${esc(pains)}</div>` : "");
  $("profile-grid").innerHTML = (grid + extras) || '<div class="muted">No fields extracted.</div>';
}

function renderRecs(r) {
  const parts = [];
  if (r.rationale) parts.push(`<div class="card" style="background:#eef3fb;border-color:#cdd9eb;"><h3>Talk-track</h3><div>${esc(r.rationale)}</div></div>`);
  if (r.device) {
    parts.push(`<div class="card"><h3>Device</h3>
      <div class="title">${esc(r.device.name)}</div>
      <div class="price">$${r.device.price_24mo}/mo · 24-mo financing (or $${r.device.price_full} full)</div>
      <ul>${(r.device.highlights||[]).map(h=>`<li>${esc(h)}</li>`).join("")}</ul>
      <div class="tag-row">${(r.device.best_for||[]).map(t=>`<span class="tag">${esc(t)}</span>`).join("")}</div>
      ${r.device_alternative ? `<div class="muted" style="margin-top:8px;">Alt: ${esc(r.device_alternative.name)} ($${r.device_alternative.price_24mo}/mo)</div>` : ""}
    </div>`);
  }
  if (r.plan) {
    const price = r.plan.total_price ? `$${r.plan.total_price}/mo total` : `$${r.plan.price_per_line}/mo per line`;
    parts.push(`<div class="card"><h3>Plan</h3>
      <div class="title">${esc(r.plan.name)}</div>
      <div class="price">${price}${r.plan.data_gb ? ` · ${r.plan.data_gb} GB` : " · Unlimited"}${r.plan.hotspot_gb ? ` · ${r.plan.hotspot_gb} GB hotspot` : ""}</div>
      <ul>${(r.plan.perks||[]).map(p=>`<li>${esc(p)}</li>`).join("")}</ul>
    </div>`);
  }
  if (r.accessories && r.accessories.length) {
    parts.push(`<div class="card"><h3>Accessory bundle</h3>
      <ul>${r.accessories.map(a=>{
        const price = a.price_monthly ? `$${a.price_monthly}/mo` : a.monthly_24mo ? `$${a.monthly_24mo}/mo` : `$${a.price}`;
        return `<li><b>${esc(a.name)}</b> — ${price}</li>`;
      }).join("")}</ul>
    </div>`);
  }
  if (r.trade_in) {
    parts.push(`<div class="card"><h3>Trade-in</h3>
      <div>Customer's <b>${esc(r.trade_in.device)}</b> in ${esc(r.trade_in.condition_assumed)} condition: estimated <b>$${r.trade_in.estimated_value}</b> credit.</div>
    </div>`);
  }
  if (r.promos && r.promos.length) {
    parts.push(`<div class="card"><h3>Eligible promos</h3>
      <ul>${r.promos.map(p=>`<li><b>${esc(p.name)}</b><div class="muted">${esc((p.eligibility||[]).join(" · "))}</div></li>`).join("")}</ul>
    </div>`);
  }
  $("recs").innerHTML = parts.join("") || '<div class="muted">No recommendation produced.</div>';
}

function renderCart(c) {
  const items = (c.items || []).map(i => `
    <tr><td>${esc(i.label)}</td>
      <td style="text-align:right;">${i.monthly ? "$"+i.monthly.toFixed(2)+"/mo" : ""}</td>
      <td style="text-align:right;">${i.upfront ? "$"+i.upfront.toFixed(2) : ""}</td>
    </tr>`).join("");
  const promos = (c.promo_lines || []).map(p => `
    <tr style="color:var(--nw-good);"><td>${esc(p.label)}</td>
      <td colspan="2" style="text-align:right;">${p.amount ? "$"+p.amount.toFixed(2) : (p.note ? esc(p.note) : "")}</td>
    </tr>`).join("");
  $("cart").innerHTML = `
    <div id="cart-en">
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead><tr style="border-bottom:1px solid var(--nw-border);text-align:left;">
          <th>Item</th><th style="text-align:right;">Monthly</th><th style="text-align:right;">Upfront</th>
        </tr></thead>
        <tbody>${items}${promos}</tbody>
      </table>
      <div class="cart-totals">
        <div>Monthly: <span class="big">$${c.monthly_total.toFixed(2)}</span></div>
        <div>Upfront: $${c.upfront_total.toFixed(2)}</div>
        <div class="muted">24-month total: $${c.total_24mo.toFixed(2)}</div>
        <div class="uplift">+ $${c.total_value_uplift_24mo.toFixed(2)} total-value uplift vs Essentials baseline (24 mo)</div>
      </div>
    </div>
    <div id="cart-es" style="display:none;"><div class="muted">Click ES to translate on-device…</div></div>
  `;
}

// Language toggle for cart (calls /translate)
document.addEventListener("click", async (e) => {
  if (!e.target.matches(".lang-toggle button")) return;
  const lang = e.target.dataset.lang;
  document.querySelectorAll(".lang-toggle button").forEach(b => b.classList.toggle("active", b.dataset.lang === lang));
  CART_LANG = lang;
  const en = $("cart-en"), es = $("cart-es");
  if (!en || !es) return;
  if (lang === "en") { en.style.display = ""; es.style.display = "none"; return; }
  en.style.display = "none"; es.style.display = ""; es.innerHTML = '<div class="muted">Translating on-device…</div>';
  try {
    const enText = en.innerText;
    const r = await api("/translate", { text: enText, target: "es" });
    es.innerHTML = `<pre style="white-space:pre-wrap;font:inherit;">${esc(r.translated)}</pre>`;
    refreshSavings();
  } catch (err) {
    es.innerHTML = `<div class="muted">Translation failed: ${esc(err.message)}</div>`;
  }
});

// ────────────────────────────────────────────────────────────────────
// Coach chat
// ────────────────────────────────────────────────────────────────────
function pushChat(role, content) {
  const div = el(`<div class="chat-msg ${role}">${esc(content)}</div>`);
  $("chat-log").appendChild(div);
  div.scrollIntoView({behavior:"smooth", block:"end"});
}
async function sendChat() {
  const inp = $("chat-text");
  const msg = inp.value.trim(); if (!msg) return;
  inp.value = "";
  pushChat("user", msg);
  const thinking = el(`<div class="chat-msg bot">…</div>`);
  $("chat-log").appendChild(thinking);
  try {
    const r = await api("/coach", { message: msg, history: CHAT_HIST });
    thinking.textContent = r.reply || "(no reply)";
    CHAT_HIST.push({role:"user", content:msg}, {role:"assistant", content:r.reply || ""});
    refreshSavings();
  } catch (err) {
    thinking.textContent = "Error: " + err.message;
  }
}
$("chat-send").addEventListener("click", sendChat);
$("chat-text").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

// ────────────────────────────────────────────────────────────────────
// Offline toggle + savings widget
// ────────────────────────────────────────────────────────────────────
$("offline-toggle").addEventListener("click", () => {
  OFFLINE = !OFFLINE;
  $("offline-toggle").textContent = OFFLINE ? "Go Online" : "Go Offline";
  $("status-pill").classList.toggle("off", OFFLINE);
  $("status-text").textContent = OFFLINE ? "Simulated offline" : "NPU ready";
});

async function refreshSavings() {
  try {
    const r = await fetch("/audit").then(r => r.json());
    const tasks = r.tokenomics.local_tasks || 0;
    // Rough estimates for showcase: $0.002 / 1k tokens cloud, ~0.4g CO2 / 1k tokens cloud
    const tokens = r.tokenomics.approx_tokens || 0;
    const dollars = (tokens / 1000) * 0.002;
    const grams = (tokens / 1000) * 0.4;
    $("savings-footer").textContent =
      `Local AI tasks: ${tasks} • Estimated cloud cost saved: $${dollars.toFixed(3)} • CO₂ avoided: ${grams.toFixed(1)}g`;
  } catch {}
}
setInterval(refreshSavings, 4000);
</script>
</body>
</html>
"""

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    init_foundry()
    print(f"\n=== Northwind Mobile Seller Assistant ===")
    print(f"Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
