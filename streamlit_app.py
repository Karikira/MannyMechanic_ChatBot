"""
streamlit_app.py — Manny RAG Chatbot v4
========================================
Customer Mode : guided diagnosis + appointment booking (max 5/day)
Mechanic Mode : technical assistant + appointment management panel
"""

import base64, os
import streamlit as st
from groq import Groq
from config import Config
from rag import RAGPipeline
import appointments as appt_db

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Manny — Auto Repair Assistant", page_icon="🔧", layout="centered")

st.markdown("""
<style>
.mode-banner-customer{background:#eff6ff;border-left:5px solid #1e3a5f;padding:8px 14px;
  border-radius:6px;color:#1e3a5f;font-weight:600;font-size:14px;margin-bottom:4px;}
.mode-banner-mechanic{background:#f5f3ff;border-left:5px solid #7c3aed;padding:8px 14px;
  border-radius:6px;color:#7c3aed;font-weight:600;font-size:14px;margin-bottom:4px;}
.appt-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:14px 16px;margin-bottom:12px;}
.appt-card-started{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:14px 16px;margin-bottom:12px;}
.appt-card-finished{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:14px 16px;margin-bottom:12px;opacity:0.75;}
.appt-card-pending{background:#fef9f0;border:1px solid #fed7aa;border-radius:10px;
  padding:14px 16px;margin-bottom:12px;}
.slot-badge{display:inline-block;background:#1e3a5f;color:white;border-radius:999px;
  padding:3px 12px;font-size:13px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    cfg = Config()
    rag = RAGPipeline(cfg)
    rag.load_knowledge_base()
    return cfg, rag

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        st.error("GROQ_API_KEY not found in .streamlit/secrets.toml")
        st.stop()
    return Groq(api_key=api_key)

cfg, rag  = load_resources()
client    = get_groq_client()
MECH_PASS = st.secrets.get("MECHANIC_PASSWORD", "santos2024")

# ── Language support ─────────────────────────────────────────────────────────
LANGUAGES = {"en":"English","tl":"Tagalog"}

# simple translation map for a few static UI texts (expand as needed)
TRANSLATIONS = {
    "tl": {
        "Book an Appointment": "Mag-book ng Appointment",
        "Customer Mode — Active": "Mode ng Customer — Aktibo",
        "Mechanic Mode — Active": "Mode ng Mekaniko — Aktibo",
        "Switch to Mechanic Mode": "Lumipat sa Mode ng Mekaniko",
        "Switch to Customer Mode": "Lumipat sa Mode ng Customer",
        "Password": "Password",
        "Enter password...": "Ipasok ang password...",
        "Cancel": "Kanselahin",
        "Enter": "Ipasok",
        "Too many attempts. Restart the app to try again.": "Sobrang daming pagsubok. I-restart ang app upang subukan muli.",
        "❌ Wrong password. {remaining} attempt(s) left.": "❌ Mali ang password. {remaining} natitirang pagtatangka.",
        "Mechanic Access": "Access ng Mekaniko",
        "Language": "Wika",
        "🔧 {cfg.agent_name} — Auto Repair Assistant": "🔧 {cfg.agent_name} — Asistente sa Pag-aayos ng Sasakyan",
        "🔬 Mechanic Diagnostic Assistant": "🔬 Mekaniko Diagnostikong Asistente",
        "Hi! I'm **{cfg.agent_name}**, your AI mechanic assistant from **{cfg.shop_name}**. 👋\n\n"      : "Hi! Ako si **{cfg.agent_name}**, iyong AI na mekanikong katulong mula sa **{cfg.shop_name}**. 👋\n\n",
    }
}

def t(text: str) -> str:
    lang = st.session_state.get("lang", "en")
    if lang == "tl":
        return TRANSLATIONS.get("tl", {}).get(text, text)
    return text


def call_groq(system, history, extra=None):
    # include language instruction in system message
    lang = st.session_state.get("lang", "en")
    if lang == "tl":
        system = system + "\nPara sa susunod na mga sagot, gamitin ang Tagalog."  # instruct model to reply in Tagalog
    msgs = [{"role":"system","content":system}] + history
    if extra: msgs += extra
    return client.chat.completions.create(
        model=cfg.model_name, messages=msgs, max_tokens=cfg.max_tokens
    ).choices[0].message.content

# ── Session state ─────────────────────────────────────────────────────────────
def _fresh_diag():
    return {"mode":"idle","symptom":"","source":"text",
            "img_desc":"","questions":[],"answers":[],"q_index":0}

def _fresh_mech_diag():
    return {"mode":"idle","vehicle":"","complaint":""}

_defaults = {
    "app_mode":         "customer",
    "mech_unlocked":    False,
    "show_pw_prompt":   False,
    "pw_error":         False,
    "pw_attempts":      0,
    "show_booking":     False,
    "booking_done":     False,
    "booked_appt":      None,
    "last_diagnosis":   "",        # pre-fill concern field
    "messages":         [],
    "conv_history":     [],
    "diag":             None,
    "mech_messages":    [],
    "mech_history":     [],
    "mech_diag":        None,
    "lang":             "en",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.diag is None:
    st.session_state.diag = _fresh_diag()
if st.session_state.mech_diag is None:
    st.session_state.mech_diag = _fresh_mech_diag()

# ── LLM helpers ───────────────────────────────────────────────────────────────
# (call_groq defined earlier above when language support was added)

def call_vision(image_b64, mime_type, prompt_text):
    try:
        return client.chat.completions.create(
            model=cfg.vision_model,
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{mime_type};base64,{image_b64}"}},
                {"type":"text","text":prompt_text},
            ]}],
            max_tokens=cfg.max_tokens,
        ).choices[0].message.content
    except Exception as e:
        # handle common Groq model errors and guide the user
        try:
            from groq import BadRequestError, NotFoundError
            if isinstance(e, BadRequestError) and hasattr(e, 'response'):
                err = e.response.json().get('error', {})
                code = err.get('code', '')
                if code == 'model_decommissioned':
                    st.error(
                        "⚠️ Vision model appears to be decommissioned. "
                        "Please update `vision_name` in agent.yaml to a supported model."
                    )
                else:
                    st.error(f"⚠️ Vision request failed: {err.get('message','unknown')}")
            elif isinstance(e, NotFoundError):
                # model not available or user lacks access
                st.error(
                    "⚠️ Vision model not found or inaccessible. "
                    "You can either disable vision by setting `vision.enabled` to false in agent.yaml "
                    "or choose a model you have access to (see Groq console)."
                )
        except ImportError:
            pass
        # re-raise so the full traceback is visible if needed
        raise

# ── Detect helpers ────────────────────────────────────────────────────────────
def is_symptom(text):
    return any(kw in text.lower() for kw in cfg.symptom_keywords)

def is_car_model(text):
    if not cfg.car_lookup_enabled: return False
    t = text.lower()
    return any(b in t for b in cfg.car_lookup_brands) and not is_symptom(text)

def extract_severity(text):
    t = text.upper()
    if "RED" in t:    return "red"
    if "YELLOW" in t: return "yellow"
    if "GREEN" in t:  return "green"
    return "unknown"

# ── UI helpers ────────────────────────────────────────────────────────────────
SEV = {
    "red":     ("#b91c1c","#fee2e2","🔴 SERIOUS — Do NOT drive"),
    "yellow":  ("#d97706","#fef3c7","🟡 MODERATE — Get checked soon"),
    "green":   ("#16a34a","#dcfce7","🟢 MINOR — Safe to drive for now"),
    "unknown": ("#64748b","#f1f5f9","⚪ Diagnosis complete"),
}

def render_severity(sev):
    col, bg, label = SEV.get(sev, SEV["unknown"])
    st.markdown(f'<div style="background:{bg};border-left:5px solid {col};padding:10px 16px;'
                f'border-radius:6px;margin:8px 0;font-weight:bold;color:{col};font-size:16px;">'
                f'{label}</div>', unsafe_allow_html=True)

def render_progress(q_num, total):
    pct = int((q_num-1)/total*100)
    st.markdown(f'<div style="background:#e2e8f0;border-radius:999px;height:8px;margin:4px 0 12px;">'
                f'<div style="background:#1e3a5f;width:{pct}%;height:8px;border-radius:999px;"></div>'
                f'</div><p style="color:#64748b;font-size:13px;margin:0;">Question {q_num} of {total}</p>',
                unsafe_allow_html=True)

def render_car_problems(text):
    lines, header, problems = text.strip().split("\n"), "", []
    for line in lines:
        line = line.strip()
        if line.startswith("COMMON PROBLEMS:"): header = line
        elif line and line[0].isdigit() and ". " in line: problems.append(line)
    if header: st.markdown(f"**{header}**")
    if problems:
        st.markdown("*Click a problem to start the guided diagnosis:*")
        for prob in problems:
            if st.button(prob, key=f"prob_{prob[:40]}"):
                handle_customer_message(prob); st.rerun()
    else:
        st.markdown(text)

# =============================================================================
#  CUSTOMER MODE — CHAT HANDLERS
# =============================================================================
def handle_customer_message(user_msg):
    d = st.session_state.diag
    st.session_state.conv_history.append({"role":"user","content":user_msg})
    st.session_state.messages.append({"role":"user","content":user_msg,"type":"text"})

    if d["mode"] == "diagnosing":
        d["answers"].append(user_msg)
        d["q_index"] += 1
        if d["q_index"] < len(d["questions"]):
            q_num  = d["q_index"]+1
            total  = len(d["questions"])
            reply  = f"**Question {q_num} of {total}:** {d['questions'][d['q_index']]}"
            st.session_state.conv_history.append({"role":"assistant","content":reply})
            st.session_state.messages.append({"role":"assistant","content":reply,
                                               "type":"question","q_num":q_num,"total":total})
            return
        # All answers collected
        base  = d["img_desc"] if d["source"]=="image" else d["symptom"]
        rich  = f"Symptom: {base}. "+" ".join(f"Q:{q} A:{a}" for q,a in zip(d["questions"],d["answers"]))
        ctx, sources = rag.retrieve(rich)
        qa   = "\n".join(f"Q:{q}\nA:{a}" for q,a in zip(d["questions"],d["answers"]))
        req  = (f"Photo visual analysis:\n{d['img_desc']}\nNote:{d['symptom']}\nInterview:\n{qa}\nDiagnose."
                if d["source"]=="image" else
                f"Symptom:{d['symptom']}\nInterview:\n{qa}\nDiagnose.")
        ans  = call_groq(rag.inject_context(cfg.prompt_diagnosis,ctx), st.session_state.conv_history,
                         [{"role":"user","content":req}])
        sev  = extract_severity(ans)
        # Save diagnosis text for pre-filling appointment form
        # `d['symptom']` should normally be a string, but guard just in case
        sym_text = d.get('symptom') or ''
        preview = ans[:200] if ans else ''
        st.session_state.last_diagnosis = f"{sym_text} — {preview}"
        st.session_state.conv_history.append({"role":"assistant","content":ans})
        st.session_state.messages.append({"role":"assistant","content":ans,
                                           "type":"diagnosis","severity":sev,"sources":sources})
        st.session_state.diag = _fresh_diag()
        return

    if is_symptom(user_msg):
        d.update({"mode":"diagnosing","symptom":user_msg,"source":"text",
                  "img_desc":"","questions":cfg.diagnostic_questions[:],"answers":[],"q_index":0})
        total = len(cfg.diagnostic_questions)
        reply = (f"I'd like to help diagnose that! I'll ask {total} quick questions.\n\n"
                 f"**Question 1 of {total}:** {cfg.diagnostic_questions[0]}")
        st.session_state.conv_history.append({"role":"assistant","content":reply})
        st.session_state.messages.append({"role":"assistant","content":reply,
                                           "type":"question","q_num":1,"total":total})
        return

    if is_car_model(user_msg):
        ctx, sources = rag.retrieve(f"{user_msg} common problems failures issues")
        prompt = cfg.car_lookup_prompt.format(car_model=user_msg, num_problems=cfg.car_lookup_num)
        ans = call_groq(rag.inject_context(prompt,ctx), st.session_state.conv_history)
        st.session_state.conv_history.append({"role":"assistant","content":ans})
        st.session_state.messages.append({"role":"assistant","content":ans,
                                           "type":"car_problems","sources":sources})
        return

    ctx, sources = rag.retrieve(user_msg)
    ans = call_groq(rag.inject_context(cfg.prompt_base,ctx), st.session_state.conv_history)
    st.session_state.conv_history.append({"role":"assistant","content":ans})
    st.session_state.messages.append({"role":"assistant","content":ans,"type":"qa","sources":sources})


def handle_customer_image(uploaded_file, user_note=""):
    if not cfg.vision_enabled: st.warning("Vision analysis is disabled."); return
    raw = uploaded_file.read()
    if len(raw)/(1024*1024) > cfg.vision_max_mb: st.error(f"Image too large."); return
    if uploaded_file.type not in cfg.vision_formats: st.error("Unsupported format."); return
    b64 = base64.b64encode(raw).decode()
    with st.spinner("Analysing your photo..."):
        desc = call_vision(b64, uploaded_file.type,
                           cfg.vision_prompt+(f"\nCustomer note:{user_note}" if user_note else ""))
    log = f"[Photo]{(' — '+user_note) if user_note else ''}\n\nVisual analysis: {desc}"
    st.session_state.conv_history.append({"role":"user","content":log})
    st.session_state.messages.append({"role":"user","content":log,"type":"image","img_bytes":raw})
    d = st.session_state.diag
    d.update({"mode":"diagnosing","symptom":user_note or "Customer uploaded a photo.",
              "source":"image","img_desc":desc,"questions":cfg.diagnostic_questions[:],"answers":[],"q_index":0})
    total = len(cfg.diagnostic_questions)
    reply = (f"I've analysed your photo:\n\n*{desc}*\n\n---\n\n"
             f"I have {total} follow-up questions.\n\n"
             f"**Question 1 of {total}:** {cfg.diagnostic_questions[0]}")
    st.session_state.conv_history.append({"role":"assistant","content":reply})
    st.session_state.messages.append({"role":"assistant","content":reply,
                                       "type":"question","q_num":1,"total":total})

# =============================================================================
#  CUSTOMER MODE — BOOKING FORM
# =============================================================================
def render_booking_form():
    slots = appt_db.slots_available()

    st.markdown("## 📅 Book an Appointment")

    # Slot availability
    if slots == 0:
        st.error("😔 Sorry, we're fully booked for today. Please call us to arrange another date.")
        st.markdown(f"📞 **{cfg.shop_phone}**")
        if st.button("← Back to chat"):
            st.session_state.show_booking = False; st.rerun()
        return

    color = "#16a34a" if slots >= 3 else "#d97706" if slots >= 1 else "#b91c1c"
    st.markdown(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:14px 20px;margin-bottom:20px;">'
        f'<span style="font-size:15px;">Available slots today: </span>'
        f'<span style="font-weight:700;color:{color};font-size:18px;">{slots} / {appt_db.MAX_PER_DAY}</span>'
        f'</div>', unsafe_allow_html=True)

    # If booking just completed — show confirmation
    if st.session_state.booking_done and st.session_state.booked_appt:
        a = st.session_state.booked_appt
        st.success("✅ Appointment request received!")
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:18px 20px;">'
            f'<p style="font-size:17px;font-weight:700;color:#15803d;margin:0 0 12px;">Booking Confirmed 🎉</p>'
            f'<p style="margin:4px 0;"><strong>Booking ID:</strong> #{a["id"]}</p>'
            f'<p style="margin:4px 0;"><strong>Name:</strong> {a["name"]}</p>'
            f'<p style="margin:4px 0;"><strong>Vehicle:</strong> {a["vehicle"]}</p>'
            f'<p style="margin:4px 0;"><strong>Concern:</strong> {a["concern"]}</p>'
            f'<p style="margin:4px 0;"><strong>Preferred time:</strong> {a["preferred_time"]}</p>'
            f'<p style="margin:4px 0;"><strong>Booked at:</strong> {a["booked_at"]}</p>'
            f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px 20px;">'
            f'<p style="font-weight:700;color:#1e3a5f;font-size:15px;margin:0 0 8px;">'
            f'📞 Questions? Give us a call!</p>'
            f'<p style="font-size:20px;font-weight:700;color:#1e3a5f;margin:0;">{cfg.shop_phone}</p>'
            f'<p style="color:#64748b;font-size:13px;margin:6px 0 0;">{cfg.shop_hours}</p>'
            f'</div>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("← Back to chat", use_container_width=True):
            st.session_state.show_booking = False
            st.session_state.booking_done = False
            st.session_state.booked_appt  = None
            st.rerun()
        return

    # Booking form
    with st.form("booking_form", clear_on_submit=False):
        st.markdown("#### Your Details")
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Full Name *", placeholder="e.g. Maria Santos")
        with col2:
            phone = st.text_input("Contact Number *", placeholder="e.g. 09171234567")

        st.markdown("#### Vehicle")
        vehicle = st.text_input("Make, Model & Year *",
                                placeholder="e.g. Toyota Vios 2019, Honda Civic 2020")

        st.markdown("#### Concern")
        pre = st.session_state.last_diagnosis[:150] if st.session_state.last_diagnosis else ""
        concern = st.text_area("Describe the issue *", value=pre,
                               placeholder="e.g. Brakes grinding when stopping, engine vibration at idle",
                               height=90)

        st.markdown("#### Preferred Time")
        time_pref = st.selectbox("Preferred time slot",
                                  ["Morning (8:00 AM – 12:00 PM)",
                                   "Afternoon (12:00 PM – 5:00 PM)",
                                   "Any available slot"])

        st.markdown("")
        submitted = st.form_submit_button("📅 Confirm Appointment", use_container_width=True)

        if submitted:
            errors = []
            if not name.strip():    errors.append("Full name is required.")
            if not phone.strip():   errors.append("Contact number is required.")
            if not vehicle.strip(): errors.append("Vehicle details are required.")
            if not concern.strip(): errors.append("Please describe the concern.")
            if appt_db.is_full():   errors.append("No slots available today.")

            if errors:
                for e in errors: st.error(e)
            else:
                new_appt = appt_db.book_appointment(name, phone, vehicle, concern, time_pref)
                st.session_state.booking_done = True
                st.session_state.booked_appt  = new_appt
                st.rerun()

    st.markdown("")
    if st.button("← Back to chat", use_container_width=True):
        st.session_state.show_booking = False; st.rerun()

# =============================================================================
#  MECHANIC MODE — CHAT HANDLERS
# =============================================================================
def handle_mechanic_message(user_msg):
    md = st.session_state.mech_diag
    st.session_state.mech_history.append({"role":"user","content":user_msg})
    st.session_state.mech_messages.append({"role":"user","content":user_msg})

    if md["mode"] == "awaiting_findings":
        ctx, sources = rag.retrieve(f"{md['vehicle']} {md['complaint']} {user_msg}")
        req = (f"Vehicle:{md['vehicle']}\nComplaint:{md['complaint']}\nFindings:{user_msg}\nFull technical diagnosis.")
        ans = call_groq(rag.inject_context(cfg.mechanic_prompt_diagnosis,ctx),
                        st.session_state.mech_history,[{"role":"user","content":req}])
        st.session_state.mech_history.append({"role":"assistant","content":ans})
        st.session_state.mech_messages.append({"role":"assistant","content":ans,
                                                "type":"mech_diagnosis","sources":sources})
        st.session_state.mech_diag = _fresh_mech_diag()
        return

    lower = user_msg.lower()
    is_diag = any(kw in lower for kw in ["diagnose","what's wrong","whats wrong","why is","why does",
                                          "fault","dtc","code p","check engine","won't start","wont start",
                                          "not starting","not working","problem with","issue with"])
    has_car = any(b in lower for b in cfg.car_lookup_brands)
    if is_diag and has_car and len(user_msg)>20:
        md.update({"mode":"awaiting_findings","vehicle":user_msg,"complaint":user_msg})
        reply = ("Got it. What are your findings?\n\n"
                 "- DTC / fault codes\n- Symptom description\n"
                 "- When it occurs\n- Any recent work done\n- Visual observations")
        st.session_state.mech_history.append({"role":"assistant","content":reply})
        st.session_state.mech_messages.append({"role":"assistant","content":reply,"type":"mech_question"})
        return

    ctx, sources = rag.retrieve(user_msg)
    ans = call_groq(rag.inject_context(cfg.mechanic_prompt_base,ctx), st.session_state.mech_history)
    st.session_state.mech_history.append({"role":"assistant","content":ans})
    st.session_state.mech_messages.append({"role":"assistant","content":ans,"type":"mech_qa","sources":sources})


def handle_mechanic_image(uploaded_file, note=""):
    raw = uploaded_file.read()
    b64 = base64.b64encode(raw).decode()
    prompt = ("Expert automotive technician examining a mechanic's photo. "
              "Identify component, describe fault/wear/damage, suggest root cause, "
              "recommend next step. Be concise and technical."
              +(f"\nMechanic note:{note}" if note else ""))
    with st.spinner("Analysing..."):
        analysis = call_vision(b64, uploaded_file.type, prompt)
    st.session_state.mech_history.append({"role":"user","content":f"[Photo] {note}"})
    st.session_state.mech_messages.append({"role":"user","content":f"[Photo] {note}",
                                            "type":"image","img_bytes":raw})
    st.session_state.mech_history.append({"role":"assistant","content":analysis})
    st.session_state.mech_messages.append({"role":"assistant","content":analysis,"type":"mech_qa"})

# =============================================================================
#  MECHANIC MODE — APPOINTMENTS PANEL
# =============================================================================
def render_appointments_panel():
    from appointments import (PENDING, ACCEPTED, STARTED, FINISHED, DECLINED,
                               STATUS_LABELS, LABEL_TO_STATUS,
                               get_today_appointments, update_appointment,
                               active_count_today, MAX_PER_DAY)
    import datetime as dt

    today = dt.date.today().strftime("%A, %B %d, %Y")
    active = active_count_today()

    st.markdown(f"### 📅 Appointments — {today}")

    # Slot meter
    pct  = int(active / MAX_PER_DAY * 100)
    free = MAX_PER_DAY - active
    bar_color = "#16a34a" if free >= 3 else "#d97706" if free >= 1 else "#b91c1c"
    st.markdown(
        f'<div style="margin-bottom:16px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-weight:600;font-size:14px;">Slots used today</span>'
        f'<span style="font-weight:700;color:{bar_color};">{active}/{MAX_PER_DAY}</span>'
        f'</div>'
        f'<div style="background:#e2e8f0;border-radius:999px;height:10px;">'
        f'<div style="background:{bar_color};width:{pct}%;height:10px;border-radius:999px;'
        f'transition:width 0.3s;"></div></div>'
        f'<p style="color:#64748b;font-size:12px;margin:4px 0 0;">'
        f'{free} slot(s) available — Finished appointments free up slots</p>'
        f'</div>', unsafe_allow_html=True)

    appointments = get_today_appointments()

    if not appointments:
        st.info("No appointments booked yet today.")
        return

    # ── Group by status ────────────────────────────────────────────────────
    pending  = [a for a in appointments if a["status"] == PENDING]
    active_a = [a for a in appointments if a["status"] in (ACCEPTED, STARTED)]
    finished = [a for a in appointments if a["status"] in (FINISHED, DECLINED)]

    # ── PENDING ────────────────────────────────────────────────────────────
    if pending:
        st.markdown(f"#### 🟠 Pending Approval ({len(pending)})")
        for a in pending:
            st.markdown(
                f'<div class="appt-card-pending">'
                f'<strong>#{a["id"]}</strong> — <strong>{a["name"]}</strong><br>'
                f'🚗 {a["vehicle"]}<br>'
                f'🔍 {a["concern"]}<br>'
                f'🕐 {a["preferred_time"]} &nbsp;|&nbsp; Booked at {a["booked_at"]}<br>'
                f'📞 {a["phone"]}'
                f'</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"✅ Accept #{a['id']}", key=f"accept_{a['id']}", use_container_width=True):
                    update_appointment(a["id"], status=ACCEPTED)
                    st.rerun()
            with c2:
                if st.button(f"❌ Decline #{a['id']}", key=f"decline_{a['id']}", use_container_width=True):
                    update_appointment(a["id"], status=DECLINED)
                    st.rerun()

    # ── ACTIVE (accepted + started) ────────────────────────────────────────
    if active_a:
        st.markdown(f"#### 🔵 Active ({len(active_a)})")
        dropdown_options = list(STATUS_LABELS.values())  # ["Hasn't arrived yet", "Started", "Finished"]

        for a in active_a:
            card_class = "appt-card-started" if a["status"] == STARTED else "appt-card"
            st.markdown(
                f'<div class="{card_class}">'
                f'<strong>#{a["id"]}</strong> — <strong>{a["name"]}</strong><br>'
                f'🚗 {a["vehicle"]}<br>'
                f'🔍 {a["concern"]}<br>'
                f'🕐 {a["preferred_time"]}<br>'
                f'📞 Customer: {a["phone"]}'
                f'</div>', unsafe_allow_html=True)

            current_label = STATUS_LABELS.get(a["status"], "Hasn't arrived yet")
            current_idx   = dropdown_options.index(current_label)

            def on_status_change(appt_id=a["id"], orig_phone=a["phone"], current_status=a["status"]):
                key        = f"status_sel_{appt_id}"
                new_label  = st.session_state[key]
                new_status = LABEL_TO_STATUS[new_label]
                # if moving into STARTED, prefill call number from booking phone
                if new_status == STARTED:
                    # only set if not already specified by mechanic
                    if not a.get("customer_call_phone"):
                        update_appointment(appt_id, status=new_status, customer_call_phone=orig_phone)
                        return
                # if moving away from started, clear the call number (stateless)
                if current_status == STARTED and new_status != STARTED:
                    update_appointment(appt_id, status=new_status, customer_call_phone="")
                    return
                update_appointment(appt_id, status=new_status)

            st.selectbox(
                f"Status — #{a['id']}",
                options    = dropdown_options,
                index      = current_idx,
                key        = f"status_sel_{a['id']}",
                on_change  = on_status_change,
                label_visibility = "visible",
            )

            # Phone input when status is Started
            if a["status"] == STARTED:
                current_call_phone = a.get("customer_call_phone","")

                def on_phone_save(appt_id=a["id"]):
                    new_phone = st.session_state[f"call_phone_{appt_id}"]
                    update_appointment(appt_id, customer_call_phone=new_phone)

                st.text_input(
                    f"📞 Customer number to call when done — #{a['id']}",
                    value        = current_call_phone,
                    placeholder  = "e.g. 09171234567",
                    key          = f"call_phone_{a['id']}",
                    on_change    = on_phone_save,
                )
                if current_call_phone:
                    st.markdown(
                        f'<div style="background:#eff6ff;border-radius:6px;padding:8px 12px;'
                        f'margin:-4px 0 8px;font-size:13px;">'
                        f'📞 Call <strong>{a["name"]}</strong> at '
                        f'<strong>{current_call_phone}</strong> when job is done.</div>',
                        unsafe_allow_html=True)

            st.markdown("---")

    # ── FINISHED / DECLINED ────────────────────────────────────────────────
    if finished:
        with st.expander(f"✅ Completed / Declined today ({len(finished)})", expanded=False):
            for a in finished:
                icon = "✅" if a["status"] == FINISHED else "❌"
                # always show phone for started/finished entries
                st.markdown(
                    f'<div class="appt-card-finished">'
                    f'{icon} <strong>#{a["id"]}</strong> — {a["name"]} — {a["vehicle"]}<br>'
                    f'📞 {a["phone"]}<br>'
                    f'<span style="color:#64748b;font-size:13px;">{a["concern"]}</span>'
                    f'</div>', unsafe_allow_html=True)
                # allow mechanic to remove/archive the record
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"🗑 Remove #{a['id']}", key=f"remove_{a['id']}", use_container_width=True):
                        from appointments import delete_appointment
                        delete_appointment(a['id'])
                        st.rerun()
                with c2:
                    st.write("")  # placeholder column to keep layout

# =============================================================================
#  PASSWORD GATE
# =============================================================================
def try_mechanic_login(pw):
    if st.session_state.pw_attempts >= 5: return
    if pw == MECH_PASS:
        st.session_state.update({"mech_unlocked":True,"app_mode":"mechanic",
                                  "show_pw_prompt":False,"pw_error":False,"pw_attempts":0})
    else:
        st.session_state.pw_attempts += 1
        st.session_state.pw_error = True

# =============================================================================
#  SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:8px 0 4px;">'
        f'<span style="font-size:36px;">🔧</span><br>'
        f'<strong style="font-size:16px;">{cfg.agent_name}</strong><br>'
        f'<span style="color:#64748b;font-size:12px;">{cfg.shop_city}</span>'
        f'</div>', unsafe_allow_html=True)
    st.caption(f"📞 {cfg.shop_phone}")
    st.caption(f"🕐 {cfg.shop_hours}")
    st.divider()

    # ── Mode switcher ──────────────────────────────────────────────────────
    st.markdown("**" + t("Mode") + "**")
    if st.session_state.app_mode == "customer":
        st.markdown('<div class="mode-banner-customer">' + t("Customer Mode — Active") + '</div>',
                    unsafe_allow_html=True)
        if not st.session_state.show_pw_prompt:
            if st.button(t("Switch to Mechanic Mode"), use_container_width=True):
                if st.session_state.mech_unlocked:
                    st.session_state.app_mode = "mechanic"; st.rerun()
                else:
                    st.session_state.show_pw_prompt = True
                    st.session_state.pw_error = False; st.rerun()
        if st.session_state.show_pw_prompt:
            st.markdown(
                '<div style="background:#f5f3ff;border:1px solid #c4b5fd;border-radius:8px;'
                'padding:12px 14px;margin:6px 0;"><strong style="color:#5b21b6;">🔒 Mechanic Access</strong><br>'
                '<span style="font-size:13px;color:#6b7280;">Enter your shop password</span></div>',
                unsafe_allow_html=True)
            if st.session_state.pw_attempts >= 5:
                st.error("🚫 Too many attempts. Restart the app to try again.")
            else:
                pw_in = st.text_input(t("Password"), type="password",
                                       placeholder=t("Enter password..."),
                                       label_visibility="collapsed", key="pw_field")
                ca, cb = st.columns(2)
                with ca:
                    if st.button(f"✅ {t('Enter')}", use_container_width=True):
                        try_mechanic_login(pw_in); st.rerun()
                with cb:
                    if st.button(f"✖ {t('Cancel')}", use_container_width=True):
                        st.session_state.show_pw_prompt = False
                        st.session_state.pw_error = False; st.rerun()
                if st.session_state.pw_error:
                    st.error(t("❌ Wrong password. {remaining} attempt(s) left.").format(remaining=5-st.session_state.pw_attempts))
    else:
        st.markdown('<div class="mode-banner-mechanic">🔬 Mechanic Mode — Active</div>',
                    unsafe_allow_html=True)
        if st.button("🧑 Switch to Customer Mode", use_container_width=True):
            # lock mechanic mode again so a password is required the next time
            st.session_state.app_mode = "customer"
            st.session_state.mech_unlocked = False
            st.session_state.show_pw_prompt = False; st.rerun()

    st.divider()

    # ── Book appointment button (customer only) ────────────────────────────
    # ── Language selector ───────────────────────────────────────────────
    st.markdown("**" + t("Language") + "**")
    sel = st.selectbox("", options=list(LANGUAGES.keys()),
                       format_func=lambda k: LANGUAGES[k], index=list(LANGUAGES.keys()).index(st.session_state.get("lang","en")),
                       key="lang")
    # fall through to book appointment section
    if st.session_state.app_mode == "customer":
        slots = appt_db.slots_available()
        slot_color = "#16a34a" if slots >= 3 else "#d97706" if slots >= 1 else "#b91c1c"
        st.markdown(
            f'<p style="margin:0 0 6px;font-size:13px;">Today\'s slots: '
            f'<strong style="color:{slot_color};">{slots}/{appt_db.MAX_PER_DAY} available</strong></p>',
            unsafe_allow_html=True)
        btn_label = "📅 Book an Appointment" if slots > 0 else "📅 View Appointment (Full)"
        if st.button(btn_label, use_container_width=True):
            st.session_state.show_booking  = True
            st.session_state.booking_done  = False
            st.rerun()
        st.divider()

    # ── Image upload ───────────────────────────────────────────────────────
    if cfg.vision_enabled:
        label = "📷 Upload a photo" if st.session_state.app_mode=="customer" else "📷 Upload vehicle photo"
        st.markdown(f"**{label}**")
        uploaded = st.file_uploader("Photo", type=["jpg","jpeg","png","webp"],
                                     label_visibility="collapsed")
        img_note = st.text_input("Note", label_visibility="collapsed",
                                  placeholder="Add a note..." if st.session_state.app_mode=="customer"
                                  else "e.g. DTC P0300, Vios 2018")
        btn = "🔍 Analyse Photo" if st.session_state.app_mode=="customer" else "🔬 Technical Analysis"
        if st.button(btn, use_container_width=True):
            if uploaded:
                if st.session_state.app_mode=="customer": handle_customer_image(uploaded, img_note)
                else: handle_mechanic_image(uploaded, img_note)
                st.rerun()
            else: st.warning("Please upload an image first.")
        st.divider()

    # ── Status ─────────────────────────────────────────────────────────────
    if st.session_state.app_mode == "customer":
        if st.session_state.diag["mode"] == "diagnosing":
            q = st.session_state.diag["q_index"]
            total = len(st.session_state.diag["questions"])
            st.info(f"🩺 Diagnosing... Q{q+1} of {total}")
        else: st.success("💬 Ready to help")
    else:
        if st.session_state.mech_diag["mode"] == "awaiting_findings": st.info("🔬 Awaiting findings...")
        else: st.success("🔧 Mechanic assistant ready")

    st.caption(f"📚 {rag.chunk_count} knowledge chunks")
    st.caption(f"🤖 {cfg.model_name}")
    st.divider()

    reset_label = "🔄 Reset Chat" if st.session_state.app_mode=="customer" else "🔄 Reset Mechanic Chat"
    if st.button(reset_label, use_container_width=True):
        if st.session_state.app_mode == "customer":
            st.session_state.messages = []; st.session_state.conv_history = []
            st.session_state.diag = _fresh_diag()
            st.session_state.show_booking = False; st.session_state.booking_done = False
        else:
            st.session_state.mech_messages = []; st.session_state.mech_history = []
            st.session_state.mech_diag = _fresh_mech_diag()
        st.rerun()

# =============================================================================
#  CUSTOMER MODE MAIN AREA
# =============================================================================
if st.session_state.app_mode == "customer":

    if st.session_state.show_booking:
        render_booking_form()

    else:
        # header localized
        if st.session_state.lang == "tl":
            st.markdown(
                f'<h2 style="margin-bottom:4px;">🔧 {cfg.agent_name} — Asistente sa Pag-aayos ng Sasakyan</h2>'
                f'<p style="color:#64748b;margin-top:0;">Itanong mo sa akin ang anumang tungkol sa iyong sasakyan o serbisyo namin.</p>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<h2 style="margin-bottom:4px;">🔧 {cfg.agent_name} — Auto Repair Assistant</h2>'
                f'<p style="color:#64748b;margin-top:0;">Ask me anything about your vehicle or our services.</p>',
                unsafe_allow_html=True)

        if not st.session_state.messages:
            with st.chat_message("assistant"):
                if st.session_state.lang == "tl":
                    st.markdown(
                        (
                            f"Hi! Ako si **{cfg.agent_name}**, iyong AI na mekanikong katulong mula sa **{cfg.shop_name}**. 👋\n\n"
                            "Maaari kang:\n"
                            "- 🔍 **Ilarawan ang sintomas** — *nga ang preno ko ay umiinis*\n"
                            "- 🚗 **Hanapin ang iyong kotse** — *Toyota Vios 2018*\n"
                            "- 📷 **Mag-upload ng larawan** ng problema (sidebar)\n"
                            "- 📅 **Mag-book ng appointment** — pindutin ang button sa sidebar\n"
                            "- ❓ **Magtanong tungkol sa serbisyo o presyo**\n\n"
                            "Paano kita matutulungan ngayon?"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"Hi! I'm **{cfg.agent_name}**, your AI mechanic assistant from "
                        f"**{cfg.shop_name}**. 👋\n\n"
                        f"You can:\n"
                        f"- 🔍 **Describe a symptom** — *my brakes are grinding*\n"
                        f"- 🚗 **Look up your car** — *Toyota Vios 2018*\n"
                        f"- 📷 **Upload a photo** of the issue (sidebar)\n"
                        f"- 📅 **Book an appointment** — tap the button in the sidebar\n"
                        f"- ❓ **Ask about services or pricing**\n\n"
                        f"How can I help you today?"
                    )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                mtype = msg.get("type","text")
                if mtype == "image":
                    if "img_bytes" in msg: st.image(msg["img_bytes"], width=280)
                    for line in msg["content"].split("\n"):
                        if line.startswith("Visual analysis:"): st.caption("📷 "+line); break
                elif mtype == "question":
                    render_progress(msg.get("q_num",1), msg.get("total",4))
                    st.markdown(msg["content"])
                elif mtype == "diagnosis":
                    render_severity(msg.get("severity","unknown"))
                    st.markdown(msg["content"])
                    if msg.get("sources"): st.caption(f"📚 {', '.join(msg['sources'])}")
                    # Offer to book after diagnosis
                    if st.button("📅 Book an appointment", key=f"book_after_{id(msg)}"):
                        st.session_state.show_booking = True; st.rerun()
                elif mtype == "car_problems":
                    render_car_problems(msg["content"])
                    if msg.get("sources"): st.caption(f"📚 {', '.join(msg['sources'])}")
                else:
                    st.markdown(msg["content"])
                    if msg.get("sources"): st.caption(f"📚 {', '.join(msg['sources'])}")

        placeholder = ("Answer the question above..."
                       if st.session_state.diag["mode"]=="diagnosing"
                       else "Describe your car issue, ask a question, or type a car model...")
        if user_input := st.chat_input(placeholder):
            with st.spinner("Manny is thinking..."):
                handle_customer_message(user_input)
            st.rerun()

# =============================================================================
#  MECHANIC MODE MAIN AREA
# =============================================================================
else:
    tab_chat, tab_appts = st.tabs(["💬 Diagnostic Chat", "📅 Appointments"])

    # ── Chat tab ───────────────────────────────────────────────────────────
    with tab_chat:
        if st.session_state.lang == "tl":
            st.markdown(
                '<h2 style="margin-bottom:4px;">🔬 Mekaniko Diagnostikong Asistente</h2>'
                '<p style="color:#64748b;margin-top:0;">Teknikal na mode — para sa mga kawani ng shop lang.</p>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<h2 style="margin-bottom:4px;">🔬 Mechanic Diagnostic Assistant</h2>'
                '<p style="color:#64748b;margin-top:0;">Technical mode — shop staff only.</p>',
                unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔎 Diagnose a fault", use_container_width=True):
                handle_mechanic_message("I need help diagnosing a fault"); st.rerun()
        with c2:
            if st.button("📋 Torque specs", use_container_width=True):
                handle_mechanic_message("Common torque specs Toyota Mitsubishi"); st.rerun()
        with c3:
            if st.button("🔩 Repair procedure", use_container_width=True):
                handle_mechanic_message("Timing belt replacement 4D56 procedure"); st.rerun()

        st.divider()

        if not st.session_state.mech_messages:
            with st.chat_message("assistant"):
                if st.session_state.lang == "tl":
                    st.markdown(
                        "**Mekaniko mode aktibo.** 🔧\n\n"
                        "- 🔎 Pagsusuri ng depekto — ilarawan ang sasakyan, reklamo, mga natuklasan\n"
                        "- 📟 Paghahanap ng DTC code\n"
                        "- 📐 Torque specs at mga clearance\n"
                        "- 🛠️ Mga pamamaraan ng pagkumpuni\n"
                        "- 🖼️ Pag-aanalisa ng larawan\n\n"
                        "**Halimbawa:** *2018 Vios, CVT kumikirot sa mababang bilis, madilim ang likido, DTC P0868*"
                    )
                else:
                    st.markdown(
                        "**Mechanic mode active.** 🔧\n\n"
                        "- 🔎 Fault diagnosis — describe vehicle, complaint, findings\n"
                        "- 📟 DTC code lookup\n- 📐 Torque specs & clearances\n"
                        "- 🛠️ Repair procedures\n- 🖼️ Photo analysis\n\n"
                        "**Example:** *2018 Vios, CVT shudders at low speed, fluid dark, DTC P0868*"
                    )

        for msg in st.session_state.mech_messages:
            with st.chat_message(msg["role"]):
                mtype = msg.get("type","text")
                if mtype == "image":
                    if "img_bytes" in msg: st.image(msg["img_bytes"], width=300)
                    st.caption(msg["content"])
                elif mtype == "mech_diagnosis":
                    st.markdown('<div style="background:#1e3a5f;color:#dbeafe;padding:8px 14px;'
                                'border-radius:6px;font-weight:bold;margin-bottom:8px;">🔬 Technical Diagnosis</div>',
                                unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    if msg.get("sources"): st.caption(f"📚 {', '.join(msg['sources'])}")
                elif mtype == "mech_question":
                    st.markdown('<div style="background:#fef3c7;color:#92400e;padding:8px 14px;'
                                'border-radius:6px;font-weight:bold;margin-bottom:8px;">📋 Additional info needed</div>',
                                unsafe_allow_html=True)
                    st.markdown(msg["content"])
                else:
                    st.markdown(msg["content"])
                    if msg.get("sources"): st.caption(f"📚 {', '.join(msg['sources'])}")

        mech_ph = ("Provide findings (DTC codes, symptoms, observations)..."
                   if st.session_state.mech_diag["mode"]=="awaiting_findings"
                   else "Describe the fault, ask for specs, or enter a DTC code...")
        if mech_input := st.chat_input(mech_ph):
            with st.spinner("Analysing..."):
                handle_mechanic_message(mech_input)
            st.rerun()

    # ── Appointments tab ───────────────────────────────────────────────────
    with tab_appts:
        render_appointments_panel()
