import streamlit as st
import sys
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "backend"))

from app.analysis.image_quality import assess_image_quality
from app.analysis.underbody import analyze_underbody_image
from app.analysis.dashboard_hook import detect_dashboard_warnings
from app.analysis.decision import decide

# ─── Inline audio analysis (scipy only — no librosa/numba needed) ─────────────
class _AudioFinding:
    def __init__(self, label: str, confidence: float, details: dict = None):
        self.label = label
        self.confidence = confidence
        self.details = details or {}

def _analyze_audio(path: str) -> tuple[list, float]:
    import soundfile as sf
    import numpy as np
    from scipy import signal as sp_signal

    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    duration = len(data) / sr

    if len(data) < sr * 2:
        return [_AudioFinding("unknown", 0.2, {"reason": "audio_too_short"})], duration

    if sr != 22050:
        target = int(len(data) * 22050 / sr)
        data = sp_signal.resample(data, target)
        sr = 22050

    chunk = sr // 2
    chunks = [data[i: i + chunk] for i in range(0, len(data) - chunk, chunk)]
    rms_vals = __import__("numpy").array([__import__("numpy").sqrt(__import__("numpy").mean(c ** 2)) for c in chunks])
    import numpy as np
    rms_mean = float(rms_vals.mean()) if len(rms_vals) else 0.0
    rms_var  = float(rms_vals.var())  if len(rms_vals) else 0.0
    zcr_mean = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)
    clip     = data[:sr * 5] if len(data) > sr * 5 else data
    spectrum = np.abs(np.fft.rfft(clip))
    freqs    = np.fft.rfftfreq(len(clip), 1.0 / sr)
    centroid_mean = float(np.sum(freqs * spectrum) / (spectrum.sum() + 1e-10))

    findings: list[_AudioFinding] = []
    if rms_mean > 0.01 and (rms_var / (rms_mean ** 2 + 1e-9)) > 0.8:
        findings.append(_AudioFinding("rough_or_unstable", 0.55, {"rms_mean": rms_mean, "rms_var": rms_var}))
    if centroid_mean > 3500 and zcr_mean > 0.10:
        findings.append(_AudioFinding("high_frequency_squeal_like", 0.45, {"centroid_mean": centroid_mean, "zcr_mean": zcr_mean}))
    if zcr_mean > 0.12 and centroid_mean < 4500:
        findings.append(_AudioFinding("ticking_like", 0.40, {"centroid_mean": centroid_mean, "zcr_mean": zcr_mean}))
    if not findings:
        findings.append(_AudioFinding("no_clear_anomaly_detected", 0.60, {"rms_mean": rms_mean, "zcr_mean": zcr_mean, "centroid_mean": centroid_mean}))
    return findings, duration

# ─── Translations ─────────────────────────────────────────────────────────────
TR = {
    "he": {
        "app_title":        "בדיקת רכב",
        "app_subtitle":     "הערכת רכב פרימיום",
        "app_subtitle_main":"הערכה ראשונית של הרכב",
        "email_label":      "כתובת אימייל",
        "password_label":   "קוד גישה",
        "enter_btn":        "כניסה",
        "email_hint":       "האימייל שלך משמש לגישה להיסטוריית הבדיקות שלך.",
        "invalid_code":     "קוד גישה שגוי.",
        "invalid_email":    "אנא הזן כתובת אימייל תקינה.",
        "new_check":        "＋  בדיקה חדשה",
        "past_checks":      "בדיקות קודמות",
        "sign_out":         "יציאה",
        "step_details":     "פרטים",
        "step_photos":      "תמונות",
        "step_audio":       "שמע",
        "step_result":      "תוצאה",
        "vehicle_details":  "פרטי הרכב",
        "manufacturer":     "יצרן",
        "manufacturer_ph":  "למשל: טויוטה",
        "model":            "דגם",
        "model_ph":         "למשל: קאמרי",
        "year":             "שנה",
        "odometer":         "קילומטראז' (ק״מ)",
        "trim":             "טריים (אופציונלי)",
        "trim_ph":          "למשל: Sport, SE",
        "nickname":         "כינוי (אופציונלי)",
        "nickname_ph":      "למשל: קאמרי אדומה",
        "continue_btn":     "המשך  ←",
        "back_btn":         "→  חזור",
        "vehicle_photos":   "תמונות הרכב",
        "photos_hint":      "העלה 4–10 תמונות: חיצוני (כל הצדדים), תא המנוע, פנים, לוח מחוונים. ודא תאורה טובה ומצלמה יציבה.",
        "underbody_title":  "תמונת תחתית הרכב — אופציונלי",
        "underbody_hint":   "תמונה של חלק התחתון של הרכב מסייעת לזיהוי דליפות אפשריות.",
        "photos_count":     "תמונות נבחרו",
        "photos_min_error": "אנא העלה לפחות 4 תמונות.",
        "photos_max_error": "מקסימום 10 תמונות מותר.",
        "engine_audio":     "שמע מנוע",
        "audio_hint":       "הקלט לפחות 10 שניות של המנוע פועל בסרלנטי. קרב את המיקרופון לתא המנוע לתוצאות הטובות ביותר.",
        "audio_missing":    "אנא העלה הקלטת שמע של המנוע.",
        "analyse_btn":      "נתח רכב  ←",
        "analysing":        "מנתח — זה עשוי לקחת רגע ...",
        "analysis_failed":  "הניתוח נכשל",
        "confidence_label": "רמת ביטחון",
        "audio_analysed":   "שמע: {:.1f} שניות נותחו",
        "findings_title":   "ממצאי הבדיקה",
        "next_steps_title": "צעדים מומלצים",
        "learn_more_title": "קרא עוד",
        "go":               "מתאים",
        "no_go":            "לא מתאים",
        "inconclusive":     "לא ברור",
        "high":             "גבוה",
        "medium":           "בינוני",
        "low":              "נמוך",
    },
    "en": {
        "app_title":        "UsedCar Check",
        "app_subtitle":     "Premium Vehicle Assessment",
        "app_subtitle_main":"Preliminary Vehicle Assessment",
        "email_label":      "Email Address",
        "password_label":   "Access Code",
        "enter_btn":        "Enter",
        "email_hint":       "Your email is used to access your check history.",
        "invalid_code":     "Invalid access code.",
        "invalid_email":    "Please enter a valid email address.",
        "new_check":        "＋  New Check",
        "past_checks":      "Past Checks",
        "sign_out":         "Sign Out",
        "step_details":     "Details",
        "step_photos":      "Photos",
        "step_audio":       "Audio",
        "step_result":      "Result",
        "vehicle_details":  "Vehicle Details",
        "manufacturer":     "Manufacturer",
        "manufacturer_ph":  "e.g. Toyota",
        "model":            "Model",
        "model_ph":         "e.g. Camry",
        "year":             "Year",
        "odometer":         "Odometer (km)",
        "trim":             "Trim (optional)",
        "trim_ph":          "e.g. Sport, SE",
        "nickname":         "Nickname (optional)",
        "nickname_ph":      "e.g. Red Camry",
        "continue_btn":     "Continue  →",
        "back_btn":         "←  Back",
        "vehicle_photos":   "Vehicle Photos",
        "photos_hint":      "Upload 4–10 photos: exterior (all sides), engine bay, interior, dashboard. Ensure good lighting and a steady hand.",
        "underbody_title":  "Underbody Photo — Optional",
        "underbody_hint":   "A photo of the underside of the vehicle helps detect possible fluid leaks.",
        "photos_count":     "photo(s) selected",
        "photos_min_error": "Please upload at least 4 photos.",
        "photos_max_error": "Maximum 10 photos allowed.",
        "engine_audio":     "Engine Audio",
        "audio_hint":       "Record at least 10 seconds of the engine running at idle. Hold the microphone near the engine bay for best results.",
        "audio_missing":    "Please upload an engine audio recording.",
        "analyse_btn":      "Analyse Vehicle  →",
        "analysing":        "Analysing — this may take a moment …",
        "analysis_failed":  "Analysis failed",
        "confidence_label": "Confidence",
        "audio_analysed":   "Audio: {:.1f}s analysed",
        "findings_title":   "Assessment Findings",
        "next_steps_title": "Recommended Next Steps",
        "learn_more_title": "Learn More",
        "go":               "GO",
        "no_go":            "NO GO",
        "inconclusive":     "INCONCLUSIVE",
        "high":             "HIGH",
        "medium":           "MEDIUM",
        "low":              "LOW",
    }
}

def t(key: str) -> str:
    return TR[st.session_state.get("lang", "he")].get(key, key)

# ─── Constants ────────────────────────────────────────────────────────────────
APP_PASSWORD = "VW2026"
DATA_DIR     = _ROOT / "data"
USERS_FILE   = DATA_DIR / "users.json"
CHECKS_DIR   = DATA_DIR / "checks"
DATA_DIR.mkdir(exist_ok=True)
CHECKS_DIR.mkdir(exist_ok=True)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UsedCar Check",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state init ───────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "email":         "",
    "step":          1,
    "car_details":   {},
    "photos":        [],
    "underbody":     None,
    "audio":         None,
    "result":        None,
    "lang":          "he",   # Hebrew default
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CSS (including RTL support) ──────────────────────────────────────────────
is_rtl = st.session_state.lang == "he"
rtl_css = "direction: rtl; text-align: right;" if is_rtl else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Jost:wght@300;400;500&display=swap');

:root {{
    --gold:      #C8A96A;
    --gold-dark: #8B7040;
    --bg:        #0C0C0C;
    --surface:   #141414;
    --elevated:  #1E1E1E;
    --border:    #2C2C2C;
    --text:      #F0EBE0;
    --muted:     #9A9080;
    --success:   #4A7A4A;
    --warning:   #C8960A;
    --danger:    #8A3535;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

html, body, .stApp {{
    background-color: var(--bg) !important;
    font-family: 'Jost', sans-serif;
    color: var(--text) !important;
    {rtl_css}
}}

section[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    {rtl_css}
}}

input, textarea, .stTextInput input, .stNumberInput input {{
    background-color: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    {"text-align: right !important;" if is_rtl else ""}
}}

.stButton > button {{
    background: linear-gradient(135deg, var(--gold-dark), var(--gold)) !important;
    color: #0C0C0C !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s ease !important;
}}
.stButton > button:hover {{ opacity: 0.82 !important; }}

/* Language flag buttons — small & unobtrusive */
.lang-btn > button {{
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50px !important;
    font-size: 1.1rem !important;
    padding: 0.1rem 0.5rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
}}
.lang-btn > button:hover {{ border-color: var(--gold-dark) !important; opacity: 1 !important; }}
.lang-btn-active > button {{
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}}

[data-testid="stFileUploader"] {{
    background-color: var(--surface) !important;
    border: 1px dashed var(--gold-dark) !important;
    border-radius: 3px !important;
}}

label, .stMarkdown p, p, li {{ color: var(--text) !important; }}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: var(--gold) !important; }}
hr {{ border-color: var(--border) !important; margin: 0.8rem 0 !important; }}
.stSpinner > div {{ border-top-color: var(--gold) !important; }}
[data-testid="stNotification"] {{ border-radius: 2px !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Data helpers ─────────────────────────────────────────────────────────────
def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {}

def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def get_or_create_user(email: str):
    users = load_users()
    if email not in users:
        users[email] = {"created_at": datetime.now().isoformat(), "checks": []}
        save_users(users)
    return users[email]

def save_check(email: str, result: dict) -> str:
    check_id = str(uuid.uuid4())[:8].upper()
    result["check_id"] = check_id
    result["email"]      = email
    result["created_at"] = datetime.now().isoformat()
    (CHECKS_DIR / f"{check_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    users = load_users()
    users.setdefault(email, {"created_at": datetime.now().isoformat(), "checks": []})
    users[email]["checks"].append(check_id)
    save_users(users)
    return check_id

def load_check(check_id: str) -> dict | None:
    f = CHECKS_DIR / f"{check_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def get_user_checks(email: str) -> list[dict]:
    users = load_users()
    ids = users.get(email, {}).get("checks", [])
    return [c for cid in reversed(ids) if (c := load_check(cid))]

# ─── UI helpers ───────────────────────────────────────────────────────────────
def gold_divider():
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);margin:1rem 0;'></div>",
        unsafe_allow_html=True,
    )

def luxury_header(title: str, subtitle: str = ""):
    align = "right" if is_rtl else "center"
    st.markdown(f"""
    <div style='text-align:{align};padding:1.8rem 0 0.8rem;'>
        <div style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:2.6rem;
                    letter-spacing:0.18em;color:var(--gold);text-transform:uppercase;'>
            {title}
        </div>
        {"<div style='font-size:0.7rem;letter-spacing:0.3em;color:var(--muted);text-transform:uppercase;margin-top:0.3rem;'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def verdict_meta(rec: str) -> tuple[str, str, str]:
    label = t(rec) if rec in ("go", "no_go", "inconclusive") else rec.upper()
    return {
        "go":           (label, "#4A7A4A", "rgba(74,122,74,0.12)"),
        "no_go":        (label, "#B04040", "rgba(176,64,64,0.12)"),
        "inconclusive": (label, "#C8A96A", "rgba(200,169,106,0.12)"),
    }.get(rec, (label, "#9A9080", "rgba(154,144,128,0.12)"))

def badge(text: str, level: str):
    colors = {
        "high":   ("#4A7A4A", "rgba(74,122,74,0.15)"),
        "medium": ("#C8A96A", "rgba(200,169,106,0.15)"),
        "low":    ("#B04040", "rgba(176,64,64,0.15)"),
    }
    fg, bg = colors.get(level, ("#9A9080", "rgba(154,144,128,0.15)"))
    label = t(level) if level in ("high", "medium", "low") else text
    st.markdown(
        f"<span style='background:{bg};color:{fg};border:1px solid {fg};padding:0.15rem 0.6rem;"
        f"border-radius:2px;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;'>{label}</span>",
        unsafe_allow_html=True,
    )

def section_label(key: str):
    st.markdown(
        f"<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem;'>{t(key)}</p>",
        unsafe_allow_html=True,
    )

# ─── Language toggle ──────────────────────────────────────────────────────────
def lang_toggle():
    """Render 🇮🇱 / 🇺🇸 flag buttons."""
    col_he, col_en, _ = st.columns([1, 1, 8])
    with col_he:
        active = "lang-btn-active" if st.session_state.lang == "he" else "lang-btn"
        st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
        if st.button("🇮🇱", key="lang_he"):
            st.session_state.lang = "he"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_en:
        active = "lang-btn-active" if st.session_state.lang == "en" else "lang-btn"
        st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
        if st.button("🇺🇸", key="lang_en"):
            st.session_state.lang = "en"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ─── Analysis runner ──────────────────────────────────────────────────────────
def run_analysis(car_details, photo_files, audio_file, underbody_file=None) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        photo_paths = []
        for i, f in enumerate(photo_files):
            ext = Path(f.name).suffix or ".jpg"
            p = tmp / f"photo_{i}{ext}"
            p.write_bytes(f.getvalue())
            photo_paths.append(str(p))

        underbody_path = None
        if underbody_file:
            ext = Path(underbody_file.name).suffix or ".jpg"
            up = tmp / f"underbody{ext}"
            up.write_bytes(underbody_file.getvalue())
            underbody_path = str(up)

        ap = tmp / f"audio{Path(audio_file.name).suffix or '.wav'}"
        ap.write_bytes(audio_file.getvalue())

        photo_qualities   = [assess_image_quality(p) for p in photo_paths]
        audio_findings, audio_dur = _analyze_audio(str(ap))
        dashboard_findings = detect_dashboard_warnings(photo_paths)

        underbody_findings = []
        if underbody_path:
            underbody_findings = [
                {"label": f.label, "confidence": f.confidence, "details": f.details}
                for f in analyze_underbody_image(underbody_path)
            ]

        decision = decide(
            photo_qualities=photo_qualities,
            audio_findings=audio_findings,
            dashboard_findings=[{"label": f.label, "confidence": f.confidence} for f in dashboard_findings],
            underbody_findings=underbody_findings,
            driven_km=car_details.get("odometer"),
        )
        return decision, audio_dur

# ─── Result renderer ──────────────────────────────────────────────────────────
def render_result(result: dict):
    rec   = result.get("recommendation", "inconclusive")
    conf  = result.get("confidence", "low")
    label, color, bg = verdict_meta(rec)
    car_label = result.get("car_label", "")
    date      = result.get("created_at", "")[:10]
    align     = "right" if is_rtl else "center"

    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color};border-radius:4px;
                padding:2rem;text-align:{align};margin:1rem 0;'>
        <div style='font-family:Cormorant Garamond,serif;font-size:3.2rem;
                    font-weight:600;letter-spacing:0.22em;color:{color};'>{label}</div>
        <div style='font-size:0.75rem;letter-spacing:0.2em;color:var(--muted);
                    margin-top:0.4rem;text-transform:uppercase;'>{car_label}</div>
        <div style='font-size:0.68rem;color:var(--muted);margin-top:0.2rem;'>{date}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<span style='font-size:0.7rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;'>{t('confidence_label')}</span>", unsafe_allow_html=True)
        badge(conf.upper(), conf)
    with col2:
        dur = result.get("audio_duration_seconds")
        if dur:
            st.markdown(f"<span style='font-size:0.7rem;color:var(--muted);'>{t('audio_analysed').format(dur)}</span>", unsafe_allow_html=True)

    gold_divider()

    reasons = result.get("top_reasons", [])
    if reasons:
        section_label("findings_title")
        sev_colors = {"high": "#B04040", "medium": "#C8A96A", "low": "#4A7A4A"}
        for r in reasons:
            sev = r.get("severity", "low")
            bc  = sev_colors.get(sev, "#9A9080")
            st.markdown(f"""
            <div style='background:var(--elevated);border-left:3px solid {bc};
                        padding:0.75rem 1rem;margin:0.4rem 0;border-radius:0 3px 3px 0;
                        {rtl_css}'>
                <span style='font-size:0.88rem;'>{r.get("title","")}</span>
            </div>
            """, unsafe_allow_html=True)

    steps = result.get("next_steps", [])
    if steps:
        gold_divider()
        section_label("next_steps_title")
        for s in steps:
            st.markdown(f"<p style='font-size:0.85rem;color:var(--text);'>{'←' if is_rtl else '→'} &nbsp;{s.get('text','')}</p>", unsafe_allow_html=True)

    edu = result.get("education", [])
    if edu:
        gold_divider()
        section_label("learn_more_title")
        for e in edu:
            st.markdown(f"<p style='font-size:0.82rem;color:var(--muted);'>◦ &nbsp;{e.get('title','')}</p>", unsafe_allow_html=True)

# ─── Step indicator ───────────────────────────────────────────────────────────
def step_indicator(current: int):
    labels = [t("step_details"), t("step_photos"), t("step_audio"), t("step_result")]
    cols   = st.columns(4)
    for i, (col, name) in enumerate(zip(cols, labels), 1):
        with col:
            active = i == current
            done   = i < current
            dot_color = "var(--gold)" if active else ("#4A7A4A" if done else "var(--border)")
            txt_color = "var(--gold)" if active else ("#4A7A4A" if done else "var(--muted)")
            st.markdown(
                f"<div style='text-align:center;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:{txt_color};'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{dot_color};margin:0 auto 0.3rem;'></div>{name}</div>",
                unsafe_allow_html=True,
            )
    gold_divider()

# ─── Login screen ─────────────────────────────────────────────────────────────
def login_screen():
    lang_toggle()
    luxury_header(t("app_title"), t("app_subtitle"))
    gold_divider()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        email    = st.text_input(t("email_label"),    placeholder="your@email.com",    key="login_email")
        password = st.text_input(t("password_label"), type="password", placeholder="· · · · · · · ·", key="login_pass")
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        if st.button(t("enter_btn"), use_container_width=True):
            if password != APP_PASSWORD:
                st.error(t("invalid_code"))
            elif not email or "@" not in email:
                st.error(t("invalid_email"))
            else:
                get_or_create_user(email.lower().strip())
                st.session_state.authenticated = True
                st.session_state.email         = email.lower().strip()
                st.session_state.step          = 1
                st.session_state.result        = None
                st.rerun()

        st.markdown(
            f"<p style='font-size:0.68rem;color:var(--muted);text-align:center;margin-top:1rem;letter-spacing:0.05em;'>{t('email_hint')}</p>",
            unsafe_allow_html=True,
        )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Language toggle at top of sidebar
        col_he, col_en, _ = st.columns([1, 1, 4])
        with col_he:
            active = "lang-btn-active" if st.session_state.lang == "he" else "lang-btn"
            st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
            if st.button("🇮🇱", key="sb_lang_he"):
                st.session_state.lang = "he"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with col_en:
            active = "lang-btn-active" if st.session_state.lang == "en" else "lang-btn"
            st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
            if st.button("🇺🇸", key="sb_lang_en"):
                st.session_state.lang = "en"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='padding:0.8rem 0 0.4rem;{rtl_css}'>
            <div style='font-family:Cormorant Garamond,serif;font-size:1.5rem;
                        color:var(--gold);letter-spacing:0.15em;text-transform:uppercase;'>
                {t("app_title")}
            </div>
            <div style='font-size:0.62rem;color:var(--muted);letter-spacing:0.25em;
                        text-transform:uppercase;margin-top:0.15rem;'>
                {t("app_subtitle")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-size:0.72rem;color:var(--muted);margin:0.2rem 0 0.6rem;{rtl_css}'>{st.session_state.email}</p>",
            unsafe_allow_html=True,
        )
        gold_divider()

        if st.button(t("new_check"), use_container_width=True):
            st.session_state.step       = 1
            st.session_state.car_details = {}
            st.session_state.photos     = []
            st.session_state.underbody  = None
            st.session_state.audio      = None
            st.session_state.result     = None
            st.rerun()

        past = get_user_checks(st.session_state.email)
        if past:
            st.markdown(
                f"<p style='font-size:0.65rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;margin-top:1.2rem;{rtl_css}'>{t('past_checks')}</p>",
                unsafe_allow_html=True,
            )
            for check in past[:15]:
                rec = check.get("recommendation", "?")
                v_label, v_color, _ = verdict_meta(rec)
                car  = check.get("car_label", check.get("check_id", ""))
                date = check.get("created_at", "")[:10]
                display = f"{car}  ·  {date}" if car else date
                st.markdown(
                    f"<div style='font-size:0.7rem;color:{v_color};letter-spacing:0.05em;margin-bottom:0.1rem;{rtl_css}'>{v_label}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(display, key=f"past_{check['check_id']}", use_container_width=True):
                    st.session_state.result = check
                    st.session_state.step   = 4
                    st.rerun()

        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
        if st.button(t("sign_out"), use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ─── Step 1 — Vehicle Details ─────────────────────────────────────────────────
def step_vehicle_details():
    section_label("vehicle_details")
    d = st.session_state.car_details

    col1, col2 = st.columns(2)
    with col1:
        manufacturer = st.text_input(t("manufacturer"), value=d.get("manufacturer", ""), placeholder=t("manufacturer_ph"))
        year         = st.number_input(t("year"), min_value=1990, max_value=2026, step=1, value=int(d.get("year", 2020)))
        odometer     = st.number_input(t("odometer"), min_value=0, max_value=2_000_000, step=1000, value=int(d.get("odometer", 80_000)))
    with col2:
        model_name = st.text_input(t("model"),    value=d.get("model_name", ""), placeholder=t("model_ph"))
        trim       = st.text_input(t("trim"),     value=d.get("trim", ""),       placeholder=t("trim_ph"))
        nickname   = st.text_input(t("nickname"), value=d.get("nickname", ""),   placeholder=t("nickname_ph"))

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button(t("continue_btn")):
        st.session_state.car_details = {
            "manufacturer": manufacturer.strip(),
            "model_name":   model_name.strip(),
            "year":         int(year),
            "trim":         trim.strip(),
            "odometer":     int(odometer),
            "nickname":     nickname.strip(),
        }
        st.session_state.step = 2
        st.rerun()

# ─── Step 2 — Photos ──────────────────────────────────────────────────────────
def step_photos():
    section_label("vehicle_photos")
    st.markdown(f"<p style='font-size:0.83rem;color:var(--muted);'>{t('photos_hint')}</p>", unsafe_allow_html=True)

    photos = st.file_uploader(
        t("vehicle_photos"), type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    gold_divider()
    section_label("underbody_title")
    st.markdown(f"<p style='font-size:0.83rem;color:var(--muted);'>{t('underbody_hint')}</p>", unsafe_allow_html=True)
    underbody = st.file_uploader(
        t("underbody_title"), type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="underbody_upload",
    )

    if photos:
        color = "#4A7A4A" if 4 <= len(photos) <= 10 else "#B04040"
        st.markdown(f"<p style='font-size:0.78rem;color:{color};margin-top:0.4rem;'>{len(photos)} {t('photos_count')}</p>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn")):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button(t("continue_btn"), key="photos_continue"):
            if not photos or len(photos) < 4:
                st.error(t("photos_min_error"))
            elif len(photos) > 10:
                st.error(t("photos_max_error"))
            else:
                st.session_state.photos   = photos
                st.session_state.underbody = underbody
                st.session_state.step     = 3
                st.rerun()

# ─── Step 3 — Audio ───────────────────────────────────────────────────────────
def step_audio():
    section_label("engine_audio")
    st.markdown(f"<p style='font-size:0.83rem;color:var(--muted);'>{t('audio_hint')}</p>", unsafe_allow_html=True)

    audio = st.file_uploader(
        t("engine_audio"), type=["mp3", "wav", "m4a", "ogg", "aac", "flac"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn"), key="audio_back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button(t("analyse_btn")):
            if not audio:
                st.error(t("audio_missing"))
            else:
                d = st.session_state.car_details
                car_label = f"{d.get('year','')} {d.get('manufacturer','')} {d.get('model_name','')}".strip()
                with st.spinner(t("analysing")):
                    try:
                        decision, audio_dur = run_analysis(d, st.session_state.photos, audio, st.session_state.underbody)
                        result = {
                            "recommendation":         decision.recommendation,
                            "confidence":             decision.confidence,
                            "top_reasons":            decision.top_reasons,
                            "breakdown":              decision.breakdown,
                            "education":              decision.education,
                            "next_steps":             decision.next_steps,
                            "audio_duration_seconds": audio_dur,
                            "car_details":            d,
                            "car_label":              car_label,
                        }
                        check_id = save_check(st.session_state.email, result)
                        result["check_id"]      = check_id
                        st.session_state.result = result
                        st.session_state.step   = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"{t('analysis_failed')}: {e}")

# ─── Main app ─────────────────────────────────────────────────────────────────
def main_app():
    render_sidebar()
    luxury_header(t("app_title"), t("app_subtitle_main"))
    gold_divider()

    step = st.session_state.step
    step_indicator(step)

    if step == 1:
        step_vehicle_details()
    elif step == 2:
        step_photos()
    elif step == 3:
        step_audio()
    elif step == 4 and st.session_state.result:
        render_result(st.session_state.result)

# ─── Entry point ──────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    login_screen()
else:
    main_app()
