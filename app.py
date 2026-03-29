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
    rms_vals = np.array([np.sqrt(np.mean(c ** 2)) for c in chunks])
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
        "app_subtitle":     "קנה? אל תקנה? — תגלה תוך דקות 🚗",
        "app_subtitle_main":"הערכה מהירה מבוססת AI",
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
        "model":            "דגם",
        "select_make":      "— בחר יצרן —",
        "select_model":     "— בחר דגם —",
        "year":             "שנה",
        "odometer":         "קילומטראז' (ק״מ)",
        "trim":             "גרסה / ורסיה (אופציונלי)",
        "trim_ph":          "למשל: Sport, SE, Luxury",
        "prev_owners":      "יד הרכב",
        "prev_owners_opts": ["יד ראשונה", "יד שנייה", "יד שלישית", "יד רביעית+"],
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
        "feat1":            "ניתוח תמונות חכם",
        "feat1_desc":       "זיהוי בעיות חיצוניות ופנימיות",
        "feat2":            "ניתוח קול מנוע",
        "feat2_desc":       "זיהוי רעשים חריגים בזמן סרלנטי",
        "feat3":            "פסיקה מיידית — קנה / אל תקנה",
        "feat3_desc":       "GO או NO-GO בהתבסס על כל הנתונים",
        "own_single_good":  "בעלים יחיד במשך {age} שנים — סימן חיובי לתחזוקה טובה",
        "own_red_flag":     "{owners} בעלים ב-{age} שנים בלבד — החלפות תכופות: דגל אדום",
        "own_concern":      "ממוצע {avg:.1f} שנים לבעלים — קצר מהמצופה",
        "own_stable":       "היסטוריית בעלות יציבה — ממוצע {avg:.1f} שנים לבעלים",
    },
    "en": {
        "app_title":        "UsedCar Check",
        "app_subtitle":     "GO or NO-GO — Know Before You Buy 🚗",
        "app_subtitle_main":"AI-Powered Quick Assessment",
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
        "model":            "Model",
        "select_make":      "— Select Make —",
        "select_model":     "— Select Model —",
        "year":             "Year",
        "odometer":         "Odometer (km)",
        "trim":             "Variant / Version (optional)",
        "trim_ph":          "e.g. Sport, SE, Luxury",
        "prev_owners":      "Previous Owners",
        "prev_owners_opts": ["1st Owner", "2nd Owner", "3rd Owner", "4th Owner+"],
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
        "feat1":            "Smart Photo Analysis",
        "feat1_desc":       "Detect exterior & interior issues",
        "feat2":            "Engine Sound Analysis",
        "feat2_desc":       "Identify abnormal sounds at idle",
        "feat3":            "Instant GO / NO-GO Verdict",
        "feat3_desc":       "Clear decision based on all data points",
        "own_single_good":  "Single owner for {age} years — positive sign of good maintenance",
        "own_red_flag":     "{owners} owners in just {age} years — frequent changes are a red flag",
        "own_concern":      "Average {avg:.1f} years per owner — shorter than expected",
        "own_stable":       "Stable ownership history — average {avg:.1f} years per owner",
    }
}

def t(key: str) -> str:
    return TR[st.session_state.get("lang", "he")].get(key, key)

# ─── Car makes & models ───────────────────────────────────────────────────────
CAR_MAKES_MODELS: dict[str, list[str]] = {
    "Toyota":        ["Corolla", "Camry", "RAV4", "Yaris", "Prius", "Land Cruiser", "Hilux", "Auris", "C-HR", "Verso", "Avensis", "Fortuner"],
    "Volkswagen":    ["Golf", "Polo", "Passat", "Tiguan", "T-Roc", "Touareg", "Arteon", "ID.3", "ID.4", "Caddy", "Touran"],
    "Hyundai":       ["Tucson", "Santa Fe", "i10", "i20", "i30", "Kona", "Ioniq 5", "Ioniq 6", "Elantra", "ix35", "Creta"],
    "Kia":           ["Sportage", "Sorento", "Rio", "Ceed", "Stonic", "Niro", "EV6", "Carnival", "Picanto", "Telluride"],
    "BMW":           ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "X1", "X2", "X3", "X5", "X6", "iX"],
    "Mercedes-Benz": ["A-Class", "B-Class", "C-Class", "E-Class", "S-Class", "CLA", "GLA", "GLC", "GLE", "GLS", "EQC"],
    "Audi":          ["A1", "A3", "A4", "A6", "A8", "Q2", "Q3", "Q5", "Q7", "Q8", "TT", "e-tron"],
    "Honda":         ["Civic", "Accord", "CR-V", "Jazz", "HR-V", "ZR-V", "Pilot", "Fit"],
    "Mazda":         ["Mazda2", "Mazda3", "Mazda6", "CX-3", "CX-5", "CX-60", "MX-5", "CX-30"],
    "Nissan":        ["Qashqai", "X-Trail", "Micra", "Juke", "Leaf", "Ariya", "Navara", "Murano", "Pathfinder"],
    "Ford":          ["Focus", "Fiesta", "Mondeo", "Kuga", "Puma", "EcoSport", "Mustang", "Explorer", "Edge", "Bronco"],
    "Opel":          ["Astra", "Corsa", "Insignia", "Crossland", "Grandland", "Mokka", "Zafira"],
    "Škoda":         ["Octavia", "Fabia", "Superb", "Kodiaq", "Karoq", "Kamiq", "Enyaq"],
    "Seat":          ["Ibiza", "Leon", "Ateca", "Arona", "Tarraco", "Mii"],
    "Renault":       ["Clio", "Megane", "Kadjar", "Duster", "Captur", "Arkana", "Austral", "Laguna", "Zoe"],
    "Peugeot":       ["208", "308", "508", "2008", "3008", "5008", "Landtrek"],
    "Citroën":       ["C3", "C4", "C5 X", "Berlingo", "C3 Aircross", "C5 Aircross"],
    "Fiat":          ["Punto", "500", "Tipo", "Panda", "Bravo", "500X", "Doblo"],
    "Chevrolet":     ["Spark", "Cruze", "Malibu", "Equinox", "Trax", "Captiva", "Blazer"],
    "Mitsubishi":    ["Outlander", "Eclipse Cross", "ASX", "Lancer", "Colt", "L200", "Pajero"],
    "Suzuki":        ["Swift", "Vitara", "SX4", "Jimny", "Baleno", "Ignis", "S-Cross"],
    "Subaru":        ["Impreza", "Forester", "Outback", "XV", "Legacy", "WRX", "BRZ"],
    "Jeep":          ["Wrangler", "Cherokee", "Grand Cherokee", "Renegade", "Compass", "Gladiator"],
    "Volvo":         ["XC40", "XC60", "XC90", "V40", "V60", "S60", "S90", "C40"],
    "Lexus":         ["IS", "ES", "LS", "UX", "NX", "RX", "GX", "LX", "CT"],
    "Land Rover":    ["Defender", "Discovery", "Discovery Sport", "Range Rover", "Range Rover Sport", "Range Rover Evoque", "Freelander"],
    "Porsche":       ["911", "Cayenne", "Macan", "Panamera", "Taycan", "Cayman", "Boxster"],
    "Tesla":         ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
    "Dacia":         ["Sandero", "Logan", "Duster", "Spring", "Jogger"],
    "Alfa Romeo":    ["Giulia", "Stelvio", "Giulietta", "MiTo", "Tonale"],
    "Mini":          ["Cooper", "Countryman", "Clubman", "Paceman", "Convertible", "Electric"],
    "Isuzu":         ["D-Max", "MU-X", "Trooper"],
    "SsangYong":     ["Tivoli", "Korando", "Rexton", "Musso"],
    "Skoda":         ["Octavia", "Fabia", "Superb", "Kodiaq", "Karoq", "Kamiq"],
    "Other":         ["Other"],
}
MAKES_LIST = [""] + sorted(k for k in CAR_MAKES_MODELS if k != "Other") + ["Other"]

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
    "lang":          "he",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CSS ──────────────────────────────────────────────────────────────────────
is_rtl  = st.session_state.lang == "he"
rtl_css = "direction:rtl;text-align:right;" if is_rtl else ""

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

/* Remove default Streamlit padding so hero can be full-width */
.block-container {{ padding-top: 1rem !important; }}

input, textarea, .stTextInput input, .stNumberInput input {{
    background-color: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    {"text-align:right !important;" if is_rtl else ""}
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
    font-size: 0.88rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s ease !important;
}}
.stButton > button:hover {{ opacity: 0.82 !important; }}

.lang-btn > button {{
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50px !important;
    font-size: 1.1rem !important;
    padding: 0.1rem 0.45rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
}}
.lang-btn > button:hover {{ border-color: var(--gold-dark) !important; opacity: 1 !important; }}
.lang-btn-active > button {{
    border-color: var(--gold) !important;
    color: var(--gold) !important;
    background: rgba(200,169,106,0.08) !important;
}}

[data-testid="stFileUploader"] {{
    background-color: var(--surface) !important;
    border: 1px dashed var(--gold-dark) !important;
    border-radius: 3px !important;
}}

label, .stMarkdown p, p, li {{ color: var(--text) !important; }}
hr {{ border-color: var(--border) !important; margin: 0.8rem 0 !important; }}
.stSpinner > div {{ border-top-color: var(--gold) !important; }}

/* Feature cards on login */
.feat-card {{
    background: rgba(200,169,106,0.06);
    border: 1px solid rgba(200,169,106,0.2);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    {rtl_css}
}}

/* Step icon circle */
.step-icon {{
    width: 42px; height: 42px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.4rem;
    font-size: 1.2rem;
}}
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

def section_label(key: str):
    st.markdown(
        f"<p style='font-size:0.82rem;color:var(--muted);letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;{rtl_css}'>{t(key)}</p>",
        unsafe_allow_html=True,
    )

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
    label = t(level) if level in ("high","medium","low") else text
    st.markdown(
        f"<span style='background:{bg};color:{fg};border:1px solid {fg};padding:0.2rem 0.8rem;"
        f"border-radius:2px;font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;'>{label}</span>",
        unsafe_allow_html=True,
    )

# ─── Language toggle ──────────────────────────────────────────────────────────
def lang_toggle(key_prefix=""):
    col_he, col_en, _ = st.columns([1, 1, 10])
    with col_he:
        active = "lang-btn-active" if st.session_state.lang == "he" else "lang-btn"
        st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
        if st.button("🇮🇱", key=f"{key_prefix}lang_he"):
            st.session_state.lang = "he"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_en:
        active = "lang-btn-active" if st.session_state.lang == "en" else "lang-btn"
        st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
        if st.button("🇺🇸", key=f"{key_prefix}lang_en"):
            st.session_state.lang = "en"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ─── Ownership analysis ───────────────────────────────────────────────────────
def _analyze_ownership(num_owners: int, year: int) -> list[dict]:
    """Return reason dicts based on ownership history heuristics."""
    from datetime import date as _date
    car_age = max(1, _date.today().year - year)
    avg_yrs = car_age / max(1, num_owners)
    lang    = st.session_state.get("lang", "he")
    reasons = []

    if num_owners == 1 and car_age >= 3:
        reasons.append({
            "severity": "low",
            "title": TR[lang]["own_single_good"].format(age=car_age),
            "_positive": True,
        })
    elif num_owners >= 3 and car_age <= 4:
        reasons.append({
            "severity": "high",
            "title": TR[lang]["own_red_flag"].format(owners=num_owners, age=car_age),
        })
    elif avg_yrs < 1.5:
        reasons.append({
            "severity": "medium",
            "title": TR[lang]["own_concern"].format(avg=avg_yrs),
        })
    elif avg_yrs >= 3 and num_owners > 1:
        reasons.append({
            "severity": "low",
            "title": TR[lang]["own_stable"].format(avg=avg_yrs),
            "_positive": True,
        })

    return reasons

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

        # ── Ownership analysis — inject into decision ─────────────────────────
        num_owners = int(car_details.get("prev_owners", 1))
        year       = int(car_details.get("year", 2015))
        own_reasons = _analyze_ownership(num_owners, year)
        if own_reasons:
            # High-severity ownership finding overrides a "go" → "inconclusive"
            has_high = any(r.get("severity") == "high" for r in own_reasons)
            if has_high and decision.recommendation == "go":
                decision.recommendation = "inconclusive"
            # Prepend ownership reasons so they appear first in the result card
            decision.top_reasons = own_reasons + decision.top_reasons

        return decision, audio_dur

# ─── Result renderer ──────────────────────────────────────────────────────────
def render_result(result: dict):
    rec   = result.get("recommendation", "inconclusive")
    conf  = result.get("confidence", "low")
    label, color, bg = verdict_meta(rec)
    car_label = result.get("car_label", "")
    date      = result.get("created_at", "")[:10]
    align     = "right" if is_rtl else "center"

    verdict_icons = {"go": "✓", "no_go": "✕", "inconclusive": "?"}
    icon = verdict_icons.get(rec, "◈")

    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color};border-radius:6px;
                padding:2.5rem 2rem;text-align:{align};margin:1rem 0;
                box-shadow:0 0 40px {color}22;'>
        <div style='font-size:2.5rem;color:{color};margin-bottom:0.5rem;'>{icon}</div>
        <div style='font-family:Cormorant Garamond,serif;font-size:3.5rem;
                    font-weight:600;letter-spacing:0.22em;color:{color};
                    line-height:1;'>{label}</div>
        <div style='height:1px;width:60px;background:{color};margin:1rem {"0 1rem auto" if not is_rtl else "0 auto 1rem 0"};opacity:0.6;'></div>
        <div style='font-size:0.9rem;letter-spacing:0.2em;color:var(--muted);
                    text-transform:uppercase;'>{car_label}</div>
        <div style='font-size:0.8rem;color:var(--muted);margin-top:0.3rem;'>{date}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<span style='font-size:0.85rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;'>{t('confidence_label')}</span>", unsafe_allow_html=True)
        badge(conf.upper(), conf)
    with col2:
        dur = result.get("audio_duration_seconds")
        if dur:
            st.markdown(f"<span style='font-size:0.85rem;color:var(--muted);'>🎙 {t('audio_analysed').format(dur)}</span>", unsafe_allow_html=True)

    gold_divider()

    reasons = result.get("top_reasons", [])
    if reasons:
        section_label("findings_title")
        sev_colors = {"high": "#B04040", "medium": "#C8A96A", "low": "#4A7A4A"}
        sev_icons  = {"high": "⚠", "medium": "◉", "low": "◎"}
        for r in reasons:
            sev = r.get("severity", "low")
            bc  = sev_colors.get(sev, "#9A9080")
            ic  = sev_icons.get(sev, "◦")
            st.markdown(f"""
            <div style='background:var(--elevated);border-left:3px solid {bc};
                        padding:0.75rem 1rem;margin:0.4rem 0;border-radius:0 4px 4px 0;{rtl_css}'>
                <span style='color:{bc};margin-{"left" if is_rtl else "right"}:0.5rem;'>{ic}</span>
                <span style='font-size:0.95rem;'>{r.get("title","")}</span>
            </div>
            """, unsafe_allow_html=True)

    steps = result.get("next_steps", [])
    if steps:
        gold_divider()
        section_label("next_steps_title")
        for i, s in enumerate(steps, 1):
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.75rem;margin:0.5rem 0;{rtl_css}'>
                <div style='min-width:24px;height:24px;border-radius:50%;background:rgba(200,169,106,0.15);
                            border:1px solid var(--gold-dark);display:flex;align-items:center;
                            justify-content:center;font-size:0.65rem;color:var(--gold);flex-shrink:0;'>{i}</div>
                <div style='font-size:0.95rem;color:var(--text);padding-top:0.2rem;'>{s.get("text","")}</div>
            </div>
            """, unsafe_allow_html=True)

    edu = result.get("education", [])
    if edu:
        gold_divider()
        section_label("learn_more_title")
        for e in edu:
            st.markdown(f"<p style='font-size:0.82rem;color:var(--muted);{rtl_css}'>◦ &nbsp;{e.get('title','')}</p>", unsafe_allow_html=True)

# ─── Step indicator ───────────────────────────────────────────────────────────
STEP_ICONS = ["🚗", "📸", "🎙", "✓"]

def step_indicator(current: int):
    labels = [t("step_details"), t("step_photos"), t("step_audio"), t("step_result")]
    cols   = st.columns(4)
    for i, (col, name, icon) in enumerate(zip(cols, labels, STEP_ICONS), 1):
        with col:
            active = i == current
            done   = i < current
            bg_c   = "rgba(200,169,106,0.15)" if active else ("rgba(74,122,74,0.1)" if done else "rgba(44,44,44,0.4)")
            border = "var(--gold)" if active else ("#4A7A4A" if done else "var(--border)")
            txt    = "var(--gold)" if active else ("#4A7A4A" if done else "var(--muted)")
            st.markdown(f"""
            <div style='text-align:center;'>
                <div class='step-icon' style='background:{bg_c};border:1px solid {border};'>
                    <span style='color:{border};'>{icon}</span>
                </div>
                <div style='font-size:0.8rem;letter-spacing:0.08em;text-transform:uppercase;color:{txt};'>{name}</div>
            </div>
            """, unsafe_allow_html=True)
    gold_divider()

# ─── Login screen ─────────────────────────────────────────────────────────────
def login_screen():
    # ── Hero banner with car image (flags embedded inside) ───────────────────
    st.markdown(f"""
    <div style="
        background:
            linear-gradient(to bottom, rgba(12,12,12,0.2) 0%, rgba(12,12,12,0.7) 55%, rgba(12,12,12,1) 100%),
            url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1600&q=80')
            center 40% / cover no-repeat;
        border-radius: 8px;
        padding: 5rem 2rem 5.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
        position: relative;
    ">
        <div style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:4rem;
                    letter-spacing:0.22em;color:#C8A96A;text-transform:uppercase;
                    text-shadow:0 2px 30px rgba(0,0,0,0.9);'>
            {t("app_title")}
        </div>
        <div style='height:1px;width:80px;background:linear-gradient(90deg,transparent,#C8A96A,transparent);
                    margin:1.2rem auto;'></div>
        <div style='font-size:1.05rem;letter-spacing:0.1em;color:rgba(240,235,224,0.75);'>
            {t("app_subtitle")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: features left, login right ────────────────────────
    col_feat, col_gap, col_form = st.columns([5, 1, 4])

    with col_feat:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        for feat_key, desc_key, icon in [
            ("feat1", "feat1_desc", "📸"),
            ("feat2", "feat2_desc", "🎙"),
            ("feat3", "feat3_desc", "✓"),
        ]:
            st.markdown(f"""
            <div class='feat-card'>
                <div style='display:flex;align-items:center;gap:0.9rem;{rtl_css}'>
                    <div style='font-size:1.6rem;'>{icon}</div>
                    <div>
                        <div style='font-size:1rem;color:var(--gold);font-weight:500;'>{t(feat_key)}</div>
                        <div style='font-size:0.88rem;color:var(--muted);margin-top:0.2rem;'>{t(desc_key)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # Language flags — inside form column, no separate rectangle
        fc1, fc2, _ = st.columns([1, 1, 5])
        with fc1:
            active = "lang-btn-active" if st.session_state.lang == "he" else "lang-btn"
            st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
            if st.button("🇮🇱", key="login_lang_he"):
                st.session_state.lang = "he"; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with fc2:
            active = "lang-btn-active" if st.session_state.lang == "en" else "lang-btn"
            st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
            if st.button("🇺🇸", key="login_lang_en"):
                st.session_state.lang = "en"; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        email    = st.text_input(t("email_label"),    placeholder="your@email.com",     key="login_email")
        password = st.text_input(t("password_label"), type="password",
                                  placeholder="· · · · · · · ·", key="login_pass")
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

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
            f"<p style='font-size:0.8rem;color:var(--muted);text-align:center;margin-top:0.8rem;'>{t('email_hint')}</p>",
            unsafe_allow_html=True,
        )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
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
            <div style='font-size:0.78rem;color:var(--muted);letter-spacing:0.08em;
                        margin-top:0.2rem;'>{t("app_subtitle")}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:0.85rem;color:var(--muted);margin:0.2rem 0 0.6rem;{rtl_css}'>{st.session_state.email}</p>", unsafe_allow_html=True)
        gold_divider()

        if st.button(t("new_check"), use_container_width=True):
            st.session_state.step        = 1
            st.session_state.car_details = {}
            st.session_state.photos      = []
            st.session_state.underbody   = None
            st.session_state.audio       = None
            st.session_state.result      = None
            st.rerun()

        past = get_user_checks(st.session_state.email)
        if past:
            st.markdown(f"<p style='font-size:0.8rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;margin-top:1.2rem;{rtl_css}'>{t('past_checks')}</p>", unsafe_allow_html=True)
            for check in past[:15]:
                rec = check.get("recommendation", "?")
                v_label, v_color, _ = verdict_meta(rec)
                car  = check.get("car_label", check.get("check_id", ""))
                date = check.get("created_at", "")[:10]
                display = f"{car}  ·  {date}" if car else date
                st.markdown(f"<div style='font-size:0.82rem;color:{v_color};margin-bottom:0.1rem;{rtl_css}'>{v_label}</div>", unsafe_allow_html=True)
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

    # ── Make dropdown ─────────────────────────────────────────────────────────
    saved_make  = d.get("manufacturer", "")
    make_idx    = MAKES_LIST.index(saved_make) if saved_make in MAKES_LIST else 0
    manufacturer = st.selectbox(
        t("manufacturer"),
        options=MAKES_LIST,
        index=make_idx,
        format_func=lambda x: t("select_make") if x == "" else x,
    )

    col1, col2 = st.columns(2)
    with col1:
        # ── Model dropdown (depends on make) ──────────────────────────────────
        if manufacturer and manufacturer in CAR_MAKES_MODELS:
            models_list = [""] + CAR_MAKES_MODELS[manufacturer]
            saved_model = d.get("model_name", "")
            model_idx   = models_list.index(saved_model) if saved_model in models_list else 0
            model_name  = st.selectbox(
                t("model"),
                options=models_list,
                index=model_idx,
                format_func=lambda x: t("select_model") if x == "" else x,
            )
        else:
            model_name = st.text_input(t("model"), value=d.get("model_name", ""))

        year     = st.number_input(t("year"), min_value=1990, max_value=2026, step=1,
                                   value=int(d.get("year", 2020)))
        odometer = st.number_input(t("odometer"), min_value=0, max_value=2_000_000,
                                   step=1000, value=int(d.get("odometer", 80_000)))

    with col2:
        trim = st.text_input(t("trim"), value=d.get("trim", ""), placeholder=t("trim_ph"))

        # ── Previous owners selectbox ─────────────────────────────────────────
        owner_opts  = t("prev_owners_opts")   # list of 4 labels
        saved_own   = int(d.get("prev_owners", 1))
        owner_idx   = min(saved_own - 1, 3)   # 1→0, 2→1, 3→2, 4+→3
        prev_owners = st.selectbox(
            t("prev_owners"),
            options=range(1, 5),
            index=owner_idx,
            format_func=lambda x: owner_opts[min(x - 1, 3)],
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button(t("continue_btn")):
        st.session_state.car_details = {
            "manufacturer": manufacturer,
            "model_name":   model_name,
            "year":         int(year),
            "trim":         trim.strip(),
            "odometer":     int(odometer),
            "prev_owners":  int(prev_owners),
        }
        st.session_state.step = 2
        st.rerun()

# ─── Step 2 — Photos ──────────────────────────────────────────────────────────
def step_photos():
    section_label("vehicle_photos")
    st.markdown(f"<p style='font-size:0.95rem;color:var(--muted);{rtl_css}'>{t('photos_hint')}</p>", unsafe_allow_html=True)
    photos = st.file_uploader(t("vehicle_photos"), type=["jpg","jpeg","png","webp"], accept_multiple_files=True, label_visibility="collapsed")
    gold_divider()
    section_label("underbody_title")
    st.markdown(f"<p style='font-size:0.95rem;color:var(--muted);{rtl_css}'>{t('underbody_hint')}</p>", unsafe_allow_html=True)
    underbody = st.file_uploader(t("underbody_title"), type=["jpg","jpeg","png"], label_visibility="collapsed", key="underbody_upload")
    if photos:
        color = "#4A7A4A" if 4 <= len(photos) <= 10 else "#B04040"
        st.markdown(f"<p style='font-size:0.78rem;color:{color};margin-top:0.4rem;'>📸 {len(photos)} {t('photos_count')}</p>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn")):
            st.session_state.step = 1; st.rerun()
    with col2:
        if st.button(t("continue_btn"), key="photos_continue"):
            if not photos or len(photos) < 4: st.error(t("photos_min_error"))
            elif len(photos) > 10:            st.error(t("photos_max_error"))
            else:
                st.session_state.photos   = photos
                st.session_state.underbody = underbody
                st.session_state.step     = 3; st.rerun()

# ─── Step 3 — Audio ───────────────────────────────────────────────────────────
def step_audio():
    section_label("engine_audio")
    st.markdown(f"<p style='font-size:0.95rem;color:var(--muted);{rtl_css}'>{t('audio_hint')}</p>", unsafe_allow_html=True)
    audio = st.file_uploader(t("engine_audio"), type=["mp3","wav","m4a","ogg","aac","flac"], label_visibility="collapsed")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn"), key="audio_back"):
            st.session_state.step = 2; st.rerun()
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
                            "recommendation": decision.recommendation, "confidence": decision.confidence,
                            "top_reasons": decision.top_reasons, "breakdown": decision.breakdown,
                            "education": decision.education, "next_steps": decision.next_steps,
                            "audio_duration_seconds": audio_dur, "car_details": d, "car_label": car_label,
                        }
                        check_id = save_check(st.session_state.email, result)
                        result["check_id"]      = check_id
                        st.session_state.result = result
                        st.session_state.step   = 4; st.rerun()
                    except Exception as e:
                        st.error(f"{t('analysis_failed')}: {e}")

# ─── Main app ─────────────────────────────────────────────────────────────────
def main_app():
    render_sidebar()

    # Compact header with decorative car silhouette SVG
    st.markdown(f"""
    <div style='text-align:center;padding:1.2rem 0 0.6rem;'>
        <svg width="120" height="40" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:0.5rem;opacity:0.55;">
            <path d="M10 28 L18 18 L30 14 L50 12 L75 12 L90 16 L105 22 L110 28 Z" stroke="#C8A96A" stroke-width="1.5" fill="none"/>
            <circle cx="28" cy="30" r="6" stroke="#C8A96A" stroke-width="1.5" fill="none"/>
            <circle cx="88" cy="30" r="6" stroke="#C8A96A" stroke-width="1.5" fill="none"/>
            <path d="M34 28 L82 28" stroke="#C8A96A" stroke-width="1" opacity="0.5"/>
            <path d="M40 20 L50 14 L70 14 L82 20 Z" stroke="#C8A96A" stroke-width="1" fill="rgba(200,169,106,0.05)"/>
        </svg>
        <div style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:2rem;
                    letter-spacing:0.18em;color:var(--gold);text-transform:uppercase;'>
            {t("app_title")}
        </div>
        <div style='font-size:0.92rem;letter-spacing:0.08em;color:var(--muted);margin-top:0.3rem;'>
            {t("app_subtitle_main")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    gold_divider()

    step = st.session_state.step
    step_indicator(step)

    if step == 1:   step_vehicle_details()
    elif step == 2: step_photos()
    elif step == 3: step_audio()
    elif step == 4 and st.session_state.result:
        render_result(st.session_state.result)

# ─── Entry point ──────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    login_screen()
else:
    main_app()
