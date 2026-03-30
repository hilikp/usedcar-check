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

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

def _extract_audio_from_video(src: Path, tmp_dir: Path) -> Path:
    """Extract the first audio track from a video file and write it as WAV."""
    import av
    import numpy as np
    import soundfile as sf

    out_path = tmp_dir / "video_audio.wav"
    container = av.open(str(src))
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        return src  # no audio track — fall through and let soundfile fail gracefully

    frames, sample_rate = [], None
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=22050)
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            sample_rate = 22050
            for out_frame in resampler.resample(frame):
                frames.append(out_frame.to_ndarray()[0])
    container.close()
    if not frames:
        return src
    data = np.concatenate(frames).astype(np.float32)
    sf.write(str(out_path), data, 22050)
    return out_path

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
        "app_subtitle":     "לקנות או לא לקנות.... תגלה תוך דקות",
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
        "audio_hint":       "הקלט לפחות 10 שניות של מנוע פועל בסרק. קרב את המיקרופון לתא המנוע לתוצאות הטובות ביותר. ניתן להעלות גם קובץ וידאו (MP4 וכד׳).",
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
        "feat2_desc":       "זיהוי רעשים חריגים בזמן מנוע פועל בסרק",
        "feat3":            "פסיקה מיידית — קנה / אל תקנה",
        "feat3_desc":       "GO או NO-GO בהתבסס על כל הנתונים",
        "own_single_good":  "בעלים יחיד במשך {age} שנים — סימן חיובי לתחזוקה טובה",
        "own_red_flag":     "{owners} בעלים ב-{age} שנים בלבד — החלפות תכופות: דגל אדום",
        "own_concern":      "ממוצע {avg:.1f} שנים לבעלים — קצר מהמצופה",
        "own_stable":       "היסטוריית בעלות יציבה — ממוצע {avg:.1f} שנים לבעלים",
        "usage_type":       "סוג שימוש קודם",
        "usage_type_opts":  ["פרטי", "השכרה / ליסינג", "רכב חברה", "לא ידוע"],
        "usage_rental":     "רכב השכרה/ליסינג — שימוש אינטנסיבי ושחיקה מהירה יותר",
        "usage_rental_hi":  "רכב השכרה/ליסינג עם {owners} בעלים — בלאי גבוה: דגל אדום",
        "usage_company":    "רכב חברה — בדוק היסטוריית תחזוקה ורשומות שירות",
        "vehicle_video":    "סרטון הרכב (אופציונלי)",
        "video_hint":       "הוסף סרטון עד דקה אחת. הAI יזהה שריטות, נזקי גוף, שינויי צבע ודליפות.",
        "exterior_score":   "ציון חיצוני",
        "interior_score":   "ציון פנים",
        "leak_none":        "לא זוהו דליפות",
        "leak_oil":         "חשד לדליפת שמן",
        "leak_water":       "חשד לדליפת מים / קירור",
        "leaks_title":      "בדיקת דליפות",
        "detailed_report":  "דוח מפורט",
        "scores_title":     "ציוני מצב",
        "of_10":            "/ 10",
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
        "audio_hint":       "Record at least 10 seconds of the engine running at idle. Hold the microphone near the engine bay for best results. You can also upload a video file (MP4, etc.).",
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
        "usage_type":       "Prior Usage Type",
        "usage_type_opts":  ["Private", "Rental / Lease", "Company Car", "Unknown"],
        "usage_rental":     "Rental/lease vehicle — typically higher wear and intensive use",
        "usage_rental_hi":  "Rental/lease with {owners} owners — high wear: red flag",
        "usage_company":    "Company car — verify service history and maintenance records",
        "vehicle_video":    "Vehicle Video (optional)",
        "video_hint":       "Upload up to 1 minute of video. AI will detect scratches, body damage, color changes and leaks.",
        "exterior_score":   "Exterior Score",
        "interior_score":   "Interior Score",
        "leak_none":        "No leaks detected",
        "leak_oil":         "Oil leak suspected",
        "leak_water":       "Water / coolant leak suspected",
        "leaks_title":      "Leak Assessment",
        "detailed_report":  "Detailed Report",
        "scores_title":     "Condition Scores",
        "of_10":            "/ 10",
    }
}

def t(key: str) -> str:
    return TR[st.session_state.get("lang", "he")].get(key, key)

# ─── Car makes & models ───────────────────────────────────────────────────────
CAR_MAKES_MODELS: dict[str, list[str]] = {
    "Alfa Romeo":    ["Giulia", "Stelvio", "Giulietta", "MiTo", "Tonale", "147", "156", "159", "Brera", "Spider"],
    "Audi":          ["A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q4 e-tron", "Q5", "Q7", "Q8", "TT", "R8", "e-tron", "e-tron GT"],
    "BMW":           ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "6 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "iX", "i4", "i5", "i7", "M3", "M5"],
    "BYD":           ["Atto 3", "Dolphin", "Han", "Seal", "Sea Lion 6", "Song", "Tang", "Yuan Plus"],
    "Chevrolet":     ["Spark", "Aveo", "Cruze", "Malibu", "Equinox", "Trax", "Captiva", "Blazer", "Suburban", "Tahoe", "Silverado", "Colorado", "Traverse"],
    "Citroën":       ["C1", "C3", "C3 Aircross", "C4", "C4 Cactus", "C5 Aircross", "C5 X", "Berlingo", "DS3", "DS4", "DS5", "Jumpy"],
    "Cupra":         ["Ateca", "Born", "Formentor", "Leon", "Terramar"],
    "Dacia":         ["Sandero", "Logan", "Duster", "Spring", "Jogger", "Lodgy"],
    "DS":            ["DS 3", "DS 4", "DS 7", "DS 9"],
    "Fiat":          ["Punto", "500", "500X", "500L", "Tipo", "Panda", "Bravo", "Doblo", "Freemont", "Stilo"],
    "Ford":          ["Fiesta", "Focus", "Mondeo", "Fusion", "Kuga", "Puma", "EcoSport", "Edge", "Explorer", "Expedition", "Mustang", "Bronco", "Ranger", "F-150", "Transit", "Galaxy", "S-Max"],
    "Genesis":       ["G70", "G80", "G90", "GV70", "GV80", "GV60"],
    "Haval":         ["H2", "H6", "H9", "Jolion", "Dargo", "F7"],
    "Honda":         ["Civic", "Accord", "Jazz", "HR-V", "CR-V", "ZR-V", "Pilot", "Fit", "City", "Stream", "FR-V", "Legend", "e:Ny1"],
    "Hyundai":       ["i10", "i20", "i30", "i40", "ix35", "ix55", "Tucson", "Santa Fe", "Kona", "Ioniq", "Ioniq 5", "Ioniq 6", "Elantra", "Sonata", "Creta", "Venue", "Nexo", "H-1"],
    "Infiniti":      ["Q30", "Q50", "Q60", "Q70", "QX30", "QX50", "QX60", "QX70", "QX80"],
    "Isuzu":         ["D-Max", "MU-X", "Trooper", "Rodeo", "KB"],
    "Jaguar":        ["XE", "XF", "XJ", "F-Pace", "E-Pace", "I-Pace", "F-Type"],
    "Jeep":          ["Wrangler", "Cherokee", "Grand Cherokee", "Grand Cherokee L", "Renegade", "Compass", "Gladiator", "Commander"],
    "Kia":           ["Picanto", "Rio", "Ceed", "ProCeed", "Stonic", "Sportage", "Sorento", "Niro", "EV6", "EV9", "Carnival", "Telluride", "Seltos", "XCeed", "Soul"],
    "Land Rover":    ["Defender", "Discovery", "Discovery Sport", "Freelander", "Range Rover", "Range Rover Sport", "Range Rover Evoque", "Range Rover Velar"],
    "Lexus":         ["CT", "IS", "ES", "GS", "LS", "UX", "NX", "RX", "GX", "LX", "LC", "RC"],
    "Mazda":         ["Mazda2", "Mazda3", "Mazda6", "CX-3", "CX-30", "CX-5", "CX-60", "CX-90", "MX-5", "BT-50"],
    "Mercedes-Benz": ["A-Class", "B-Class", "C-Class", "CLA", "CLS", "E-Class", "EQA", "EQB", "EQC", "EQE", "EQS", "G-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "S-Class", "SL", "AMG GT"],
    "MG":            ["MG3", "MG4", "MG5", "HS", "ZS", "ZS EV", "Marvel R", "Cyberster"],
    "Mini":          ["Cooper", "Hatch", "Convertible", "Clubman", "Countryman", "Paceman", "Coupe", "Roadster", "Electric", "Aceman"],
    "Mitsubishi":    ["Colt", "Lancer", "Eclipse Cross", "ASX", "Outlander", "Pajero", "Pajero Sport", "L200", "Galant", "Carisma"],
    "Nissan":        ["Micra", "Juke", "Qashqai", "X-Trail", "Leaf", "Ariya", "Navara", "Murano", "Pathfinder", "Note", "Pulsar", "Sentra", "Almera", "Primera", "Tiida"],
    "Opel":          ["Corsa", "Astra", "Insignia", "Mokka", "Crossland", "Grandland", "Zafira", "Meriva", "Adam", "Ampera", "Combo"],
    "Peugeot":       ["107", "108", "207", "208", "307", "308", "407", "508", "2008", "3008", "4008", "5008", "Landtrek", "Partner"],
    "Polestar":      ["Polestar 2", "Polestar 3", "Polestar 4"],
    "Porsche":       ["911", "Boxster", "Cayman", "Cayenne", "Macan", "Panamera", "Taycan"],
    "Renault":       ["Twingo", "Clio", "Captur", "Megane", "Arkana", "Kadjar", "Austral", "Koleos", "Duster", "Laguna", "Scenic", "Talisman", "Zoe", "Kangoo"],
    "Seat":          ["Mii", "Ibiza", "Arona", "Leon", "Tarraco", "Ateca", "Alhambra", "Altea"],
    "Škoda":         ["Fabia", "Rapid", "Scala", "Octavia", "Superb", "Kamiq", "Karoq", "Kodiaq", "Enyaq"],
    "Smart":         ["Fortwo", "Forfour", "#1", "#3"],
    "SsangYong":     ["Tivoli", "Korando", "Rexton", "Musso", "Rodius", "XLV"],
    "Subaru":        ["Impreza", "Legacy", "Outback", "Forester", "XV", "Crosstrek", "WRX", "BRZ", "Solterra", "Levorg"],
    "Suzuki":        ["Alto", "Swift", "Baleno", "Ignis", "Celerio", "SX4", "S-Cross", "Vitara", "Grand Vitara", "Jimny", "Kizashi"],
    "Tesla":         ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
    "Toyota":        ["Aygo", "Yaris", "Corolla", "Camry", "Prius", "RAV4", "C-HR", "Auris", "Avensis", "Verso", "Land Cruiser", "Hilux", "Fortuner", "Rush", "Proace", "GR86", "bZ4X"],
    "Volkswagen":    ["Up!", "Polo", "Golf", "Golf Plus", "Jetta", "Passat", "Arteon", "T-Cross", "T-Roc", "Tiguan", "Touareg", "ID.3", "ID.4", "ID.5", "Caddy", "Touran", "Sharan", "Amarok"],
    "Volvo":         ["V40", "V60", "V90", "S60", "S90", "XC40", "XC60", "XC90", "C40"],
    "Other":         ["Other / לא ברשימה"],
}
import unicodedata as _ud
MAKES_LIST = [""] + sorted(
    (k for k in CAR_MAKES_MODELS if k != "Other"),
    key=lambda x: _ud.normalize("NFD", x).lower()
) + ["Other"]

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
    "vehicle_video": None,
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

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ display: none !important; }}

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
.block-container {{ padding-top: 0.5rem !important; }}

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
    font-size: 1.14rem !important;
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

/* Car driving animation */
@keyframes car-drive {{
    0%   {{ transform: translateX(0px)    scaleX(1);  }}
    6%   {{ transform: translateX(0px)    scaleX(-1); }}
    44%  {{ transform: translateX(-44vw)  scaleX(-1); }}
    50%  {{ transform: translateX(-44vw)  scaleX(1);  }}
    94%  {{ transform: translateX(0px)    scaleX(1);  }}
    100% {{ transform: translateX(0px)    scaleX(1);  }}
}}
.car-animated {{
    animation: car-drive 7s ease-in-out infinite;
    display: inline-block;
    transform-origin: center center;
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
        f"<p style='font-size:1.07rem;color:var(--muted);letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;{rtl_css}'>{t(key)}</p>",
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
        f"border-radius:2px;font-size:1.07rem;letter-spacing:0.08em;text-transform:uppercase;'>{label}</span>",
        unsafe_allow_html=True,
    )

# ─── Language toggle ──────────────────────────────────────────────────────────
# (Inline single-flag toggle used directly in login_screen() and render_sidebar())

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

# ─── Usage-type analysis ──────────────────────────────────────────────────────
def _analyze_usage(usage_type: int, num_owners: int) -> list[dict]:
    """Return reason dicts based on prior usage type (rental/lease/company)."""
    lang    = st.session_state.get("lang", "he")
    reasons = []
    if usage_type == 1:   # Rental / Lease
        if num_owners >= 3:
            reasons.append({
                "severity": "high",
                "title": TR[lang]["usage_rental_hi"].format(owners=num_owners),
            })
        else:
            reasons.append({
                "severity": "medium",
                "title": TR[lang]["usage_rental"],
            })
    elif usage_type == 2:  # Company Car
        reasons.append({
            "severity": "low",
            "title": TR[lang]["usage_company"],
        })
    return reasons

# ─── Video frame extraction ───────────────────────────────────────────────────
def _extract_video_frames(video_path: str, tmp_dir: Path, max_frames: int = 15) -> list[str]:
    """Extract frames from a video at regular intervals using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total       = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Clamp to first 60 s
    max_dur_frames = int(fps * 60)
    effective_total = min(total, max_dur_frames)
    interval    = max(1, int(effective_total / max_frames))
    frame_paths: list[str] = []
    frame_count = 0
    while cap.isOpened() and len(frame_paths) < max_frames:
        ret, frame = cap.read()
        if not ret or frame_count >= max_dur_frames:
            break
        if frame_count % interval == 0:
            out = tmp_dir / f"vframe_{len(frame_paths):03d}.jpg"
            cv2.imwrite(str(out), frame)
            frame_paths.append(str(out))
        frame_count += 1
    cap.release()
    return frame_paths

# ─── Comprehensive AI report ──────────────────────────────────────────────────
def _generate_comprehensive_report(
    car_details: dict,
    decision,
    photo_paths: list[str],
    video_frame_paths: list[str],
    lang: str,
    api_key: str,
) -> dict:
    """
    Single Claude Sonnet call that:
      • Writes a 10-sentence detailed assessment in Hebrew or English
      • Scores exterior 1–10 and interior 1–10
      • Detects fluid leaks
      • Translates all backend findings titles to the target language
    Returns dict with keys: report, exterior_score, interior_score,
                            leak_assessment, translated_reasons, translated_steps
    """
    import anthropic, base64, json, re

    client = anthropic.Anthropic(api_key=api_key)
    lang_name = "Hebrew" if lang == "he" else "English"

    # Combine photos + video frames, sample up to 8 for the API call
    all_paths = photo_paths + video_frame_paths
    sample    = all_paths[:8]
    img_blocks = []
    for p in sample:
        try:
            ext = Path(p).suffix.lower().lstrip(".")
            media = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                     "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
            with open(p, "rb") as fh:
                data = base64.standard_b64encode(fh.read()).decode()
            img_blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        except Exception:
            continue

    # Summarise existing findings (in English from backend) for context
    existing_reasons = [r.get("title", "") for r in (decision.top_reasons or [])]
    existing_steps   = [s.get("text", "")  for s in (decision.next_steps   or [])]

    prompt = f"""You are a professional used-car inspector. You have {len(img_blocks)} image(s) of this vehicle plus the automated system's preliminary findings.

Vehicle: {car_details.get("year","")} {car_details.get("manufacturer","")} {car_details.get("model_name","")} {car_details.get("trim","")}
Odometer: {car_details.get("odometer","?")} km | Prior owners: {car_details.get("prev_owners",1)} | Usage: {["Private","Rental/Lease","Company Car","Unknown"][min(car_details.get("usage_type",0),3)]}
System verdict: {decision.recommendation.upper()} (confidence: {decision.confidence})

Automated findings (English): {"; ".join(existing_reasons) or "none"}
Automated next steps (English): {"; ".join(existing_steps) or "none"}

Your task — respond ONLY in {lang_name} — produce this exact JSON (no extra text):
{{
  "report": "<exactly 10 sentences covering: overall condition, body/panel damage visible, paint quality & colour consistency, glass & lights condition, wheel & tyre condition, engine bay cleanliness, interior condition, dashboard & controls, underbody if visible, and a final buying-advice sentence>",
  "exterior_score": <integer 1-10, where 10=showroom perfect, 1=severely damaged>,
  "interior_score": <integer 1-10, where 10=pristine, 1=heavily worn/damaged>,
  "leak_assessment": "<one of: none detected | oil leak suspected | water/coolant leak suspected | multiple leaks suspected>",
  "translated_reasons": ["<translate each automated finding to {lang_name}, preserving meaning — keep the same count>"],
  "translated_steps": ["<translate each next step to {lang_name}>"]
}}

Base exterior_score and interior_score ONLY on what you can visually observe in the images. Be honest and specific. Write all text in {lang_name}."""

    content = img_blocks + [{"type": "text", "text": prompt}]
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        raw  = resp.content[0].text
        m    = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    # Fallback — return empty structure so result still displays
    return {
        "report": "",
        "exterior_score": None,
        "interior_score": None,
        "leak_assessment": "none detected",
        "translated_reasons": existing_reasons,
        "translated_steps":   existing_steps,
    }

# ─── Analysis runner ──────────────────────────────────────────────────────────
def run_analysis(car_details, photo_files, audio_file, underbody_file=None, video_file=None) -> tuple:
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
        if ap.suffix.lower() in _VIDEO_EXTS:
            ap = _extract_audio_from_video(ap, tmp)

        # ── Video frame extraction ────────────────────────────────────────────
        video_frame_paths: list[str] = []
        if video_file is not None:
            vp = tmp / f"vehicle_video{Path(video_file.name).suffix or '.mp4'}"
            vp.write_bytes(video_file.getvalue())
            video_frame_paths = _extract_video_frames(str(vp), tmp)

        # All photo paths including video frames (for backend quality assessment)
        all_visual_paths = photo_paths + video_frame_paths

        photo_qualities   = [assess_image_quality(p) for p in all_visual_paths]
        audio_findings, audio_dur = _analyze_audio(str(ap))
        dashboard_findings = detect_dashboard_warnings(all_visual_paths)
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

        # ── Ownership & usage analysis — inject into decision ─────────────────
        num_owners  = int(car_details.get("prev_owners", 1))
        usage_type  = int(car_details.get("usage_type", 0))
        year        = int(car_details.get("year", 2015))
        own_reasons   = _analyze_ownership(num_owners, year)
        usage_reasons = _analyze_usage(usage_type, num_owners)
        extra_reasons = own_reasons + usage_reasons
        if extra_reasons:
            has_high = any(r.get("severity") == "high" for r in extra_reasons)
            if has_high and decision.recommendation == "go":
                decision.recommendation = "inconclusive"
            decision.top_reasons = extra_reasons + decision.top_reasons

        # ── Comprehensive AI report (scores + translation + 10-sentence report)
        lang = st.session_state.get("lang", "he")
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        ai_report = _generate_comprehensive_report(
            car_details, decision, all_visual_paths, [], lang, api_key
        )

        # Apply translations back to decision reasons/steps
        tr_reasons = ai_report.get("translated_reasons", [])
        tr_steps   = ai_report.get("translated_steps",   [])
        if tr_reasons:
            for i, r in enumerate(decision.top_reasons):
                if i < len(tr_reasons) and tr_reasons[i]:
                    r["title"] = tr_reasons[i]
        if tr_steps:
            for i, s in enumerate(decision.next_steps or []):
                if i < len(tr_steps) and tr_steps[i]:
                    s["text"] = tr_steps[i]

        return decision, audio_dur, ai_report

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
        <div style='font-size:1.17rem;letter-spacing:0.2em;color:var(--muted);
                    text-transform:uppercase;'>{car_label}</div>
        <div style='font-size:1.04rem;color:var(--muted);margin-top:0.3rem;'>{date}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence + Audio meta ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<span style='font-size:1.11rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;'>{t('confidence_label')}</span>", unsafe_allow_html=True)
        badge(conf.upper(), conf)
    with col2:
        dur = result.get("audio_duration_seconds")
        if dur:
            st.markdown(f"<span style='font-size:1.11rem;color:var(--muted);'>🎙 {t('audio_analysed').format(dur)}</span>", unsafe_allow_html=True)

    gold_divider()

    # ── Condition Scores (exterior + interior) ────────────────────────────────
    ext_score = result.get("exterior_score")
    int_score = result.get("interior_score")
    if ext_score is not None or int_score is not None:
        section_label("scores_title")
        sc1, sc2 = st.columns(2)
        def _score_bar(col, label_key: str, score):
            with col:
                if score is None:
                    return
                pct   = int(score) * 10
                s_col = "#4A7A4A" if score >= 7 else ("#C8A96A" if score >= 5 else "#B04040")
                col.markdown(f"""
                <div style='{rtl_css}margin-bottom:0.4rem;'>
                    <span style='font-size:1.07rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;'>{t(label_key)}</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.8rem;{rtl_css}'>
                    <div style='flex:1;height:8px;background:var(--elevated);border-radius:4px;overflow:hidden;'>
                        <div style='width:{pct}%;height:100%;background:{s_col};border-radius:4px;
                                    transition:width 0.6s ease;'></div>
                    </div>
                    <span style='font-size:1.5rem;font-weight:600;color:{s_col};min-width:2.5rem;text-align:center;'>{score}</span>
                    <span style='font-size:1rem;color:var(--muted);'>{t("of_10")}</span>
                </div>
                """, unsafe_allow_html=True)
        _score_bar(sc1, "exterior_score", ext_score)
        _score_bar(sc2, "interior_score", int_score)
        gold_divider()

    # ── Leak assessment ───────────────────────────────────────────────────────
    leak_raw = (result.get("leak_assessment") or "none detected").lower()
    if "oil" in leak_raw:
        leak_label, leak_color = t("leak_oil"),   "#B04040"
        leak_icon = "🔴"
    elif "water" in leak_raw or "coolant" in leak_raw:
        leak_label, leak_color = t("leak_water"), "#C8A96A"
        leak_icon = "🟡"
    elif "multiple" in leak_raw:
        leak_label, leak_color = t("leak_oil") + " + " + t("leak_water"), "#B04040"
        leak_icon = "🔴"
    else:
        leak_label, leak_color = t("leak_none"),  "#4A7A4A"
        leak_icon = "🟢"
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.7rem;margin:0.2rem 0 0.8rem;{rtl_css}'>
        <span style='font-size:1.07rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;'>{t("leaks_title")}:</span>
        <span style='font-size:1.1rem;'>{leak_icon}</span>
        <span style='font-size:1.17rem;color:{leak_color};font-weight:500;'>{leak_label}</span>
    </div>
    """, unsafe_allow_html=True)

    gold_divider()

    # ── Detailed 10-sentence AI report ───────────────────────────────────────
    report_text = result.get("detailed_report", "")
    if report_text:
        section_label("detailed_report")
        # Split into sentences for better readability
        import re as _re
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', report_text) if s.strip()]
        for sentence in sentences:
            st.markdown(f"""
            <div style='border-left:2px solid rgba(200,169,106,0.3);padding:0.4rem 0.9rem;
                        margin:0.3rem 0;{rtl_css}'>
                <span style='font-size:1.17rem;color:var(--text);line-height:1.6;'>{sentence}</span>
            </div>
            """, unsafe_allow_html=True)
        gold_divider()

    # ── Assessment Findings ───────────────────────────────────────────────────
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
                <span style='font-size:1.24rem;'>{r.get("title","")}</span>
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
                            justify-content:center;font-size:0.85rem;color:var(--gold);flex-shrink:0;'>{i}</div>
                <div style='font-size:1.24rem;color:var(--text);padding-top:0.2rem;'>{s.get("text","")}</div>
            </div>
            """, unsafe_allow_html=True)

    edu = result.get("education", [])
    if edu:
        gold_divider()
        section_label("learn_more_title")
        for e in edu:
            st.markdown(f"<p style='font-size:1.07rem;color:var(--muted);{rtl_css}'>◦ &nbsp;{e.get('title','')}</p>", unsafe_allow_html=True)

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
                <div style='font-size:1.04rem;letter-spacing:0.08em;text-transform:uppercase;color:{txt};'>{name}</div>
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
        <div style='font-size:1.37rem;letter-spacing:0.1em;color:rgba(240,235,224,0.75);'>
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
                        <div style='font-size:1.3rem;color:var(--gold);font-weight:500;'>{t(feat_key)}</div>
                        <div style='font-size:1.14rem;color:var(--muted);margin-top:0.2rem;'>{t(desc_key)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # Language flag — single button showing the OTHER language flag
        fc1, _ = st.columns([1, 6])
        with fc1:
            st.markdown("<div class='lang-btn'>", unsafe_allow_html=True)
            _other_flag = "🇺🇸" if st.session_state.lang == "he" else "🇮🇱"
            _other_lang = "en"  if st.session_state.lang == "he" else "he"
            if st.button(_other_flag, key="login_lang_toggle"):
                st.session_state.lang = _other_lang; st.rerun()
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
            f"<p style='font-size:1.04rem;color:var(--muted);text-align:center;margin-top:0.8rem;'>{t('email_hint')}</p>",
            unsafe_allow_html=True,
        )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        col_flag, _ = st.columns([1, 5])
        with col_flag:
            st.markdown("<div class='lang-btn'>", unsafe_allow_html=True)
            _other_flag = "🇺🇸" if st.session_state.lang == "he" else "🇮🇱"
            _other_lang = "en"  if st.session_state.lang == "he" else "he"
            if st.button(_other_flag, key="sb_lang_toggle"):
                st.session_state.lang = _other_lang
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='padding:0.8rem 0 0.4rem;{rtl_css}'>
            <div style='font-family:Cormorant Garamond,serif;font-size:1.95rem;
                        color:var(--gold);letter-spacing:0.15em;text-transform:uppercase;'>
                {t("app_title")}
            </div>
            <div style='font-size:1.01rem;color:var(--muted);letter-spacing:0.08em;
                        margin-top:0.2rem;'>{t("app_subtitle")}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:1.11rem;color:var(--muted);margin:0.2rem 0 0.6rem;{rtl_css}'>{st.session_state.email}</p>", unsafe_allow_html=True)
        gold_divider()

        if st.button(t("new_check"), use_container_width=True):
            st.session_state.step          = 1
            st.session_state.car_details   = {}
            st.session_state.photos        = []
            st.session_state.underbody     = None
            st.session_state.vehicle_video = None
            st.session_state.audio         = None
            st.session_state.result        = None
            st.rerun()

        past = get_user_checks(st.session_state.email)
        if past:
            st.markdown(f"<p style='font-size:1.04rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;margin-top:1.2rem;{rtl_css}'>{t('past_checks')}</p>", unsafe_allow_html=True)
            for check in past[:15]:
                rec = check.get("recommendation", "?")
                v_label, v_color, _ = verdict_meta(rec)
                car  = check.get("car_label", check.get("check_id", ""))
                date = check.get("created_at", "")[:10]
                display = f"{car}  ·  {date}" if car else date
                st.markdown(f"<div style='font-size:1.07rem;color:{v_color};margin-bottom:0.1rem;{rtl_css}'>{v_label}</div>", unsafe_allow_html=True)
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

        # ── Prior usage type ──────────────────────────────────────────────────
        usage_opts = t("usage_type_opts")   # list of 4 labels
        saved_usage = int(d.get("usage_type", 0))
        usage_type  = st.selectbox(
            t("usage_type"),
            options=range(4),
            index=min(saved_usage, 3),
            format_func=lambda x: usage_opts[x],
        )

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
            "usage_type":   int(usage_type),
            "prev_owners":  int(prev_owners),
        }
        st.session_state.step = 2
        st.rerun()

# ─── Step 2 — Photos ──────────────────────────────────────────────────────────
def step_photos():
    section_label("vehicle_photos")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('photos_hint')}</p>", unsafe_allow_html=True)
    photos = st.file_uploader(t("vehicle_photos"), type=["jpg","jpeg","png","webp"], accept_multiple_files=True, label_visibility="collapsed")
    gold_divider()
    section_label("underbody_title")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('underbody_hint')}</p>", unsafe_allow_html=True)
    underbody = st.file_uploader(t("underbody_title"), type=["jpg","jpeg","png"], label_visibility="collapsed", key="underbody_upload")
    if photos:
        color = "#4A7A4A" if 4 <= len(photos) <= 10 else "#B04040"
        st.markdown(f"<p style='font-size:1.01rem;color:{color};margin-top:0.4rem;'>📸 {len(photos)} {t('photos_count')}</p>", unsafe_allow_html=True)
    gold_divider()
    section_label("vehicle_video")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('video_hint')}</p>", unsafe_allow_html=True)
    vehicle_video = st.file_uploader(t("vehicle_video"), type=["mp4","mov","avi","mkv","webm"], label_visibility="collapsed", key="vehicle_video_upload")
    if vehicle_video:
        size_mb = len(vehicle_video.getvalue()) / (1024 * 1024)
        st.markdown(f"<p style='font-size:1.01rem;color:#4A7A4A;margin-top:0.3rem;'>🎬 {vehicle_video.name} ({size_mb:.1f} MB)</p>", unsafe_allow_html=True)
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
                st.session_state.photos        = photos
                st.session_state.underbody     = underbody
                st.session_state.vehicle_video = vehicle_video
                st.session_state.step          = 3; st.rerun()

# ─── Step 3 — Audio ───────────────────────────────────────────────────────────
def step_audio():
    section_label("engine_audio")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('audio_hint')}</p>", unsafe_allow_html=True)
    audio = st.file_uploader(t("engine_audio"), type=["mp3","wav","m4a","ogg","aac","flac","mp4","mov","avi","mkv","webm","3gp"], label_visibility="collapsed")
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
                        decision, audio_dur, ai_report = run_analysis(
                            d, st.session_state.photos, audio,
                            st.session_state.underbody,
                            st.session_state.get("vehicle_video"),
                        )
                        result = {
                            "recommendation":  decision.recommendation,
                            "confidence":      decision.confidence,
                            "top_reasons":     decision.top_reasons,
                            "breakdown":       decision.breakdown,
                            "education":       decision.education,
                            "next_steps":      decision.next_steps,
                            "audio_duration_seconds": audio_dur,
                            "car_details":     d,
                            "car_label":       car_label,
                            "exterior_score":  ai_report.get("exterior_score"),
                            "interior_score":  ai_report.get("interior_score"),
                            "leak_assessment": ai_report.get("leak_assessment", "none detected"),
                            "detailed_report": ai_report.get("report", ""),
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

    # Compact header with animated car silhouette
    st.markdown(f"""
    <div style='text-align:center;padding:1.2rem 0 0.2rem;overflow:hidden;'>
        <div class='car-animated' style='margin-bottom:0.6rem;'>
            <svg width="360" height="120" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity:0.75;filter:drop-shadow(0 0 8px rgba(200,169,106,0.4));">
                <path d="M10 28 L18 18 L30 14 L50 12 L75 12 L90 16 L105 22 L110 28 Z" stroke="#C8A96A" stroke-width="1.2" fill="rgba(200,169,106,0.04)"/>
                <circle cx="28" cy="30" r="6" stroke="#C8A96A" stroke-width="1.2" fill="none"/>
                <circle cx="28" cy="30" r="3" fill="rgba(200,169,106,0.15)"/>
                <circle cx="88" cy="30" r="6" stroke="#C8A96A" stroke-width="1.2" fill="none"/>
                <circle cx="88" cy="30" r="3" fill="rgba(200,169,106,0.15)"/>
                <path d="M34 28 L82 28" stroke="#C8A96A" stroke-width="0.8" opacity="0.4"/>
                <path d="M40 20 L50 14 L70 14 L82 20 Z" stroke="#C8A96A" stroke-width="0.8" fill="rgba(200,169,106,0.08)"/>
                <path d="M4 28 L116 28" stroke="#C8A96A" stroke-width="0.5" opacity="0.2"/>
            </svg>
        </div>
        <div style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:2rem;
                    letter-spacing:0.18em;color:var(--gold);text-transform:uppercase;'>
            {t("app_title")}
        </div>
        <div style='font-size:1.56rem;letter-spacing:0.08em;color:var(--muted);margin-top:0.3rem;'>
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
