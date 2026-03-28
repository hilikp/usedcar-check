import streamlit as st
import sys
import os
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
    """Lightweight stand-in for app.analysis.audio_analysis.AudioFinding."""
    def __init__(self, label: str, confidence: float, details: dict = None):
        self.label = label
        self.confidence = confidence
        self.details = details or {}

def _analyze_audio(path: str) -> tuple[list, float]:
    """
    Compute basic spectral features with soundfile + numpy + scipy.
    Mirrors the heuristics in backend/app/analysis/audio_analysis.py.
    """
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

    # Resample to 22 050 Hz if needed
    if sr != 22050:
        target = int(len(data) * 22050 / sr)
        data = sp_signal.resample(data, target)
        sr = 22050

    # ── RMS variance (rough/unstable idle proxy) ──────────────────────────────
    chunk = sr // 2  # 0.5-second chunks
    chunks = [data[i: i + chunk] for i in range(0, len(data) - chunk, chunk)]
    rms_vals = np.array([np.sqrt(np.mean(c ** 2)) for c in chunks])
    rms_mean = float(rms_vals.mean()) if len(rms_vals) else 0.0
    rms_var = float(rms_vals.var()) if len(rms_vals) else 0.0

    # ── Zero-crossing rate (global) ────────────────────────────────────────────
    zcr_mean = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)

    # ── Spectral centroid via FFT (first 5 s) ─────────────────────────────────
    clip = data[: sr * 5] if len(data) > sr * 5 else data
    spectrum = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(len(clip), 1.0 / sr)
    centroid_mean = float(np.sum(freqs * spectrum) / (spectrum.sum() + 1e-10))

    # ── Heuristics (same thresholds as backend) ───────────────────────────────
    findings: list[_AudioFinding] = []
    if rms_mean > 0.01 and (rms_var / (rms_mean ** 2 + 1e-9)) > 0.8:
        findings.append(_AudioFinding("rough_or_unstable", 0.55,
                                      {"rms_mean": rms_mean, "rms_var": rms_var}))
    if centroid_mean > 3500 and zcr_mean > 0.10:
        findings.append(_AudioFinding("high_frequency_squeal_like", 0.45,
                                      {"centroid_mean": centroid_mean, "zcr_mean": zcr_mean}))
    if zcr_mean > 0.12 and centroid_mean < 4500:
        findings.append(_AudioFinding("ticking_like", 0.40,
                                      {"centroid_mean": centroid_mean, "zcr_mean": zcr_mean}))
    if not findings:
        findings.append(_AudioFinding("no_clear_anomaly_detected", 0.60,
                                      {"rms_mean": rms_mean, "zcr_mean": zcr_mean,
                                       "centroid_mean": centroid_mean}))
    return findings, duration

# ─── Constants ────────────────────────────────────────────────────────────────
APP_PASSWORD = "VW2026"
DATA_DIR = _ROOT / "data"
USERS_FILE = DATA_DIR / "users.json"
CHECKS_DIR = DATA_DIR / "checks"
DATA_DIR.mkdir(exist_ok=True)
CHECKS_DIR.mkdir(exist_ok=True)

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="UsedCar Check",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Luxury CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Jost:wght@300;400;500&display=swap');

:root {
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
}

#MainMenu, footer, header { visibility: hidden; }

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Jost', sans-serif;
    color: var(--text) !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Inputs */
input, textarea, select,
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, var(--gold-dark), var(--gold)) !important;
    color: #0C0C0C !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: var(--surface) !important;
    border: 1px dashed var(--gold-dark) !important;
    border-radius: 3px !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* Labels & text */
label, .stMarkdown p, p, li { color: var(--text) !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--gold) !important; }

/* Error / success / info */
[data-testid="stNotification"] { border-radius: 2px !important; }

hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--gold) !important; }
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
    result["email"] = email
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

# ─── Session state init ───────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "email": "",
    "step": 1,
    "car_details": {},
    "photos": [],
    "underbody": None,
    "audio": None,
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── UI helpers ───────────────────────────────────────────────────────────────
def gold_divider():
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);margin:1rem 0;'></div>",
        unsafe_allow_html=True,
    )

def luxury_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style='text-align:center;padding:1.8rem 0 0.8rem;'>
        <div style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:2.6rem;
                    letter-spacing:0.18em;color:var(--gold);text-transform:uppercase;'>
            {title}
        </div>
        {"<div style='font-size:0.7rem;letter-spacing:0.35em;color:var(--muted);text-transform:uppercase;margin-top:0.3rem;'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def verdict_meta(rec: str) -> tuple[str, str, str]:
    return {
        "go":           ("GO",           "#4A7A4A", "rgba(74,122,74,0.12)"),
        "no_go":        ("NO GO",        "#B04040", "rgba(176,64,64,0.12)"),
        "inconclusive": ("INCONCLUSIVE", "#C8A96A", "rgba(200,169,106,0.12)"),
    }.get(rec, ("?", "#9A9080", "rgba(154,144,128,0.12)"))

def badge(text: str, level: str):
    colors = {
        "high":   ("#4A7A4A", "rgba(74,122,74,0.15)"),
        "medium": ("#C8A96A", "rgba(200,169,106,0.15)"),
        "low":    ("#B04040", "rgba(176,64,64,0.15)"),
    }
    fg, bg = colors.get(level, ("#9A9080", "rgba(154,144,128,0.15)"))
    st.markdown(
        f"<span style='background:{bg};color:{fg};border:1px solid {fg};padding:0.15rem 0.6rem;"
        f"border-radius:2px;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;'>{text}</span>",
        unsafe_allow_html=True,
    )

# ─── Analysis runner ──────────────────────────────────────────────────────────
def run_analysis(car_details, photo_files, audio_file, underbody_file=None) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Save photos to disk
        photo_paths = []
        for i, f in enumerate(photo_files):
            ext = Path(f.name).suffix or ".jpg"
            p = tmp / f"photo_{i}{ext}"
            p.write_bytes(f.getvalue())
            photo_paths.append(str(p))

        # Save underbody
        underbody_path = None
        if underbody_file:
            ext = Path(underbody_file.name).suffix or ".jpg"
            up = tmp / f"underbody{ext}"
            up.write_bytes(underbody_file.getvalue())
            underbody_path = str(up)

        # Save audio
        ap = tmp / f"audio{Path(audio_file.name).suffix or '.wav'}"
        ap.write_bytes(audio_file.getvalue())

        # Run analysis
        photo_qualities = [assess_image_quality(p) for p in photo_paths]
        audio_findings, audio_dur = _analyze_audio(str(ap))
        audio_result_findings = audio_findings
        dashboard_findings = detect_dashboard_warnings(photo_paths)

        underbody_findings = []
        if underbody_path:
            underbody_findings = [
                {"label": f.label, "confidence": f.confidence, "details": f.details}
                for f in analyze_underbody_image(underbody_path)
            ]

        decision = decide(
            photo_qualities=photo_qualities,
            audio_findings=audio_result_findings,
            dashboard_findings=[
                {"label": f.label, "confidence": f.confidence} for f in dashboard_findings
            ],
            underbody_findings=underbody_findings,
            driven_km=car_details.get("odometer"),
        )

        return decision, audio_dur

# ─── Result renderer ──────────────────────────────────────────────────────────
def render_result(result: dict):
    rec = result.get("recommendation", "inconclusive")
    conf = result.get("confidence", "low")
    label, color, bg = verdict_meta(rec)
    car_label = result.get("car_label", "")
    date = result.get("created_at", "")[:10]

    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color};border-radius:4px;
                padding:2rem;text-align:center;margin:1rem 0;'>
        <div style='font-family:Cormorant Garamond,serif;font-size:3.2rem;
                    font-weight:600;letter-spacing:0.22em;color:{color};'>{label}</div>
        <div style='font-size:0.75rem;letter-spacing:0.2em;color:var(--muted);
                    margin-top:0.4rem;text-transform:uppercase;'>{car_label}</div>
        <div style='font-size:0.68rem;color:var(--muted);margin-top:0.2rem;'>{date}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span style='font-size:0.7rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;'>Confidence</span>", unsafe_allow_html=True)
        badge(conf.upper(), conf)
    with col2:
        dur = result.get("audio_duration_seconds")
        if dur:
            st.markdown(f"<span style='font-size:0.7rem;color:var(--muted);'>Audio: {dur:.1f}s analysed</span>", unsafe_allow_html=True)

    gold_divider()

    reasons = result.get("top_reasons", [])
    if reasons:
        st.markdown("<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Assessment Findings</p>", unsafe_allow_html=True)
        sev_colors = {"high": "#B04040", "medium": "#C8A96A", "low": "#4A7A4A"}
        for r in reasons:
            sev = r.get("severity", "low")
            bc = sev_colors.get(sev, "#9A9080")
            st.markdown(f"""
            <div style='background:var(--elevated);border-left:3px solid {bc};
                        padding:0.75rem 1rem;margin:0.4rem 0;border-radius:0 3px 3px 0;'>
                <span style='font-size:0.88rem;'>{r.get("title","")}</span>
            </div>
            """, unsafe_allow_html=True)

    steps = result.get("next_steps", [])
    if steps:
        gold_divider()
        st.markdown("<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Recommended Next Steps</p>", unsafe_allow_html=True)
        for s in steps:
            st.markdown(f"<p style='font-size:0.85rem;color:var(--text);'>→ &nbsp;{s.get('text','')}</p>", unsafe_allow_html=True)

    edu = result.get("education", [])
    if edu:
        gold_divider()
        st.markdown("<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Learn More</p>", unsafe_allow_html=True)
        for e in edu:
            st.markdown(f"<p style='font-size:0.82rem;color:var(--muted);'>◦ &nbsp;{e.get('title','')}</p>", unsafe_allow_html=True)

# ─── Step indicator ───────────────────────────────────────────────────────────
def step_indicator(current: int):
    labels = ["Details", "Photos", "Audio", "Result"]
    cols = st.columns(4)
    for i, (col, name) in enumerate(zip(cols, labels), 1):
        with col:
            active = i == current
            done = i < current
            dot_color = "var(--gold)" if active else ("#4A7A4A" if done else "var(--border)")
            txt_color = "var(--gold)" if active else ("#4A7A4A" if done else "var(--muted)")
            st.markdown(
                f"<div style='text-align:center;font-size:0.7rem;letter-spacing:0.1em;"
                f"text-transform:uppercase;color:{txt_color};'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{dot_color};"
                f"margin:0 auto 0.3rem;'></div>{name}</div>",
                unsafe_allow_html=True,
            )
    gold_divider()

# ─── Login screen ─────────────────────────────────────────────────────────────
def login_screen():
    luxury_header("UsedCar Check", "Premium Vehicle Assessment")
    gold_divider()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        email = st.text_input("Email Address", placeholder="your@email.com", key="login_email")
        password = st.text_input("Access Code", type="password", placeholder="· · · · · · · ·", key="login_pass")
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("Enter", use_container_width=True):
            if password != APP_PASSWORD:
                st.error("Invalid access code.")
            elif not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                get_or_create_user(email.lower().strip())
                st.session_state.authenticated = True
                st.session_state.email = email.lower().strip()
                st.session_state.step = 1
                st.session_state.result = None
                st.rerun()

        st.markdown(
            "<p style='font-size:0.68rem;color:var(--muted);text-align:center;margin-top:1rem;"
            "letter-spacing:0.05em;'>Your email is used to access your check history.</p>",
            unsafe_allow_html=True,
        )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:1.2rem 0 0.4rem;'>
            <div style='font-family:Cormorant Garamond,serif;font-size:1.5rem;
                        color:var(--gold);letter-spacing:0.15em;text-transform:uppercase;'>
                UsedCar Check
            </div>
            <div style='font-size:0.62rem;color:var(--muted);letter-spacing:0.25em;
                        text-transform:uppercase;margin-top:0.15rem;'>
                Premium Assessment
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-size:0.72rem;color:var(--muted);margin:0.2rem 0 0.6rem;'>{st.session_state.email}</p>",
            unsafe_allow_html=True,
        )
        gold_divider()

        if st.button("＋  New Check", use_container_width=True):
            st.session_state.step = 1
            st.session_state.car_details = {}
            st.session_state.photos = []
            st.session_state.underbody = None
            st.session_state.audio = None
            st.session_state.result = None
            st.rerun()

        # Past checks list
        past = get_user_checks(st.session_state.email)
        if past:
            st.markdown(
                "<p style='font-size:0.65rem;color:var(--muted);letter-spacing:0.2em;"
                "text-transform:uppercase;margin-top:1.2rem;'>Past Checks</p>",
                unsafe_allow_html=True,
            )
            for check in past[:15]:
                rec = check.get("recommendation", "?")
                v_label, v_color, _ = verdict_meta(rec)
                car = check.get("car_label", check.get("check_id", ""))
                date = check.get("created_at", "")[:10]
                display = f"{car}  ·  {date}" if car else date
                st.markdown(
                    f"<div style='font-size:0.7rem;color:{v_color};letter-spacing:0.05em;"
                    f"margin-bottom:0.1rem;'>{v_label}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(display, key=f"past_{check['check_id']}", use_container_width=True):
                    st.session_state.result = check
                    st.session_state.step = 4
                    st.rerun()

        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ─── Step 1 — Vehicle Details ─────────────────────────────────────────────────
def step_vehicle_details():
    st.markdown(
        "<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Vehicle Details</p>",
        unsafe_allow_html=True,
    )
    d = st.session_state.car_details

    col1, col2 = st.columns(2)
    with col1:
        manufacturer = st.text_input("Manufacturer", value=d.get("manufacturer", ""), placeholder="e.g. Toyota")
        year = st.number_input("Year", min_value=1990, max_value=2026, step=1,
                               value=int(d.get("year", 2020)))
        odometer = st.number_input("Odometer (km)", min_value=0, max_value=2_000_000,
                                   step=1000, value=int(d.get("odometer", 80_000)))
    with col2:
        model_name = st.text_input("Model", value=d.get("model_name", ""), placeholder="e.g. Camry")
        trim = st.text_input("Trim (optional)", value=d.get("trim", ""), placeholder="e.g. Sport, SE")
        nickname = st.text_input("Nickname (optional)", value=d.get("nickname", ""),
                                 placeholder="e.g. Red Camry")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Continue  →", use_container_width=False):
        st.session_state.car_details = {
            "manufacturer": manufacturer.strip(),
            "model_name": model_name.strip(),
            "year": int(year),
            "trim": trim.strip(),
            "odometer": int(odometer),
            "nickname": nickname.strip(),
        }
        st.session_state.step = 2
        st.rerun()

# ─── Step 2 — Photos ──────────────────────────────────────────────────────────
def step_photos():
    st.markdown(
        "<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Vehicle Photos</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.83rem;color:var(--muted);'>Upload 4–10 photos: exterior (all sides), "
        "engine bay, interior, dashboard. Ensure good lighting and a steady hand.</p>",
        unsafe_allow_html=True,
    )

    photos = st.file_uploader(
        "Vehicle Photos", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    gold_divider()
    st.markdown(
        "<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>"
        "Underbody Photo — Optional</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.83rem;color:var(--muted);'>A photo of the underside of the vehicle "
        "helps detect possible fluid leaks.</p>",
        unsafe_allow_html=True,
    )
    underbody = st.file_uploader(
        "Underbody Photo", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="underbody_upload",
    )

    if photos:
        color = "#4A7A4A" if 4 <= len(photos) <= 10 else "#B04040"
        st.markdown(
            f"<p style='font-size:0.78rem;color:{color};margin-top:0.4rem;'>"
            f"{len(photos)} photo(s) selected</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("←  Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Continue  →"):
            if not photos or len(photos) < 4:
                st.error("Please upload at least 4 photos.")
            elif len(photos) > 10:
                st.error("Maximum 10 photos allowed.")
            else:
                st.session_state.photos = photos
                st.session_state.underbody = underbody
                st.session_state.step = 3
                st.rerun()

# ─── Step 3 — Audio ───────────────────────────────────────────────────────────
def step_audio():
    st.markdown(
        "<p style='font-size:0.7rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;'>Engine Audio</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.83rem;color:var(--muted);'>Record at least 10 seconds of the engine "
        "running at idle. Hold the microphone near the engine bay for best results.</p>",
        unsafe_allow_html=True,
    )

    audio = st.file_uploader(
        "Engine Audio", type=["mp3", "wav", "m4a", "ogg", "aac", "flac"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("←  Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Analyse Vehicle  →"):
            if not audio:
                st.error("Please upload an engine audio recording.")
            else:
                d = st.session_state.car_details
                car_label = f"{d.get('year','')} {d.get('manufacturer','')} {d.get('model_name','')}".strip()

                with st.spinner("Analysing — this may take a moment …"):
                    try:
                        decision, audio_dur = run_analysis(
                            d,
                            st.session_state.photos,
                            audio,
                            st.session_state.underbody,
                        )
                        result = {
                            "recommendation":     decision.recommendation,
                            "confidence":         decision.confidence,
                            "top_reasons":        decision.top_reasons,
                            "breakdown":          decision.breakdown,
                            "education":          decision.education,
                            "next_steps":         decision.next_steps,
                            "audio_duration_seconds": audio_dur,
                            "car_details":        d,
                            "car_label":          car_label,
                        }
                        check_id = save_check(st.session_state.email, result)
                        result["check_id"] = check_id
                        st.session_state.result = result
                        st.session_state.step = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

# ─── Main app ─────────────────────────────────────────────────────────────────
def main_app():
    render_sidebar()
    luxury_header("UsedCar Check", "Preliminary Vehicle Assessment")
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
