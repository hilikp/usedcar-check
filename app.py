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

def _analyze_audio(path: str) -> tuple[list, float, dict]:
    """Enhanced audio analysis — 6 named findings + full acoustic metrics dict."""
    import soundfile as sf
    import numpy as np
    from scipy import signal as sp_signal

    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    duration = len(data) / sr

    if len(data) < sr * 2:
        empty_metrics = {"reason": "audio_too_short", "duration_s": round(duration, 2)}
        return [_AudioFinding("unknown", 0.2, {"reason": "audio_too_short"})], duration, empty_metrics

    if sr != 22050:
        target = int(len(data) * 22050 / sr)
        data = sp_signal.resample(data, target)
        sr = 22050

    # ── Per-chunk RMS (0.5-second chunks) ────────────────────────────────────
    chunk_size = sr // 2
    chunks   = [data[i: i + chunk_size] for i in range(0, len(data) - chunk_size, chunk_size)]
    rms_vals = np.array([np.sqrt(np.mean(c ** 2)) for c in chunks])
    rms_mean = float(rms_vals.mean()) if len(rms_vals) else 0.0
    rms_cv   = float(rms_vals.var()) / (rms_mean ** 2 + 1e-9)
    zcr_mean = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)

    # ── Power spectrum (first 10 s) + band energies ───────────────────────────
    clip    = data[:sr * 10] if len(data) > sr * 10 else data
    freqs   = np.fft.rfftfreq(len(clip), 1.0 / sr)
    power   = np.abs(np.fft.rfft(clip)) ** 2
    total_p = power.sum() + 1e-10

    def _band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(power[m].sum() / total_p)

    bass_energy   = _band(20,   250)
    mid_energy    = _band(250,  2000)
    treble_energy = _band(2000, 8000)
    tick_energy   = _band(800,  3500)   # valve/tappet tick range
    spectral_centroid = float(np.sum(freqs * np.sqrt(power)) / (np.sqrt(power).sum() + 1e-10))

    # ── Autocorrelation periodicity at idle-RPM lags (600–1200 RPM) ──────────
    ac_clip = data[:sr * 2]
    ac_norm = ac_clip / (np.max(np.abs(ac_clip)) + 1e-9)
    acorr   = np.correlate(ac_norm, ac_norm, mode="full")[len(ac_norm) - 1:]
    acorr  /= (acorr[0] + 1e-9)
    lag_lo, lag_hi = int(sr * 0.05), int(sr * 0.1)   # 600–1200 RPM window
    periodicity_score = float(acorr[lag_lo: lag_hi].max()) if lag_hi < len(acorr) else 0.0

    # ── Bass frame variability (periodic bass impulses = knock) ──────────────
    frame_sz = int(sr * 0.1)
    bass_per_frame = []
    for fr in [data[i: i + frame_sz] for i in range(0, min(len(data) - frame_sz, sr * 10), frame_sz)]:
        ps = np.abs(np.fft.rfft(fr)) ** 2
        fs = np.fft.rfftfreq(len(fr), 1.0 / sr)
        bass_per_frame.append(float(ps[(fs >= 20) & (fs < 250)].sum()))
    bass_frame_mean = float(np.mean(bass_per_frame)) if bass_per_frame else 1.0
    bass_cv = float(np.var(bass_per_frame)) / (bass_frame_mean ** 2 + 1e-9) if bass_per_frame else 0.0

    metrics = {
        "rms_mean":          round(rms_mean, 4),
        "rms_cv":            round(rms_cv, 3),
        "bass_energy":       round(bass_energy, 3),
        "mid_energy":        round(mid_energy, 3),
        "treble_energy":     round(treble_energy, 3),
        "tick_energy":       round(tick_energy, 3),
        "spectral_centroid": round(spectral_centroid, 1),
        "zcr_mean":          round(zcr_mean, 4),
        "periodicity_score": round(periodicity_score, 3),
        "bass_cv":           round(bass_cv, 3),
        "duration_s":        round(duration, 2),
    }

    # ── 6 named findings (lowered thresholds — phone recordings have lower SNR)
    findings: list[_AudioFinding] = []

    # Rod knock: periodic low-frequency impulses
    if periodicity_score > 0.35 and bass_cv > 0.4 and bass_energy > 0.12:
        findings.append(_AudioFinding("rod_knock_suspected", 0.70,
            {"periodicity": periodicity_score, "bass_cv": bass_cv, "bass_energy": bass_energy}))

    # Valve/tappet tick: elevated tick-range energy (800-3500 Hz) with any periodicity
    if tick_energy > 0.08 and (treble_energy > 0.06 or periodicity_score > 0.10):
        findings.append(_AudioFinding("valve_tick_suspected", 0.60,
            {"tick_energy": tick_energy, "treble_energy": treble_energy, "centroid_hz": spectral_centroid}))

    # Belt squeal: very high-frequency with high zero-crossing rate
    if treble_energy > 0.14 and zcr_mean > 0.08 and spectral_centroid > 3000:
        findings.append(_AudioFinding("belt_squeal_suspected", 0.55,
            {"treble_energy": treble_energy, "zcr": zcr_mean, "centroid_hz": spectral_centroid}))

    # Exhaust leak: heavy low-frequency energy with variable amplitude
    if bass_energy > 0.30 and rms_cv > 0.5 and spectral_centroid < 1800:
        findings.append(_AudioFinding("exhaust_leak_suspected", 0.55,
            {"bass_energy": bass_energy, "rms_cv": rms_cv, "centroid_hz": spectral_centroid}))

    # Rough idle: high amplitude variation, not periodic
    if rms_cv > 0.6 and rms_mean > 0.008 and periodicity_score < 0.35:
        findings.append(_AudioFinding("rough_idle_suspected", 0.55,
            {"rms_cv": rms_cv, "rms_mean": rms_mean, "periodicity": periodicity_score}))

    if not findings:
        findings.append(_AudioFinding("engine_sounds_normal", 0.65,
            {"rms_cv": rms_cv, "periodicity": periodicity_score, "centroid_hz": spectral_centroid}))

    return findings, duration, metrics

# ─── Paint consistency analysis ──────────────────────────────────────────────
def _analyze_paint_consistency(photo_paths: list[str]) -> dict:
    """
    Compare LAB color histograms across regions of each exterior photo.
    Returns a dict with per-photo panel anomalies and an overall suspicion level.
    """
    import cv2
    import numpy as np

    _EMPTY = {"anomalies": [], "suspicion": "none", "panels_checked": 0}
    if not photo_paths:
        return _EMPTY

    # Bhattacharyya distance threshold — above this = potential mismatch
    # Raised from 0.28 to 0.38 to suppress natural lighting variation false positives
    _THRESH = 0.38

    def _hist(roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        h_a = cv2.calcHist([lab], [1], None, [64], [0, 256])
        h_b = cv2.calcHist([lab], [2], None, [64], [0, 256])
        cv2.normalize(h_a, h_a)
        cv2.normalize(h_b, h_b)
        return h_a, h_b

    def _dist(h1a, h1b, h2a, h2b):
        da = cv2.compareHist(h1a, h2a, cv2.HISTCMP_BHATTACHARYYA)
        db = cv2.compareHist(h1b, h2b, cv2.HISTCMP_BHATTACHARYYA)
        return round((da + db) / 2, 3)

    anomalies = []
    panels_checked = 0

    for path in photo_paths[:8]:   # cap at 8 images
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            # Skip small / low-quality frames
            if h < 80 or w < 80:
                continue

            # Divide image into 6 regions: top-left, top-right, mid-left,
            # mid-right, bottom-left, bottom-right
            rows = [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
            cols = [(0, w // 2), (w // 2, w)]
            regions, region_names = [], []
            for ri, (r0, r1) in enumerate(rows[:2]):   # top + mid rows only
                for ci, (c0, c1) in enumerate(cols):
                    roi = img[r0:r1, c0:c1]
                    if roi.size == 0:
                        continue
                    regions.append(_hist(roi))
                    side = ["L", "R"][ci]
                    row_name = ["top", "mid"][ri]
                    region_names.append(f"{row_name}-{side}")
            panels_checked += len(regions)

            # Compare every adjacent pair (top-L vs top-R, top-L vs mid-L, etc.)
            pairs = [
                (0, 1, "עליון-שמאל / עליון-ימין"),
                (2, 3, "אמצע-שמאל / אמצע-ימין"),
                (0, 2, "עליון-שמאל / אמצע-שמאל"),
                (1, 3, "עליון-ימין / אמצע-ימין"),
            ]
            for i, j, pair_name in pairs:
                if i >= len(regions) or j >= len(regions):
                    continue
                d = _dist(*regions[i], *regions[j])
                if d > _THRESH:
                    anomalies.append({
                        "photo": Path(path).name,
                        "pair":  pair_name,
                        "score": d,
                        "severity": "high" if d > 0.55 else "medium",
                    })
        except Exception:
            continue

    # Aggregate overall suspicion — require multiple strong signals
    high_count = sum(1 for a in anomalies if a["severity"] == "high")
    if not anomalies:
        suspicion = "none"
    elif high_count >= 2:
        suspicion = "high"
    elif high_count == 1 or len(anomalies) >= 4:
        suspicion = "medium"
    elif len(anomalies) >= 2:
        suspicion = "low"
    else:
        suspicion = "none"  # single borderline hit — ignore

    return {
        "anomalies":      anomalies,
        "suspicion":      suspicion,
        "panels_checked": panels_checked,
    }

# ─── NHTSA Safety Data ────────────────────────────────────────────────────────
def _fetch_nhtsa_data(make: str, model: str, year: int) -> dict:
    """Fetch recall + complaint data from the free NHTSA API (no key needed)."""
    import urllib.request, urllib.parse, json as _json
    from concurrent.futures import ThreadPoolExecutor

    _EMPTY = {"recalls": [], "complaint_components": {}, "total_complaints": 0, "recall_count": 0}
    if not make or not model or not year:
        return _EMPTY

    params = urllib.parse.urlencode({"make": make, "model": model, "modelYear": year})
    recall_url    = f"https://api.nhtsa.dot.gov/recalls/recallsByVehicle?{params}"
    complaint_url = f"https://api.nhtsa.dot.gov/complaints/complaintsByVehicle?{params}"

    def _get(url: str):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UsedCarCheck/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                return _json.loads(resp.read().decode())
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_recalls    = pool.submit(_get, recall_url)
        f_complaints = pool.submit(_get, complaint_url)
        recalls_raw    = f_recalls.result()
        complaints_raw = f_complaints.result()

    recalls = []
    if recalls_raw and isinstance(recalls_raw.get("results"), list):
        for r in recalls_raw["results"]:
            recalls.append({
                "component":   r.get("Component", ""),
                "summary":     r.get("Summary", "")[:300],
                "consequence": r.get("Consequence", "")[:200],
            })

    comp_counts: dict[str, int] = {}
    total_complaints = 0
    if complaints_raw and isinstance(complaints_raw.get("results"), list):
        total_complaints = len(complaints_raw["results"])
        for c in complaints_raw["results"]:
            for part in (c.get("components") or "").upper().replace(",", "/").split("/"):
                part = part.strip()
                if part:
                    comp_counts[part] = comp_counts.get(part, 0) + 1

    return {
        "recalls":             recalls,
        "complaint_components": comp_counts,
        "total_complaints":    total_complaints,
        "recall_count":        len(recalls),
    }

# ─── Translations ─────────────────────────────────────────────────────────────
TR = {
    "he": {
        "app_title":        "בדיקת רכב",
        "app_subtitle":     "האמת על הרכב | לפני שהוא שלך",
        "app_subtitle_main":"ניתוח חכם. החלטה בטוחה.",
        "email_label":      "כתובת אימייל",
        "password_label":   "קוד גישה",
        "enter_btn":        "כניסה",
        "email_hint":       "האימייל שלך משמש לגישה להיסטוריית הבדיקות שלך.",
        "disclaimer":       "אתר זה מספק סיוע בסיסי בלבד לרוכשי רכב משומש. הגולש/ת הוא/היא האחראי/ת הבלעדי/ת להחלטה בדבר רכישת הרכב. האתר ומפתחיו אינם אחראים בשום צורה ואופן לכל החלטה שגויה או נזק הנובע משימוש בשירות. ההחלטה הסופית לגבי רכישת רכב צריכה להתקבל בהתבסס על חוות דעת מקצועית בכתב ממכונאי מוסמך, או לאחר בדיקת הרכב במכון בדיקה מורשה לבדיקות רכב לפני קנייה או מכירה.",
        "disclaimer_accept":"בהזנת האימייל שלך אתה מאשר/ת את תנאי השימוש של האתר.",
        "invalid_code":     "קוד גישה שגוי.",
        "invalid_email":    "אנא הזן כתובת אימייל תקינה.",
        "plate_label":      "מספר לוחית רישוי",
        "plate_hint":       "הזן מספר רישוי לטעינה אוטומטית של פרטי הרכב ממשרד התחבורה",
        "plate_lookup_btn": "טעינה ↓",
        "plate_found":      "✓ פרטי הרכב נטענו ממשרד התחבורה",
        "plate_not_found":  "לא נמצאו נתונים | מלא ידנית",
        "plate_optional":   "אופציונלי | או מלא ידנית למטה",
        "yad2_ref_label":     "השוואת מחירים בשוק",
        "yad2_ref_hint":     "בדוק מה רכבים דומים נמכרים היום ביד2",
        "yad2_ref_btn":      "פתח ביד2 ←",
        "yad2_search_hint":  "חפש ביד2:",
        "yad2_price_label":  "מחירון יד2",
        "yad2_price_range":  "טווח מחיר שוק",
        "yad2_price_model":  "דגם:",
        "yad2_price_note":   "מחיר בסיס לפי מחירון יד2 | לא כולל התאמה לקילומטראז'",
        "yad2_price_na":     "מחירון לא זמין לרכב זה",
        "registry_title":        "נתוני רישוי רשמיים",
        "registry_source":       "מקור: משרד התחבורה הישראלי",
        "ownership_type":        "סוג בעלות",
        "ownership_private":     "פרטי",
        "ownership_lease":       "ליסינג",
        "ownership_rental":      "השכרה",
        "ownership_govt":        "ממשלתי",
        "ownership_company":     "עסקי / חברה",
        "ownership_warn":        "⚠️ הרכב רשום כ-{type} | אמת מול המוכר לפני רכישה",
        "first_road_reg":        "עלייה לכביש",
        "last_inspection":       "טסט אחרון",
        "reg_valid_until":       "רישיון בתוקף עד",
        "car_color":             "צבע",
        "fuel_type":             "סוג דלק",
        "no_plate_data":         "לא בוצעה בדיקת לוחית | הזן לוחית רישוי בפרטי הרכב לאימות בעלות",
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
        "select_make":      "| בחר יצרן |",
        "select_model":     "| בחר דגם |",
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
        "underbody_title":  "תמונת תחתית הרכב | אופציונלי",
        "underbody_hint":   "תמונה של חלק התחתון של הרכב מסייעת לזיהוי דליפות אפשריות.",
        "photos_count":     "תמונות נבחרו",
        "photos_min_error": "אנא העלה לפחות 4 תמונות.",
        "photos_max_error": "מקסימום 10 תמונות מותר.",
        "engine_audio":     "שמע מנוע",
        "audio_hint":       "הקלט לפחות 15 שניות של מנוע פועל בסרק. קרב את המיקרופון לתא המנוע לתוצאות הטובות ביותר. ניתן להעלות גם קובץ וידאו (MP4 וכד׳).",
        "audio_missing":    "אנא העלה הקלטת שמע של המנוע.",
        "analyse_btn":      "נתח רכב  ←",
        "analysing":        "מנתח | זה עשוי לקחת רגע ...",
        "analysis_failed":  "הניתוח נכשל",
        "confidence_label": "מהימנות האבחון",
        "conf_ctx_go":      "מערכת מזהה סיכוי {} שהרכב במצב תקין",
        "conf_ctx_nogo":    "זוהו בעיות | רמת ודאות {}. מומלץ לבדוק לפני רכישה",
        "conf_ctx_inc":     "נדרש מידע נוסף לאבחון מלא | רמת ודאות {}",
        "audio_analysed":   "שמע: {:.1f} שניות נותחו",
        "findings_title":   "ממצאי הבדיקה",
        "next_steps_title": "צעדים מומלצים",
        "learn_more_title": "קרא עוד",
        "go":               "מתאים",
        "no_go":            "זוהו בעיות",
        "inconclusive":     "מידע חסר",
        "high":             "גבוה",
        "medium":           "בינוני",
        "low":              "נמוך",
        "feat1":            "ניתוח תמונות חכם",
        "feat1_desc":       "זיהוי בעיות חיצוניות ופנימיות",
        "feat2":            "ניתוח קול מנוע",
        "feat2_desc":       "זיהוי רעשים חריגים בזמן מנוע פועל בסרק",
        "feat3":            "פסיקה מיידית | קנה / אל תקנה",
        "feat3_desc":       "GO או NO-GO בהתבסס על כל הנתונים",
        "own_single_good":  "בעלים יחיד במשך {age} שנים | סימן חיובי לתחזוקה טובה",
        "own_red_flag":     "{owners} בעלים ב-{age} שנים בלבד | החלפות תכופות: דגל אדום",
        "own_concern":      "ממוצע {avg:.1f} שנים לבעלים | קצר מהמצופה",
        "own_stable":       "היסטוריית בעלות יציבה | ממוצע {avg:.1f} שנים לבעלים",
        "km_very_low":      "קילומטראז' נמוך מאוד ({km:,} ק\"מ) | שחיקה מינימלית צפויה",
        "km_low":           "קילומטראז' נמוך ({km:,} ק\"מ) | מצב מנוע ורכיבים טוב יותר בממוצע",
        "km_high":          "קילומטראז' גבוה ({km:,} ק\"מ) | שחיקת מנוע גבוהה יותר באופן טבעי",
        "km_very_high":     "קילומטראז' גבוה מאוד ({km:,} ק\"מ) | בלאי משמעותי: מומלץ לבדוק בדחיפות",
        "usage_type":       "סוג שימוש קודם",
        "usage_type_opts":  ["פרטי", "השכרה / ליסינג", "רכב חברה", "לא ידוע"],
        "usage_rental":     "רכב השכרה/ליסינג | שימוש אינטנסיבי ושחיקה מהירה יותר",
        "usage_rental_hi":  "רכב השכרה/ליסינג עם {owners} בעלים | בלאי גבוה: דגל אדום",
        "usage_company":    "רכב חברה | בדוק היסטוריית תחזוקה ורשומות שירות",
        "recalls_title":    "ריקולים ובעיות ידועות",
        "no_recalls_found": "לא נמצאו ריקולים פתוחים",
        "open_recalls":     "ריקולים פתוחים",
        "complaints_title": "תלונות מוכרות",
        "total_complaints": "סך תלונות NHTSA",
        "nhtsa_source":     "מקור: NHTSA | מינהל בטיחות הרכב האמריקאי",
        "audio_diagnosis":       "אבחון קולי | פירוט",
        "vehicle_video":         "סרטון הרכב (אופציונלי)",
        "video_hint":            "הוסף סרטון עד דקה אחת. הAI יזהה שריטות, נזקי גוף, שינויי צבע ודליפות.",
        "interior_photos_title": "תמונות פנים הרכב",
        "interior_photos_hint":  "2–4 תמונות: מושבים, לוח מחוונים, תקרה, רצפה",
        "interior_photos_count": "תמונות פנים נבחרו",
        "exterior_score":        "ציון חיצוני",
        "interior_score":        "ציון פנים",
        "leak_none":             "לא זוהו דליפות",
        "leak_oil":              "חשד לדליפת שמן",
        "leak_water":            "חשד לדליפת מים / קירור",
        "leaks_title":           "בדיקת דליפות",
        "detailed_report":       "דוח מפורט",
        "scores_title":          "ציוני מצב",
        "sec_visual":            "חזות ומעטפת",
        "sec_mechanical":        "בדיקה מכאנית",
        "sec_conclusion":        "סיכום",
        "conc_external_label":   "מצב חיצוני",
        "conc_internal_label":   "מצב פנים",
        "conc_mechanical_label": "מצב מכאני",
        "of_10":            "/ 10",
        "paint_title":      "ניתוח צבע והיסטוריית תאונות/תיקונים",
        "paint_consistent": "הצבע אחיד | לא נמצאו חשדות לצביעה מחדש",
        "paint_suspect":    "חשד לצביעה מחדש | {count} חלק(י) רכב חריגים",
        "paint_panel":      "חלק מרכב",
        "paint_diff":       "חריגת צבע",
        "paint_overspray":  "חשד לאוברספריי",
        "paint_orange_peel":"מרקם אורנג' פיל",
        "paint_flake":      "אי-התאמת ניצוצות מתכת",
        "paint_hue":        "הבדל גוון",
        "paint_gap":        "פערים לא אחידים",
        "paint_severity_high":   "🔴 חשד גבוה לתיקון גוף",
        "paint_severity_medium": "🟡 חשד בינוני לצביעה מחדש",
        "paint_severity_low":    "🟢 עקביות צבע תקינה",
        "paint_note":       "ניתוח מבוסס השוואת היסטוגרמת צבע LAB בין חלקי הרכב + בדיקה ויזואלית של AI",
        "action_label":     "המלצת פעולה",
        "action_green":     "הרכב נראה תקין בבדיקה הוויזואלית | לא זוהו בעיות צבע, דליפות או רעשי מנוע. ייתכן שמדובר ברכישה טובה, אך חובה להגיע לבדיקת רכב מוסמכת לפני חתימה על כל עסקה.",
        "action_yellow":    "זוהה סיכון קולי במנוע. לא זוהו בעיות צבע או דליפות | ייתכן שמדובר ברכב שניתן לתקן. חובה להגיע לבדיקת רכב מוסמכת לפני כל שיקול רכישה.",
        "action_red":       "זוהו מספר גורמי סיכון: בעיות צבע/גוף ורעשי מנוע חריגים. לא מומלץ לרכוש רכב זה. אין כדאיות לבדיקה טכנית עד שהבעיות הוויזואליות יובהרו ויטופלו.",
        "action_leak":      "זוהתה דליפת נוזל | זהו סימן אזהרה חמור. יש לבדוק במוסך מורשה לפני כל שיקול רכישה. אל תרכוש ללא בדיקה מקיפה.",
        "refine_title":         "שפר את הניתוח",
        "refine_details_btn":   "עדכן פרטי רכב",
        "refine_photos_btn":    "הוסף / החלף תמונות",
        "refine_audio_btn":     "הוסף הקלטת מנוע",
        "view_original_btn":    "דוח מקורי ←",
        "original_report_title":"הניתוח המקורי",
        "refine_banner":        "מצב עדכון | הוסף חומר חדש לניתוח מחודש",
        "prev_photos_kept":     "📂 {n} תמונות מהבדיקה הקודמת שמורות — ניתן להוסיף עוד",
        "img_validation_running": "🔍 בודק תמונות...",
        "img_warn_unrelated":   "⚠️ ייתכן שאחת מהתמונות אינה תמונת רכב. בדוק ונסה שנית.",
        "img_warn_two_vehicles":"⚠️ נראה שהועלו תמונות של שני רכבים שונים! ודא שכל התמונות שייכות לאותו רכב.",
        "img_warn_brand_mismatch": "⚠️ לוגו הרכב בתמונות לא תואם את הדגם שנבחר ({selected}). ייתכן שגיבוב תמונות או לוחית לא תואמת.",
        "no_new_audio_err":     "אנא העלה קובץ שמע חדש להפעלת ניתוח מחודש",
        "back_to_result_btn":   "← חזור לתוצאה",
        "data_quality_title":   "⚠️ הדוח מבוסס על נתונים חלקיים",
        "data_quality_msg":     "חלק מהנתונים לא סופקו. הוספת מידע נוסף תשפר את דיוק הניתוח.",
        "data_missing_interior":"לא הועלו תמונות פנים",
        "data_missing_underbody":"לא הועלה תמונת תחתית הרכב",
        "data_few_exterior":    "מעט תמונות חיצוניות ({n} | מומלץ 6 ומעלה)",
        "data_short_audio":     "הקלטת המנוע קצרה ({s:.0f} שניות | מומלץ 15 שניות לפחות)",
        "data_refine_hint":     "לחץ על כפתורי 'שפר את הניתוח' בתחתית הדף להוספת נתונים",
        "whatsapp_share":       "📲 שתף תוצאה ב-WhatsApp",
        "download_report":      "⬇️ הורד דוח",
        "damage_history_title": "היסטוריית נזקים ותאונות",
        "damage_history_hint":  "לבדיקה מקיפה של היסטוריית תאונות:",
        "damage_history_external": "בדיקה חיצונית מומלצת",
    },
    "en": {
        "app_title":        "UsedCar Check",
        "app_subtitle":     "The full picture | before you commit",
        "app_subtitle_main":"Smart Analysis. Confident Decision.",
        "email_label":      "Email Address",
        "password_label":   "Access Code",
        "enter_btn":        "Enter",
        "email_hint":       "Your email is used to access your check history.",
        "disclaimer":       "This website provides basic assistance only to used car buyers. The user bears sole and exclusive responsibility for the decision to purchase any vehicle. The website and its developers bear no responsibility whatsoever for any wrong decision or damage resulting from use of this service. The final decision regarding a vehicle purchase must be based on a written professional opinion from a certified mechanic, or following inspection at an authorized vehicle testing facility.",
        "disclaimer_accept":"By entering your email you accept the website's terms of use.",
        "invalid_code":     "Invalid access code.",
        "invalid_email":    "Please enter a valid email address.",
        "plate_label":      "License Plate",
        "plate_hint":       "Enter plate number to auto-fill details from the Transport Ministry",
        "plate_lookup_btn": "Auto-Fill",
        "plate_found":      "✓ Vehicle details loaded from Transport Authority",
        "plate_not_found":  "Plate not found | please fill in manually",
        "plate_optional":   "Optional | or fill in manually below",
        "yad2_ref_label":    "Market Price Reference",
        "yad2_ref_hint":    "See what similar cars are selling for on Yad2",
        "yad2_ref_btn":     "Open on Yad2 →",
        "yad2_search_hint": "Search on Yad2:",
        "yad2_price_label": "Yad2 Pricelist",
        "yad2_price_range": "Market Price Range",
        "yad2_price_model": "Model:",
        "yad2_price_note":  "Base price from Yad2 pricelist | not adjusted for mileage",
        "yad2_price_na":    "Pricelist not available for this vehicle",
        "registry_title":        "Official Registry Data",
        "registry_source":       "Source: Israeli Ministry of Transport",
        "ownership_type":        "Ownership Type",
        "ownership_private":     "Private",
        "ownership_lease":       "Leasing",
        "ownership_rental":      "Rental",
        "ownership_govt":        "Government",
        "ownership_company":     "Business / Company",
        "ownership_warn":        "⚠️ This vehicle is registered as {type} | verify with the seller before purchase",
        "first_road_reg":        "First on Road",
        "last_inspection":       "Last Inspection",
        "reg_valid_until":       "License Valid Until",
        "car_color":             "Color",
        "fuel_type":             "Fuel Type",
        "no_plate_data":         "No plate lookup performed | enter a license plate number to verify ownership",
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
        "select_make":      "| Select Make |",
        "select_model":     "| Select Model |",
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
        "underbody_title":  "Underbody Photo | Optional",
        "underbody_hint":   "A photo of the underside of the vehicle helps detect possible fluid leaks.",
        "photos_count":     "photo(s) selected",
        "photos_min_error": "Please upload at least 4 photos.",
        "photos_max_error": "Maximum 10 photos allowed.",
        "engine_audio":     "Engine Audio",
        "audio_hint":       "Record at least 15 seconds of the engine running at idle. Hold the microphone near the engine bay for best results. You can also upload a video file (MP4, etc.).",
        "audio_missing":    "Please upload an engine audio recording.",
        "analyse_btn":      "Analyse Vehicle  →",
        "analysing":        "Analysing | this may take a moment …",
        "analysis_failed":  "Analysis failed",
        "confidence_label": "Diagnostic Reliability",
        "conf_ctx_go":      "System detects {} chance the vehicle is in good condition",
        "conf_ctx_nogo":    "Issues detected | {} confidence. Inspection recommended before purchase",
        "conf_ctx_inc":     "More data needed for a full diagnosis | certainty level {}",
        "audio_analysed":   "Audio: {:.1f}s analysed",
        "findings_title":   "Assessment Findings",
        "next_steps_title": "Recommended Next Steps",
        "learn_more_title": "Learn More",
        "go":               "GO",
        "no_go":            "RISK DETECTED",
        "inconclusive":     "INSUFFICIENT DATA",
        "high":             "HIGH",
        "medium":           "MEDIUM",
        "low":              "LOW",
        "feat1":            "Smart Photo Analysis",
        "feat1_desc":       "Detect exterior & interior issues",
        "feat2":            "Engine Sound Analysis",
        "feat2_desc":       "Identify abnormal sounds at idle",
        "feat3":            "Instant GO / NO-GO Verdict",
        "feat3_desc":       "Clear decision based on all data points",
        "own_single_good":  "Single owner for {age} years | positive sign of good maintenance",
        "own_red_flag":     "{owners} owners in just {age} years | frequent changes are a red flag",
        "own_concern":      "Average {avg:.1f} years per owner | shorter than expected",
        "own_stable":       "Stable ownership history | average {avg:.1f} years per owner",
        "km_very_low":      "Very low mileage ({km:,} km) | minimal wear expected",
        "km_low":           "Low mileage ({km:,} km) | engine and components likely in better than average condition",
        "km_high":          "High mileage ({km:,} km) | naturally higher engine wear",
        "km_very_high":     "Very high mileage ({km:,} km) | significant wear: urgent inspection strongly advised",
        "usage_type":       "Prior Usage Type",
        "usage_type_opts":  ["Private", "Rental / Lease", "Company Car", "Unknown"],
        "usage_rental":     "Rental/lease vehicle | typically higher wear and intensive use",
        "usage_rental_hi":  "Rental/lease with {owners} owners | high wear: red flag",
        "usage_company":    "Company car | verify service history and maintenance records",
        "recalls_title":    "Recalls & Known Issues",
        "no_recalls_found": "No open recalls found",
        "open_recalls":     "Open Recalls",
        "complaints_title": "Reported Complaints",
        "total_complaints": "Total NHTSA Complaints",
        "nhtsa_source":     "Source: NHTSA | US National Highway Traffic Safety Administration",
        "audio_diagnosis":       "Audio Diagnosis | Detail",
        "vehicle_video":         "Vehicle Video (optional)",
        "video_hint":            "Upload up to 1 minute of video. AI will detect scratches, body damage, color changes and leaks.",
        "interior_photos_title": "Interior Photos",
        "interior_photos_hint":  "2–4 photos: seats, dashboard, ceiling, floor",
        "interior_photos_count": "interior photo(s) selected",
        "exterior_score":        "Exterior Score",
        "interior_score":        "Interior Score",
        "leak_none":             "No leaks detected",
        "leak_oil":              "Oil leak suspected",
        "leak_water":            "Water / coolant leak suspected",
        "sec_visual":            "Visual Assessment",
        "sec_mechanical":        "Mechanical Check",
        "sec_conclusion":        "Summary",
        "conc_external_label":   "Exterior",
        "conc_internal_label":   "Interior",
        "conc_mechanical_label": "Mechanical",
        "leaks_title":      "Leak Assessment",
        "detailed_report":  "Detailed Report",
        "scores_title":     "Condition Scores",
        "of_10":            "/ 10",
        "paint_title":      "Paint Analysis & Accident / Repair History",
        "paint_consistent": "Paint is consistent | no repaint indicators found",
        "paint_suspect":    "Repaint suspected | {count} anomalous panel(s)",
        "paint_panel":      "Panel",
        "paint_diff":       "Color deviation",
        "paint_overspray":  "Overspray suspected",
        "paint_orange_peel":"Orange peel texture",
        "paint_flake":      "Metallic flake mismatch",
        "paint_hue":        "Hue shift",
        "paint_gap":        "Uneven panel gaps",
        "paint_severity_high":   "🔴 High suspicion of body repair",
        "paint_severity_medium": "🟡 Moderate repaint suspicion",
        "paint_severity_low":    "🟢 Paint consistency normal",
        "paint_note":       "Based on LAB color histogram comparison between panels + AI visual inspection",
        "action_label":     "Action Recommendation",
        "action_green":     "The vehicle looks clean visually | no paint issues, leaks or engine problems detected. This could be a good buy, but an official inspection at a certified mechanic is mandatory before signing any deal.",
        "action_yellow":    "Engine noise risk detected. No paint or leak issues found | the car may be fixable. You must take it to a certified inspection center before making any purchase decision.",
        "action_red":       "Multiple risk factors detected: paint/body issues and abnormal engine sounds. We recommend avoiding this vehicle. There is no point taking it to a test center until the visual issues are fully investigated.",
        "action_leak":      "Fluid leak detected | this is a serious warning sign. Have the vehicle inspected at a certified garage before any purchase consideration. Do not buy without a full mechanical check.",
        "refine_title":         "Refine Analysis",
        "refine_details_btn":   "Update Vehicle Details",
        "refine_photos_btn":    "Add / Replace Photos",
        "refine_audio_btn":     "Add Engine Recording",
        "view_original_btn":    "Original Report ←",
        "original_report_title":"Original Analysis",
        "refine_banner":        "Update Mode | upload new material for a fresh analysis",
        "prev_photos_kept":     "📂 {n} photos from previous check saved — you can add more",
        "img_validation_running": "🔍 Checking photos...",
        "img_warn_unrelated":   "⚠️ One or more photos may not be a car image. Please review and re-upload.",
        "img_warn_two_vehicles":"⚠️ Photos appear to show two different vehicles! Make sure all photos are of the same car.",
        "img_warn_brand_mismatch": "⚠️ The car brand visible in the photos does not match the selected model ({selected}). Possible photo mix-up or plate mismatch.",
        "no_new_audio_err":     "Please upload new audio to run a refreshed analysis",
        "back_to_result_btn":   "Back to Result →",
        "data_quality_title":   "⚠️ Report is based on incomplete data",
        "data_quality_msg":     "Some data was not provided. Adding more material will improve the accuracy of the analysis.",
        "data_missing_interior":"No interior photos uploaded",
        "data_missing_underbody":"No underbody photo uploaded",
        "data_few_exterior":    "Few exterior photos ({n} | 6+ recommended)",
        "data_short_audio":     "Engine audio is short ({s:.0f}s | 15+ seconds recommended)",
        "data_refine_hint":     "Use the 'Refine Analysis' buttons at the bottom of this page to add more data",
        "whatsapp_share":       "📲 Share result on WhatsApp",
        "download_report":      "⬇️ Download Report",
        "damage_history_title": "Damage & Accident History",
        "damage_history_hint":  "For full accident history:",
        "damage_history_external": "Recommended external check",
    }
}

# ─── Reject code table ────────────────────────────────────────────────────────────────────────────────
REJECT_TABLE: dict[str, dict] = {
    "R01": {"severity": "hard",
            "title_he": "נזק גוף חמור",
            "title_en": "Severe body damage",
            "expl_he":  "זוהה נזק משמעותי לגוף הרכב. הרכב לא מתאים לרכישה ללא בדיקה מקצועית.",
            "expl_en":  "Significant exterior damage was detected. This vehicle should not move forward without professional inspection."},
    "R02": {"severity": "soft",
            "title_he": "חוסר עקביות בצבע / חלקי רכב",
            "title_en": "Paint or panel mismatch",
            "expl_he":  "הרכב מראה סימני צביעה מחדש או החלפת חלק מרכב, שעשויים להעיד על נזק קודם.",
            "expl_en":  "The car shows signs of repainting or panel replacement, which may indicate previous accident damage."},
    "R03": {"severity": "hard",
            "title_he": "חשד לדליפת נוזלים",
            "title_en": "Possible fluid leak",
            "expl_he":  "זוהו עקבות לחות, כתמים או שאריות בתחתית הרכב / תא המנוע. ייתכן שקיימת בעיה מכאנית הדורשת בדיקה.",
            "expl_en":  "Possible fluid leakage was detected. This may point to a mechanical issue that requires inspection."},
    "R04": {"severity": "soft",
            "title_he": "חלקים חיצוניים שבורים / חסרים",
            "title_en": "Broken or missing exterior parts",
            "expl_he":  "חלק אחד או יותר של המעטפת החיצונית נראה פגום או חסר. עלויות התיקון עשויות להיות גבוהות מהצפוי.",
            "expl_en":  "One or more exterior parts appear damaged or missing. Repair costs may be higher than expected."},
    "R05": {"severity": "soft",
            "title_he": "מצב צמיגים / גלגלים",
            "title_en": "Tire or wheel condition concern",
            "expl_he":  "מצב הצמיגים או הגלגלים מעלה חשש. הדבר עלול להשפיע על הבטיחות ולהעיד על בעיית ישור או מתלה.",
            "expl_en":  "Tire or wheel condition raises concern. This can affect safety and may indicate alignment or suspension issues."},
    "R06": {"severity": "soft",
            "title_he": "בעיית תא מנוע",
            "title_en": "Engine bay concern",
            "expl_he":  "תא המנוע מראה סימני הזנחה, חלקים מנותקים, קורוזיה, או מצב לא תקין.",
            "expl_en":  "The engine bay shows signs that may indicate poor maintenance or a potential mechanical problem."},
    "R07": {"severity": "soft",
            "title_he": "מצב פנים מתחת לצפוי",
            "title_en": "Cabin condition below expected",
            "expl_he":  "מצב הפנים נמוך באופן ניכר מהצפוי ועשוי להעיד על תחזוקה ירודה.",
            "expl_en":  "The interior condition appears significantly below normal and may reflect poor overall vehicle care."},
    "R08": {"severity": "hard",
            "title_he": "חדירת מים / לחות",
            "title_en": "Water or moisture intrusion",
            "expl_he":  "זוהו סימני לחות או חדירת מים בתא הנוסעים. מדובר בליקוי חמור הדורש בדיקה מיידית.",
            "expl_en":  "Signs of moisture or previous water intrusion were detected inside the vehicle."},
    "R09": {"severity": "tech",
            "title_he": "איכות מדיה ירודה",
            "title_en": "Low media quality",
            "expl_he":  "התמונות/וידאו שהועלו אינם ברורים מספיק להערכה אמינה. אנא העלה תמונות טובות יותר.",
            "expl_en":  "The uploaded media is not clear enough for a reliable evaluation. Please upload better images or video."},
    "R10": {"severity": "tech",
            "title_he": "כיסוי חיצוני לא מספיק",
            "title_en": "Insufficient exterior coverage",
            "expl_he":  "אין מספיק תמונות חיצוניות להערכה מלאה של הרכב.",
            "expl_en":  "There is not enough exterior coverage to evaluate the vehicle properly."},
    "R11": {"severity": "tech",
            "title_he": "אותות ויזואליים סותרים",
            "title_en": "Conflicting visual signals",
            "expl_he":  "המערכת מצאה סימנים ויזואליים סותרים | לא ניתן להגיע לתוצאה אמינה מהמדיה הנוכחית.",
            "expl_en":  "The system found unclear or conflicting visual signs, so a reliable result cannot be given from the current media."},
    "R12": {"severity": "hard",
            "title_he": "ממצאים מרובים בסיכון גבוה | נדרשת בדיקה דחופה",
            "title_en": "Multiple high-risk findings | immediate professional inspection",
            "expl_he":  "זוהו מספר ממצאים בעלי סיכון גבוה. הרכב צריך לעבור בדיקה מקצועית בלבד.",
            "expl_en":  "Multiple high-risk issues were detected. This vehicle should only proceed with a professional inspection."},
}

_REJECT_HARD = {"R01", "R03", "R08", "R12"}
_REJECT_SOFT = {"R02", "R04", "R05", "R06", "R07"}
_REJECT_TECH = {"R09", "R10", "R11"}

_ACTIVE_LANG: str = "he"   # set before threads; fallback when session_state unavailable

_PORSCHE_B64: str = "iVBORw0KGgoAAAANSUhEUgAAAhMAAAEsCAYAAAB5ZzHBAAEAAElEQVR42uz92a9tWXrdif3mnKvb3enPPbePPiIzIjMjsmN2ylTRolUluGQBdgmwoXqwDBgwbD8U/OIHw/+EBcNNuQooGRJcRasXKYpiqSQmRTIzmW00GU1Gc+P29/Rnd6uZjR/mXGvvfe69kZEkk0wy5whcnDjdPnuvvdaaY37f+MYQzjkiIiIiIiIiIv64kPEQREREREREREQyERERERERERHJREREREREREQkExERERERERGRTERERERERERERDIREREREREREclERERERERERCQTEREREREREZFMREREREREREREMhERERERERERyUREREREREREJBMRERERERERkUxEREREREREREQyERERERERERHJREREREREREQkExERERERERGRTERERERERERERDIREREREREREclERERERERERCQTEREREREREZFMREREREREREQyERERERERERERyUREREREREREJBMRERERERERkUxERERERERERDIRERERERERERHJREREREREREQkExERERERERGRTEREREREREREMhEREREREREREclERERERERERCQTEREREREREZFMREREREREREQyERERERERERERyUREREREREREJBMRERERERERkUxERERERERERDIREREREREREclEREREREREREQkExERERERERGRTEREREREREREMhERERERERERyURERERERERERCQTEREREREREZFMREREREREREQyERERERERERHJREREREREREREJBMRERERERERkUxERERERERERDIREREREREREclEREREREREREQkExERERERERGRTEREREREREREMhERERERERERyURERERERERERCQTEREREREREZFMREREREREREQyERERERERERHJREREREREREQkExERERERERERkUxERERERERERDIREREREREREclERERERERERCQTERERERERERGRTEREREREREREMhERERERERERyUREREREREREJBMREREREREREZFMREREREREREQyERERERERERHJREREREREREQkExERERERERERkUxERERERERERDIREREREREREclERERERERERCQTEREREREREZFMREREREREREREMhERERERERERyUREREREREREJBMRERERERERkUxEREREREREREQyERERERERERHJREREREREREQkExERERERERGRTERERERERERERDIREREREREREclERERERERERCQTEREREREREZFMREREREREREREMhERERERERERyUREREREREREJBMRERERERERkUxERERERERERDIRERERERERERHJRERERERERMSfD5J4CCIi/ozgnP/wx/rdj/lz4tzviJ/9T3W/IkT3Z0V89yIiIj7qvuGci0chIuJnhXXdKusetWgLsAjkX+iF2OHcMo9xCER4PWLpdT5MRgQsvhHrnxERkUxERPyycwaHW1kPBeLjMwTnH6RpDE1d01QVtqqoygpdN043hrIq0U2DMw6tG7TRJIB1FmcdDtcxFun8M7EShPDLtpBh5RYWrMBZt3SBhwKFlAglEVIghECpBCkFqUpI85w0z8h7uch7OflgQJrnyESBUh9xcMAIGwogAoToDouMZCIiIpKJiIhfCri2suBWvuYX6Z9SUtAWZy3GGMr5nGpeMp/NmE9nrpnOmI9n1GVDXZXoWmONpi4rXKNJlAwPLUnSlCRJQPqWQqISsjRBCZBSkKQpDpBKgZIgEoSUIAVIgUgShBAIHBLrqwnG+oXcWZwxWG2wzvm/LwRWa5qmodYaozXGGhrd0GiNEBKZSFSqyPKc3qBPf9hnuD6iNxoxXFsTeb9PWvRQeeafw9LhbP8J50KF4mcgXhEREZFMRET8ohMHd0544JxDysdvm02jaaqSuqyppjMmp2duNp0xm0ypplN03WAbA8ZimgZrLYmF1EmUlCRFQdrrkQx79EdD+kWPJEvI8lwopbCBtDjnQEiwnqCgrXPGILBI6xDOgvUtB2uXyI8UftsvJEI4hBA4IXwVQiiUkAipkIkUqNR/L0sgkBEhBc4XFHBYtK7RTU1VlkxPTt389IzJ6Rln41OMacCCQpIkCSJRJMMeg811dvf2GG6si7XRCFXknvTIlWZJ1yvpiFpEREQkExERv8gix/Nn9EctYLqsKeczZpMZs8nElfMp89mc6XjC/HiMnjZYrRFSkOU5g0GPfn+IyDPyPCeRAqkUSZKghCQVkgSJMJ4YWGtptMHoxpOTxj9eXdU4C84atNVhRXc4a9HadK9FOOerDNZ6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"

def _get_lang() -> str:
    """Thread-safe lang getter — uses session_state when available, else module fallback."""
    try:
        return st.session_state.get("lang", _ACTIVE_LANG)
    except Exception:
        return _ACTIVE_LANG

def t(key: str) -> str:
    return TR[_get_lang()].get(key, key)

# ─── Backend → Hebrew translation lookup (all known English strings) ──────────
_HE_STRINGS = {
    # Backend decide() reasons
    "Some photos were too blurry or poorly lit to assess":
        "חלק מהתמונות היו מטושטשות מדי או עם תאורה ירודה לאבחון",
    "Engine audio suggests an unstable/rough pattern":
        "קול המנוע מצביע על דפוס לא יציב / גס",
    "High driven KM means engine wear risk is naturally higher":
        "קילומטראז' גבוה | שחיקת מנוע גבוהה יותר באופן טבעי",
    "Dashboard warning indicator may be present":
        "ייתכן שנוכח נורת אזהרה בלוח המחוונים",
    "Underbody image shows possible fluid-stain pattern":
        "תמונת תחתית הרכב מראה דפוס אפשרי של דליפת נוזל",
    "No clear red flags detected in this quick check":
        "לא זוהו דגלים אדומים ברורים בבדיקה המהירה",
    # Backend next_steps
    "Retake at least 2 photos in brighter light and keep the phone steady.":
        "צלם מחדש לפחות 2 תמונות באור טוב יותר עם מצלמה יציבה.",
    "Check warning lights with OBD scan before any purchase decision.":
        "בדוק נורות אזהרה עם סורק OBD לפני כל החלטת רכישה.",
    "If you proceed, ask for an inspection focused on engine idle quality and scan for codes.":
        "אם תמשיך, בקש בדיקה הממוקדת באיכות סרק המנוע וסריקת קודים.",
    "A professional inspection is still recommended before buying.":
        "עדיין מומלץ לבצע בדיקה מקצועית לפני הרכישה.",
    # Backend education
    "What a rough idle can indicate":
        "מה יכולה להצביע ריצת סרק גסה",
    "Best photo angles for a used-car check":
        "זוויות צילום מומלצות לבדיקת רכב משומש",
}

# Audio finding label → Hebrew display text
_AUDIO_LABELS_HE = {
    "rod_knock_suspected":    "חשד לדפיקות מנוע (מוטות/בוכנות)",
    "valve_tick_suspected":   "חשד לרעש שסתומים / תפטירים",
    "belt_squeal_suspected":  "חשד לחריקת רצועה",
    "exhaust_leak_suspected": "חשד לדליפת פליטה",
    "rough_idle_suspected":   "חשד לסרק לא יציב",
    "engine_sounds_normal":   "קול המנוע תקין",
    "unknown":                "ממצא לא מזוהה",
}

def _tr_backend(text: str) -> str:
    """Translate a known backend English string to current UI language."""
    if _get_lang() == "he":
        return _HE_STRINGS.get(text, text)
    return text

# Reverse lookup: Hebrew → English (for render-time correction of saved results)
_EN_STRINGS = {v: k for k, v in _HE_STRINGS.items()}
# Add hardcoded Hebrew strings used in _mech_issues / _urgent_steps
_EN_STRINGS.update({
    "חשד לדליפת שמן":                       "Oil / fluid leak detected in underbody image",
    "דפיקות מנוע | נדרשת בדיקה דחופה":      "Engine knock detected | urgent inspection required",
    "רעש חריג זוהה בהקלטת המנוע | מומלץ לבדוק": "Abnormal engine noise detected | inspection recommended",
    "דליפת שמן/נוזל זוהתה | יש לבדוק במוסך מורשה לפני כל שיקול של רכישה.":
        "Fluid leak detected | have the vehicle inspected by a certified mechanic before any purchase decision.",
    "זוהו סימנים לדפיקות מנוע | נדרשת בדיקת לחץ צילינדרים ומסב בדחיפות.":
        "Engine knock detected | compression test and bearing inspection required urgently.",
    "זוהה רעש חריג במנוע | מומלץ לבדוק שסתומים ורצועות הנע לפני רכישה.":
        "Abnormal engine noise detected | have valves and drive belts checked before purchase.",
    "עדיין מומלץ לבצע בדיקה מקצועית לפני הרכישה.":
        "A professional inspection is still recommended before buying.",
})

import re as _re
# Regex patterns for dynamic Hebrew ownership/usage strings → English
_HE_DYNAMIC_TO_EN = [
    (_re.compile(r"בעלים יחיד במשך ([\d.]+) שנ"),
     lambda m: f"Single owner for {m.group(1)} years | good sign for maintenance"),
    (_re.compile(r"([\d]+) בעלים ב-([\d]+) שנ"),
     lambda m: f"{m.group(1)} owners in {m.group(2)} years | red flag!"),
    (_re.compile(r"החלפת בעלים תכופה.*ממוצע ([\d.]+)"),
     lambda m: f"Frequent ownership changes | average {m.group(1)} years per owner"),
    (_re.compile(r"היסטוריית בעלות יציבה.*ממוצע ([\d.]+)"),
     lambda m: f"Stable ownership history | average {m.group(1)} years per owner"),
    (_re.compile(r"רכב השכרה.*שימוש אינטנסיבי"),
     lambda m: "Rental/lease vehicle | intensive use and higher wear"),
    (_re.compile(r"רכב חברה.*בלאי מואץ"),
     lambda m: "Company car | accelerated wear expected"),
]
# Regex patterns for dynamic English ownership/usage strings → Hebrew
_EN_DYNAMIC_TO_HE = [
    (_re.compile(r"Single owner for ([\d.]+) years"),
     lambda m: f"בעלים יחיד במשך {m.group(1)} שנים | סימן חיוב לתחזוקה טובה"),
    (_re.compile(r"([\d]+) owners in ([\d]+) years"),
     lambda m: f"{m.group(1)} בעלים ב-{m.group(2)} שנים | דגל אדום!"),
    (_re.compile(r"Frequent ownership.*average ([\d.]+)"),
     lambda m: f"החלפת בעלים תכופה | ממוצע {m.group(1)} שנים לבעלים"),
    (_re.compile(r"Stable ownership.*average ([\d.]+)"),
     lambda m: f"היסטוריית בעלות יציבה | ממוצע {m.group(1)} שנים לבעלים"),
]

def _tr_for_display(text: str) -> str:
    """Bidirectional translation for render-time: ensures text matches current UI language.
    Corrects saved results that were stored in the opposite language."""
    lang = _get_lang()
    if lang == "he":
        result = _HE_STRINGS.get(text, None)
        if result:
            return result
        for pat, fn in _EN_DYNAMIC_TO_HE:
            m = pat.search(text)
            if m:
                return fn(m)
        return text
    else:
        result = _EN_STRINGS.get(text, None)
        if result:
            return result
        for pat, fn in _HE_DYNAMIC_TO_EN:
            m = pat.search(text)
            if m:
                return fn(m)
        return text

def _audio_label(lbl: str) -> str:
    """Return display label for an audio finding in current UI language."""
    if _get_lang() == "he":
        return _AUDIO_LABELS_HE.get(lbl, lbl.replace("_", " ").title())
    return lbl.replace("_", " ").title()

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
    "Honda":         ["Civic", "Civic Type R", "Accord", "Jazz", "HR-V", "CR-V", "ZR-V", "Pilot", "Fit", "City", "Stream", "FR-V", "CR-Z", "Legend", "e:Ny1", "e:HEV"],
    "Hyundai":       ["i10", "i20", "i20N", "i30", "i30N", "i40", "i45", "ix35", "ix55", "Bayon", "Tucson", "Santa Fe", "Kona", "Ioniq", "Ioniq 5", "Ioniq 6", "Ioniq 9", "Elantra", "Sonata", "Grandeur", "Accent", "Getz", "Matrix", "Trajet", "Staria", "Creta", "Venue", "Nexo", "H-1"],
    "Infiniti":      ["Q30", "Q50", "Q60", "Q70", "QX30", "QX50", "QX60", "QX70", "QX80"],
    "Isuzu":         ["D-Max", "MU-X", "Trooper", "Rodeo", "KB"],
    "Jaguar":        ["XE", "XF", "XJ", "F-Pace", "E-Pace", "I-Pace", "F-Type"],
    "Jeep":          ["Wrangler", "Cherokee", "Grand Cherokee", "Grand Cherokee L", "Renegade", "Compass", "Gladiator", "Commander"],
    "Kia":           ["Picanto", "Morning", "Rio", "Ceed", "Ceed GT", "Ceed SW", "ProCeed", "Cerato", "Stonic", "Sportage", "Sorento", "Mohave", "Niro", "EV3", "EV5", "EV6", "EV9", "Carnival", "Telluride", "Seltos", "XCeed", "Soul", "Carens", "K5", "Stinger"],
    "Land Rover":    ["Defender", "Discovery", "Discovery Sport", "Freelander", "Range Rover", "Range Rover Sport", "Range Rover Evoque", "Range Rover Velar"],
    "Lexus":         ["CT", "IS", "ES", "GS", "LS", "UX", "NX", "RX", "GX", "LX", "LC", "RC"],
    "Mazda":         ["Mazda2", "Mazda3", "Mazda6", "CX-3", "CX-30", "CX-5", "CX-50", "CX-60", "CX-70", "CX-80", "CX-90", "MX-5", "MX-30", "BT-50", "RX-8"],
    "Mercedes-Benz": ["A-Class", "B-Class", "C-Class", "CLA", "CLS", "E-Class", "EQA", "EQB", "EQC", "EQE", "EQS", "G-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "S-Class", "SL", "AMG GT"],
    "MG":            ["MG3", "MG4", "MG5", "HS", "ZS", "ZS EV", "Marvel R", "Cyberster"],
    "Mini":          ["Cooper", "Hatch", "Convertible", "Clubman", "Countryman", "Paceman", "Coupe", "Roadster", "Electric", "Aceman"],
    "Mitsubishi":    ["Colt", "Lancer", "Lancer Evolution", "Eclipse Cross", "ASX", "Outlander", "Outlander PHEV", "Pajero", "Pajero Sport", "L200", "Galant", "Carisma", "Space Star", "Grandis"],
    "Nissan":        ["Micra", "Juke", "Qashqai", "X-Trail", "Leaf", "Ariya", "Navara", "Murano", "Pathfinder", "Kicks", "Note", "Pulsar", "Sentra", "Almera", "Altima", "Primera", "Tiida", "370Z", "GT-R", "Patrol"],
    "Opel":          ["Corsa", "Astra", "Insignia", "Mokka", "Crossland", "Grandland", "Zafira", "Meriva", "Adam", "Ampera", "Combo"],
    "Peugeot":       ["107", "108", "207", "208", "307", "308", "407", "508", "2008", "3008", "4008", "5008", "Landtrek", "Partner"],
    "Polestar":      ["Polestar 2", "Polestar 3", "Polestar 4"],
    "Porsche":       ["911", "Boxster", "Cayman", "Cayenne", "Macan", "Panamera", "Taycan"],
    "Renault":       ["Twingo", "Clio", "Captur", "Megane", "Arkana", "Kadjar", "Austral", "Koleos", "Duster", "Laguna", "Scenic", "Talisman", "Zoe", "Kangoo"],
    "Seat":          ["Mii", "Ibiza", "Arona", "Leon", "Tarraco", "Ateca", "Alhambra", "Altea"],
    "Škoda":         ["Fabia", "Rapid", "Scala", "Octavia", "Superb", "Kamiq", "Karoq", "Kodiaq", "Enyaq", "Citigo", "Yeti", "Roomster"],
    "Smart":         ["Fortwo", "Forfour", "#1", "#3"],
    "SsangYong":     ["Tivoli", "Korando", "Rexton", "Musso", "Rodius", "XLV"],
    "Subaru":        ["Impreza", "Legacy", "Outback", "Forester", "XV", "Crosstrek", "WRX", "WRX STI", "BRZ", "Solterra", "Levorg", "Tribeca"],
    "Suzuki":        ["Alto", "Swift", "Baleno", "Ignis", "Celerio", "SX4", "S-Cross", "Vitara", "Grand Vitara", "Jimny", "Kizashi"],
    "Tesla":         ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
    "Toyota":        ["Aygo", "Aygo X", "Yaris", "Yaris Cross", "GR Yaris", "Corolla", "Corolla Cross", "Camry", "Crown", "Prius", "Prius Plus", "RAV4", "C-HR", "Auris", "Avensis", "Verso", "Land Cruiser", "Land Cruiser Prado", "Hilux", "Fortuner", "Rush", "Urban Cruiser", "Proace", "GR86", "Supra", "bZ4X", "Alphard", "Granvia", "Sienna", "Sequoia", "4Runner"],
    "Volkswagen":    ["Up!", "Polo", "Golf", "Golf Plus", "Jetta", "Passat", "Arteon", "T-Cross", "T-Roc", "Tiguan", "Tiguan Allspace", "Touareg", "Teramont", "ID.3", "ID.4", "ID.5", "ID.7", "Caddy", "Touran", "Sharan", "Amarok"],
    "Volvo":         ["V40", "V60", "V90", "S60", "S90", "XC40", "XC60", "XC90", "C40"],
    "Other":         ["Other / לא ברשימה"],
}
import unicodedata as _ud
MAKES_LIST = [""] + sorted(
    (k for k in CAR_MAKES_MODELS if k != "Other"),
    key=lambda x: _ud.normalize("NFD", x).lower()
) + ["Other"]

# ─── Constants ────────────────────────────────────────────────────────────────
# Password removed — access is open, email-only
DATA_DIR     = _ROOT / "data"
USERS_FILE        = DATA_DIR / "users.json"
MAILING_LIST_FILE = DATA_DIR / "mailing_list.csv"
CHECKS_DIR        = DATA_DIR / "checks"
DATA_DIR.mkdir(exist_ok=True)
CHECKS_DIR.mkdir(exist_ok=True)

def _backfill_mailing_list():
    """On first run, populate mailing_list.csv from existing users.json."""
    if MAILING_LIST_FILE.exists():
        return   # already exists — nothing to backfill
    if not USERS_FILE.exists():
        return
    try:
        import csv as _csv
        _users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        with MAILING_LIST_FILE.open("w", encoding="utf-8", newline="") as _fh:
            _w = _csv.writer(_fh)
            _w.writerow(["email", "first_seen"])
            for _em, _info in _users.items():
                _w.writerow([_em, _info.get("created_at", "")])
    except Exception:
        pass

_backfill_mailing_list()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UsedCar Check",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",   # collapses automatically on mobile
)

# ─── Session state init ───────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "email":         "",
    "step":          1,
    "car_details":   {},
    "photos":           [],
    "interior_photos":  [],
    "underbody":        None,
    "vehicle_video": None,
    "audio":          None,
    "result":         None,
    "original_result":None,
    "refine_mode":    False,
    "lang":           "he",
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
    white-space: nowrap !important;
    transition: opacity 0.2s ease !important;
}}
.stButton > button:hover {{ opacity: 0.82 !important; }}

/* ── All buttons: never wrap text ─────────────────────────── */
/* ── Plate row: input same height as button, both bottom-aligned ─ */
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
    align-items: flex-end !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input {{
    height: 2.75rem !important;
    font-size: 1rem !important;
    padding: 0 0.75rem !important;
}}

/* ── Flag language toggle (st.radio) ─────────────────────── */
div[data-testid="stRadio"] > div {{
    gap: 6px !important;
    align-items: center !important;
}}
div[data-testid="stRadio"] label {{
    padding: 3px 6px !important;
    border-radius: 6px !important;
    border: 2px solid transparent !important;
    cursor: pointer !important;
    transition: border-color 0.15s !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    border-color: var(--gold) !important;
    background: rgba(200,169,106,0.12) !important;
}}
div[data-testid="stRadio"] label > div:first-child {{
    display: none !important;
}}
div[data-testid="stRadio"] label > div:last-child p {{
    font-size: 1.55rem !important;
    line-height: 1 !important;
    margin: 0 !important;
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

/* ── Hide Streamlit "Press Enter to apply" hint — all known DOM variants ── */
.stTextInput small,
.stTextInput p small,
.stTextInput div small,
.stNumberInput small,
small[data-testid="InputInstructions"],
div[data-testid="InputInstructions"],
p[data-testid="InputInstructions"],
[class*="InputInstructions"],
[class*="inputInstructions"],
[class*="input-instructions"],
[data-testid="stTextInput"] ~ small,
[data-testid="stTextInput"] small {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}}

/* ── Force Streamlit input labels to stay ABOVE the field, never inside it ── */
[data-testid="stTextInput"] > label,
[data-testid="stNumberInput"] > label,
[data-testid="stSelectbox"] > label,
[data-testid="stTextArea"] > label,
[data-testid="stFileUploader"] > label {{
    position: static !important;
    top: auto !important;
    left: auto !important;
    transform: none !important;
    float: none !important;
    display: block !important;
    font-size: 1rem !important;
    color: var(--muted) !important;
    margin-bottom: 0.3rem !important;
    padding: 0 !important;
    z-index: auto !important;
    pointer-events: none !important;
    background: transparent !important;
}}
/* Ensure the input wrapper keeps normal flow */
[data-testid="stTextInput"],
[data-testid="stNumberInput"],
[data-testid="stSelectbox"] {{
    position: relative !important;
}}
/* Prevent Streamlit focus-floating-label trick */
[data-testid="stTextInput"] input:not(:placeholder-shown) ~ label,
[data-testid="stTextInput"] input:focus ~ label {{
    transform: none !important;
    top: auto !important;
    font-size: 1rem !important;
}}

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

/* ── Mobile responsive ─────────────────────────────────────────────────────── */
@media (max-width: 768px) {{

    /* ── Prevent any horizontal overflow ── */
    html, body, .stApp {{ overflow-x: hidden !important; }}

    /* ── Hide sidebar on mobile — push it entirely off-screen left ── */
    /* NOTE: translateX(-110%) was broken because it calculated % of width:0 = 0px */
    /* Using left:-100vw + overflow:hidden is bulletproof */
    section[data-testid="stSidebar"] {{
        position: fixed !important;
        left: -100vw !important;
        top: 0 !important;
        overflow: hidden !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }}
    /* Keep the expand-sidebar arrow button reachable */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }}

    /* ── Main content fills full width ── */
    .stMain, section.stMain, [data-testid="stMain"] {{
        margin-left: 0 !important;
        width: 100% !important;
    }}
    .block-container, .stMainBlockContainer {{
        padding: 0.5rem 0.75rem !important;
        max-width: 100% !important;
    }}

    /* ── Stack ALL Streamlit columns on mobile ── */
    /* flex-wrap:wrap is REQUIRED — without it children overflow right instead of wrapping */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        align-items: center !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex: 0 0 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }}

    /* ── Login hero — scale down huge title & reduce padding ── */
    .login-hero-title {{
        font-size: 2.8rem !important;
        letter-spacing: 0.08em !important;
        line-height: 1.2 !important;
    }}
    .login-hero-subtitle {{
        font-size: 1.1rem !important;
        letter-spacing: 0.05em !important;
    }}
    .login-hero-wrap {{
        padding: 2.5rem 1rem 3rem !important;
    }}

    /* ── Continue / back / analyse buttons full-width on mobile ── */
    .stButton > button {{
        width: 100% !important;
    }}

    /* ── Hero section ── */
    .hero-title    {{ font-size: 1.7rem !important; letter-spacing: 0.08em !important; }}
    .hero-subtitle {{ font-size: 1.1rem !important; }}
    .hero-img      {{ height: 65px !important; }}

    /* ── Verdict box ── */
    .verdict-icon  {{ font-size: 1.5rem !important; }}
    .verdict-label {{ font-size: 2rem !important; letter-spacing: 0.08em !important; }}
    .verdict-car   {{ font-size: 0.9rem !important; letter-spacing: 0.06em !important; }}

    /* ── Cards: tighter padding, wrap text ── */
    .mobile-card {{
        padding: 0.7rem 0.85rem !important;
        word-break: break-word !important;
    }}

    /* ── Hide the desktop plate-button alignment spacer ── */
    .plate-btn-spacer {{ display: none !important; }}

    /* ── Step indicator ── */
    .step-icon {{ width: 30px !important; height: 30px !important; font-size: 0.85rem !important; }}

    /* ── Buttons ── */
    .stButton > button {{
        font-size: 0.92rem !important;
        padding: 0.5rem 0.7rem !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# Inject proper viewport meta — critical for iOS Safari (Streamlit Cloud omits it)
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">',
    unsafe_allow_html=True,
)

# ─── Data helpers ─────────────────────────────────────────────────────────────
def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {}

def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def _append_mailing_list(email: str, first_seen: str):
    """Append a new email to mailing_list.csv (one line per unique user, no duplicates)."""
    import csv
    # Read existing emails to avoid duplicates
    existing = set()
    if MAILING_LIST_FILE.exists():
        try:
            with MAILING_LIST_FILE.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    existing.add(row.get("email", "").lower())
        except Exception:
            pass
    if email.lower() in existing:
        return   # already listed
    write_header = not MAILING_LIST_FILE.exists() or MAILING_LIST_FILE.stat().st_size == 0
    with MAILING_LIST_FILE.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["email", "first_seen"])
        writer.writerow([email, first_seen])

def get_or_create_user(email: str):
    users = load_users()
    if email not in users:
        first_seen = datetime.now().isoformat()
        users[email] = {"created_at": first_seen, "checks": []}
        save_users(users)
        _append_mailing_list(email, first_seen)   # ← add to CSV mailing list
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
    lang    = _get_lang()
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
    lang    = _get_lang()
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

# ─── Mileage analysis ────────────────────────────────────────────────────────
def _analyze_mileage(km: int) -> tuple[list[dict], str | None]:
    """
    Return (reason_list, confidence_override).
    confidence_override is 'high' for very positive signals, 'low' for danger,
    or None to leave the backend value unchanged.
    """
    lang = _get_lang()
    reasons: list[dict] = []
    confidence_override = None

    if km < 30_000:
        reasons.append({
            "severity": "low",
            "title": TR[lang]["km_very_low"].format(km=km),
            "_positive": True,
        })
        confidence_override = "high"          # very low km is a strong positive
    elif km < 80_000:
        reasons.append({
            "severity": "low",
            "title": TR[lang]["km_low"].format(km=km),
            "_positive": True,
        })
    elif km >= 250_000:
        reasons.append({
            "severity": "high",
            "title": TR[lang]["km_very_high"].format(km=km),
        })
        confidence_override = "low"           # extreme mileage is a strong negative
    elif km >= 150_000:
        reasons.append({
            "severity": "medium",
            "title": TR[lang]["km_high"].format(km=km),
        })

    return reasons, confidence_override


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
    nhtsa_data: dict | None = None,
    audio_metrics: dict | None = None,
    paint_data: dict | None = None,
    interior_paths: list[str] | None = None,
    underbody_cv_hints: str | None = None,
) -> dict:
    """
    Single Claude Sonnet call that:
      • Writes a structured assessment split into exterior, interior, mechanical
      • Scores exterior 1–10 and interior 1–10
      • Detects fluid leaks
      • Translates all backend findings titles to the target language
    Returns dict with keys: report, exterior_score, interior_score,
                            leak_assessment, conclusion_external,
                            conclusion_internal, conclusion_mechanical,
                            translated_reasons, translated_steps
    """
    import anthropic, base64, json, re

    client = anthropic.Anthropic(api_key=api_key)
    lang_name = "Hebrew" if lang == "he" else "English"

    # Combine photos + interior + video frames, sample up to 10 for the API call
    # Label interior photos separately so Claude knows context
    exterior_sample  = photo_paths[:5]
    interior_sample  = (interior_paths or [])[:3]
    video_sample     = video_frame_paths[:2]
    all_paths        = exterior_sample + interior_sample + video_sample
    n_exterior       = len(exterior_sample)
    n_interior       = len(interior_sample)

    img_blocks = []
    for idx, p in enumerate(all_paths):
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

    # ── NHTSA context block ───────────────────────────────────────────────────
    nhtsa_block = ""
    if nhtsa_data and (nhtsa_data.get("recall_count", 0) > 0 or nhtsa_data.get("total_complaints", 0) > 0):
        recall_str = "; ".join(
            f"{r['component']}: {r['summary'][:120]}"
            for r in (nhtsa_data.get("recalls") or [])[:5]
        ) or "none"
        top_comps = sorted(
            (nhtsa_data.get("complaint_components") or {}).items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        comp_str = ", ".join(f"{k}({v})" for k, v in top_comps) or "none"
        nhtsa_block = (
            f"\nNHTSA Safety Data ({car_details.get('year','')} {car_details.get('manufacturer','')} "
            f"{car_details.get('model_name','')}):\n"
            f"  Open recalls ({nhtsa_data['recall_count']}): {recall_str}\n"
            f"  Top complaint components: {comp_str} (total {nhtsa_data['total_complaints']} filed)\n"
            f"  INSTRUCTION: If any recall is open, explicitly mention it as a safety priority. "
            f"Cross-reference these complaint categories against your visual and audio findings.\n"
        )

    # ── Audio metrics context block ───────────────────────────────────────────
    audio_block = ""
    if audio_metrics and "reason" not in audio_metrics:
        audio_block = (
            f"\nEngine Audio Metrics (scipy spectral analysis of uploaded engine recording):\n"
            f"  tick_energy (800-3500 Hz valve/tappet range): {audio_metrics.get('tick_energy', 0):.3f}  — >0.08 = valve noise risk\n"
            f"  treble_energy (2000-8000 Hz): {audio_metrics.get('treble_energy', 0):.3f}  — >0.06 = high-frequency noise present\n"
            f"  bass_energy (20-250 Hz): {audio_metrics.get('bass_energy', 0):.3f}  — >0.30 + rms_cv>0.5 = exhaust/bottom-end risk\n"
            f"  periodicity_score (autocorrelation): {audio_metrics.get('periodicity_score', 0):.3f}  — >0.35 = knock/tick rhythm detected\n"
            f"  rms_cv (amplitude variation): {audio_metrics.get('rms_cv', 0):.3f}  — >0.6 = unstable/rough idle\n"
            f"  spectral_centroid: {audio_metrics.get('spectral_centroid', 0):.0f} Hz\n"
            f"  bass_cv: {audio_metrics.get('bass_cv', 0):.3f}\n"
            f"  INSTRUCTION: Write an honest, specific engine-health sentence based on these raw numbers. "
            f"If tick_energy > 0.08, explicitly mention possible valve/tappet noise and recommend inspection. "
            f"If treble_energy > 0.06, note elevated high-frequency content. "
            f"Do NOT say the engine sounds normal unless tick_energy < 0.05 AND rms_cv < 0.4 AND periodicity_score < 0.15.\n"
        )

    # ── Paint context block ───────────────────────────────────────────────────
    paint_block = ""
    if paint_data and paint_data.get("panels_checked", 0) > 0:
        suspicion = paint_data.get("suspicion", "none")
        anomalies = paint_data.get("anomalies", [])
        anom_str  = "; ".join(
            f"{a['pair']} (score {a['score']:.2f}, {a['severity']})"
            for a in anomalies[:6]
        ) or "none"
        paint_block = (
            f"\nOpenCV LAB Paint Analysis ({paint_data['panels_checked']} panel regions checked):\n"
            f"  Overall suspicion level: {suspicion.upper()}\n"
            f"  Color histogram anomalies: {anom_str}\n"
            f"  INSTRUCTION: Only report paint findings if you can see CLEAR VISUAL EVIDENCE in the images "
            f"(overspray on rubber/plastic trim, visible texture differences between panels, obvious hue mismatch). "
            f"Do NOT report paint issues based solely on histogram numbers if the images look visually consistent. "
            f"Natural lighting variation and shadows are NOT paint issues. "
            f"If suspicion level is NONE or LOW, set paint_findings to an empty array []. "
            f"Only flag an issue if suspicion is MEDIUM or HIGH AND you can visually confirm it.\n"
        )

    # ── Underbody CV hint block ───────────────────────────────────────────────
    underbody_block = ""
    if underbody_cv_hints:  # always truthy now — sent regardless of CV result
        underbody_block = (
            f"\nUnderbody / Engine-bay Leak Assessment — CRITICAL:\n"
            f"  {underbody_cv_hints}\n"
            f"  Look for ALL of the following leak indicators:\n"
            f"  • Wet or shiny patches on engine components, hoses, gaskets, or underbody pan\n"
            f"  • Oil residue — dark brown/black coating or drips on any surface\n"
            f"  • Coolant stains — green, pink, or white crystalline residue near hoses/radiator\n"
            f"  • Fresh fluid pooling on the underbody or ground-facing surfaces\n"
            f"  • Discolouration or streaking suggesting fluid has run across a surface\n"
            f"  • Any surface that appears wetter, shinier, or darker than surrounding clean metal\n"
            f"  IMPORTANT: Err on the side of caution. If you see ANYTHING suspicious, report it.\n"
            f"  Fluid leaks are a critical safety and mechanical issue. A missed leak is far more\n"
            f"  dangerous than a false positive. If in doubt — flag it.\n"
        )

    # ── Reject code context (Claude detects visual ones; R03/R10/R12 set programmatically)
    reject_ctx = (
        "\nReject Code Reference — include a code ONLY when you have CLEAR visual evidence:\n"
        "  R01: Severe body damage — major dent, crushed panel, visible impact/structural damage\n"
        "  R02: Paint/panel mismatch — color diff, repaint signs, panel misalignment\n"
        "  R04: Broken/missing exterior parts — cracked lights, missing mirrors/trim/covers\n"
        "  R05: Tire or wheel concern — heavily worn, damaged tyre, bent or cracked rim\n"
        "  R06: Engine bay concern — visible neglect, corrosion, disconnected parts, oil residue\n"
        "  R07: Cabin below expected — heavy wear, broken controls, torn upholstery\n"
        "  R08: Water/moisture intrusion — stains, mold marks, damp flooring, ceiling marks\n"
        "  R09: Low media quality — images too dark/blurry/obstructed for reliable assessment\n"
        "  R11: Conflicting visual signals — contradictory evidence prevents reliable conclusion\n"
        "  DO NOT include R03, R10, or R12 — those are set programmatically by the system.\n"
        "  Return an EMPTY array [] if none clearly apply.\n"
    )

    photo_ctx = f"Images provided: {n_exterior} exterior photo(s)"
    if n_interior:
        photo_ctx += f" + {n_interior} interior photo(s)"
    if video_sample:
        photo_ctx += f" + {len(video_sample)} video frame(s)"

    prompt = f"""CRITICAL: This response must be written ENTIRELY in {lang_name}. Every word in every JSON string field must be in {lang_name}. Using any English words in the output is an error — even for technical terms, translate them.

You are a professional used-car inspector. {photo_ctx}. The first {n_exterior} image(s) are EXTERIOR photos; the next {n_interior} image(s) are INTERIOR photos.

Vehicle: {car_details.get("year","")} {car_details.get("manufacturer","")} {car_details.get("model_name","")} {car_details.get("trim","")}
Odometer: {car_details.get("odometer","?")} km | Prior owners: {car_details.get("prev_owners",1)} | Usage: {["Private","Rental/Lease","Company Car","Unknown"][min(car_details.get("usage_type",0),3)]}
System verdict: {decision.recommendation.upper()} (confidence: {decision.confidence})

Automated findings (English): {"; ".join(existing_reasons) or "none"}
Automated next steps (English): {"; ".join(existing_steps) or "none"}
{nhtsa_block}{audio_block}{paint_block}{underbody_block}{reject_ctx}
Your task — respond ONLY in {lang_name} — produce this exact JSON (no extra text):
{{
  "report": "<4–6 sentences: overall condition summary covering body, paint, interior and any concerns>",
  "exterior_score": <integer 1-10, where 10=showroom perfect, 1=severely damaged>,
  "interior_score": <integer 1-10, where 10=pristine, 1=heavily worn/damaged>,
  "leak_assessment": "<one of: none detected | oil leak suspected | water/coolant leak suspected | multiple leaks suspected>",
  "conclusion_external": "<1–2 sentences on exterior: body condition, paint consistency, any accident/repair signs, overall preservation>",
  "conclusion_internal": "<1–2 sentences on interior: cleanliness, seat wear, dashboard condition, general upkeep — use interior photos if provided>",
  "conclusion_mechanical": "<1–2 sentences on mechanical concerns visible in images (engine bay, underbody if shown) — do NOT repeat audio findings here>",
  "translated_reasons": ["<MUST be in {lang_name} — translate each automated finding, preserving meaning, same count>"],
  "translated_steps": ["<MUST be in {lang_name} — translate each next step, do not leave any in English>"],
  "paint_findings": [
    {{"panel": "<panel name in {lang_name}>", "indicator": "<what you observed>", "severity": "<high|medium|low>"}}
  ],
  "reject_codes": ["<only include codes from this list that CLEARLY apply based on visual evidence: R01 R04 R05 R06 R07 R08 R09 R11 — omit R02/R03/R10/R12, those are set programmatically>"]
}}

Base exterior_score and interior_score ONLY on what you can visually observe in the images. Be honest and specific.
FINAL REMINDER: Every string in the JSON — report, conclusion_external, conclusion_internal, conclusion_mechanical, leak_assessment, translated_reasons, translated_steps, paint_findings — must be in {lang_name} only. reject_codes must be a JSON array of strings, only codes that apply, empty array [] if none."""

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
        "reject_codes": [],
    }

# ─── Result email sender ──────────────────────────────────────────────────────
def _send_result_email(to_email: str, result: dict, lang: str) -> bool:
    """
    Send the analysis summary to the user's email via Gmail SMTP.
    Requires GMAIL_USER and GMAIL_APP_PASSWORD in st.secrets.
    Returns True on success, False on failure (silently — never block the user).
    """
    import smtplib, html as _html
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    try:
        gmail_user = st.secrets.get("GMAIL_USER", "")
        gmail_pwd  = st.secrets.get("GMAIL_APP_PASSWORD", "")
        if not gmail_user or not gmail_pwd:
            return False  # credentials not configured yet
    except Exception:
        return False

    # ── Build content ─────────────────────────────────────────────────────────
    is_he     = (lang == "he")
    rec       = result.get("recommendation", "inconclusive")
    car_label = result.get("car_label", "")
    date      = result.get("created_at", "")[:10]
    ext_score = result.get("exterior_score")
    int_score = result.get("interior_score")

    rec_labels = {
        "go":           ("מתאים",       "GO",               "#4A7A4A"),
        "no_go":        ("זוהו בעיות",  "RISK DETECTED",    "#B04040"),
        "inconclusive": ("מידע חסר",    "INSUFFICIENT DATA","#C8A96A"),
    }
    rec_he, rec_en, rec_color = rec_labels.get(rec, ("?", "?", "#9A9080"))
    rec_label = rec_he if is_he else rec_en

    # Action recommendation
    paint_susp   = (result.get("paint_data") or {}).get("suspicion", "none")
    leak_raw     = (result.get("leak_assessment") or "none detected").lower()
    audio_labels = {f.get("label","") for f in (result.get("audio_findings_raw") or [])}
    bad_audio    = {"rod_knock_suspected","valve_tick_suspected","belt_squeal_suspected",
                    "exhaust_leak_suspected","rough_idle_suspected"}
    has_paint    = paint_susp in ("medium","high")
    has_leak     = leak_raw != "none detected"
    has_audio    = bool(audio_labels & bad_audio)

    if has_leak:
        action_he = "זוהתה דליפת נוזל | יש לבדוק במוסך מורשה לפני כל שיקול רכישה."
        action_en = "Fluid leak detected | inspect at a certified garage before any purchase decision."
    elif has_paint and has_audio:
        action_he = "זוהו מספר גורמי סיכון: בעיות צבע ורעשי מנוע. לא מומלץ לרכוש."
        action_en = "Multiple risk factors: paint/body issues and engine noise. We recommend avoiding this vehicle."
    elif has_audio:
        action_he = "זוהה סיכון קולי במנוע. חובה לבדיקה מוסמכת לפני רכישה."
        action_en = "Engine noise risk detected. Must go to a certified inspection center before purchase."
    else:
        action_he = "הרכב נראה תקין | ייתכן שמדובר ברכישה טובה. חובה לבדיקה מוסמכת לפני חתימה."
        action_en = "Vehicle looks clean | could be a good buy. An official certified inspection is mandatory before signing."

    action_text = action_he if is_he else action_en

    # Top findings
    reasons_html = ""
    for r in (result.get("top_reasons") or [])[:6]:
        sev = r.get("severity","low")
        dot = {"high":"🔴","medium":"🟡","low":"🟢"}.get(sev,"◦")
        title = _html.escape(r.get("title",""))
        reasons_html += f"<tr><td style='padding:4px 8px;'>{dot}</td><td style='padding:4px 8px;font-size:14px;'>{title}</td></tr>"

    score_row = ""
    if ext_score is not None:
        score_row += f"<td style='padding:8px 16px;text-align:center;'><div style='font-size:22px;font-weight:700;color:#C8A96A;'>{ext_score}/10</div><div style='font-size:11px;color:#888;'>{'חיצוני' if is_he else 'Exterior'}</div></td>"
    if int_score is not None:
        score_row += f"<td style='padding:8px 16px;text-align:center;'><div style='font-size:22px;font-weight:700;color:#C8A96A;'>{int_score}/10</div><div style='font-size:11px;color:#888;'>{'פנים' if is_he else 'Interior'}</div></td>"

    conc_ext  = _html.escape(result.get("conclusion_external",""))
    conc_int  = _html.escape(result.get("conclusion_internal",""))
    conc_mech = _html.escape(result.get("conclusion_mechanical",""))

    dir_attr = 'dir="rtl"' if is_he else 'dir="ltr"'

    subject = f"{'בדיקת רכב' if is_he else 'Car Check'} | {car_label} | {rec_label}"
    html_body = f"""<!DOCTYPE html>
<html {dir_attr}>
<head><meta charset="utf-8"/></head>
<body style="margin:0;padding:0;background:#111;font-family:Georgia,serif;color:#E8E0D0;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#111;">
<tr><td align="center" style="padding:32px 16px;">
<table width="600" cellpadding="0" cellspacing="0" style="background:#1A1A1A;border-radius:10px;border:1px solid #333;">

<!-- Header -->
<tr><td style="background:#141414;border-radius:10px 10px 0 0;padding:28px 32px;text-align:center;border-bottom:1px solid #C8A96A44;">
  <div style="font-size:11px;letter-spacing:4px;color:#C8A96A;text-transform:uppercase;margin-bottom:6px;">{'בדיקת רכב' if is_he else 'UsedCar Check'}</div>
  <div style="font-size:26px;font-weight:700;color:#C8A96A;">{_html.escape(car_label)}</div>
  <div style="font-size:12px;color:#666;margin-top:4px;">{date}</div>
</td></tr>

<!-- Verdict -->
<tr><td style="padding:28px 32px;text-align:center;border-bottom:1px solid #2A2A2A;">
  <div style="display:inline-block;background:{rec_color}22;border:2px solid {rec_color};border-radius:8px;padding:14px 36px;">
    <div style="font-size:28px;font-weight:700;color:{rec_color};letter-spacing:2px;">{rec_label}</div>
  </div>
  <div style="margin-top:16px;font-size:15px;color:#B8A880;font-style:italic;">{action_text}</div>
</td></tr>

{"" if not score_row else f"<tr><td style='padding:20px 32px;border-bottom:1px solid #2A2A2A;'><table width='100%' cellpadding='0' cellspacing='0'><tr>{score_row}</tr></table></td></tr>"}

{"" if not reasons_html else f"<tr><td style='padding:20px 32px;border-bottom:1px solid #2A2A2A;'><div style='font-size:11px;letter-spacing:3px;color:#888;text-transform:uppercase;margin-bottom:10px;'>{'ממצאים' if is_he else 'Findings'}</div><table width='100%' cellpadding='0' cellspacing='0'>{reasons_html}</table></td></tr>"}

{"" if not (conc_ext or conc_int or conc_mech) else f"""
<tr><td style='padding:20px 32px;border-bottom:1px solid #2A2A2A;'>
  <div style='font-size:11px;letter-spacing:3px;color:#888;text-transform:uppercase;margin-bottom:10px;'>{'סיכום' if is_he else 'Summary'}</div>
  {"" if not conc_ext else f"<div style='margin-bottom:10px;'><span style='font-size:12px;color:#C8A96A;'>{'חיצוני' if is_he else 'Exterior'}: </span><span style='font-size:14px;'>{conc_ext}</span></div>"}
  {"" if not conc_int else f"<div style='margin-bottom:10px;'><span style='font-size:12px;color:#C8A96A;'>{'פנים' if is_he else 'Interior'}: </span><span style='font-size:14px;'>{conc_int}</span></div>"}
  {"" if not conc_mech else f"<div><span style='font-size:12px;color:#C8A96A;'>{'מכאני' if is_he else 'Mechanical'}: </span><span style='font-size:14px;'>{conc_mech}</span></div>"}
</td></tr>"""}

<!-- Footer -->
<tr><td style="padding:20px 32px;text-align:center;border-radius:0 0 10px 10px;">
  <div style="font-size:11px;color:#555;line-height:1.7;">
    {'הדוח מיועד לסיוע בקבלת החלטה בלבד ואינו תחליף לבדיקה מקצועית מוסמכת.' if is_he else 'This report is for decision support only and is not a substitute for a certified professional inspection.'}
  </div>
</td></tr>

</table>
</td></tr>
</table>
</body></html>"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{'בדיקת רכב' if is_he else 'UsedCar Check'} <{gmail_user}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pwd)
            smtp.sendmail(gmail_user, to_email, msg.as_string())
        return True
    except Exception:
        return False


# ─── Analysis runner ──────────────────────────────────────────────────────────
def run_analysis(car_details, photo_files, audio_file, underbody_file=None, video_file=None, interior_files=None) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        photo_paths = []
        for i, f in enumerate(photo_files):
            ext = Path(f.name).suffix or ".jpg"
            p = tmp / f"photo_{i}{ext}"
            p.write_bytes(f.getvalue())
            photo_paths.append(str(p))
        # Interior photos saved separately so Claude knows context
        interior_paths = []
        for i, f in enumerate(interior_files or []):
            ext = Path(f.name).suffix or ".jpg"
            p = tmp / f"interior_{i}{ext}"
            p.write_bytes(f.getvalue())
            interior_paths.append(str(p))
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

        audio_findings, audio_dur, audio_metrics = _analyze_audio(str(ap))

        # ── Video frame extraction ────────────────────────────────────────────
        video_frame_paths: list[str] = []
        if video_file is not None:
            vp = tmp / f"vehicle_video{Path(video_file.name).suffix or '.mp4'}"
            vp.write_bytes(video_file.getvalue())
            video_frame_paths = _extract_video_frames(str(vp), tmp)

        # All photo paths including interior + video frames (for backend quality assessment)
        all_visual_paths = photo_paths + interior_paths + video_frame_paths

        photo_qualities    = [assess_image_quality(p) for p in all_visual_paths]
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

        # ── Ownership, mileage & usage analysis — inject into decision ──────────
        num_owners  = int(car_details.get("prev_owners", 1))
        usage_type  = int(car_details.get("usage_type", 0))
        year        = int(car_details.get("year", 2015))
        odometer_km = int(car_details.get("odometer") or 0)
        lang_now    = _get_lang()

        own_reasons    = _analyze_ownership(num_owners, year)
        usage_reasons  = _analyze_usage(usage_type, num_owners)
        km_reasons, km_conf = _analyze_mileage(odometer_km)

        extra_reasons = own_reasons + usage_reasons + km_reasons
        if extra_reasons:
            has_high = any(r.get("severity") == "high" for r in extra_reasons)
            if has_high and decision.recommendation == "go":
                decision.recommendation = "inconclusive"
            decision.top_reasons = extra_reasons + decision.top_reasons

        # Apply mileage confidence override only when verdict is GO
        # (negative km override only applied if no harder signal already present)
        if km_conf == "high" and decision.recommendation == "go":
            # Boost confidence for very low-km single-owner cars
            if num_owners == 1:
                decision.confidence = "high"
        elif km_conf == "low" and decision.recommendation in ("go", "inconclusive"):
            decision.confidence = "low"

        # ── Force verdict down based on hard mechanical signals ───────────────
        # Match the exact label emitted by analyze_underbody_image
        _LEAK_LABELS   = {"possible_underbody_fluid_stain", "oil_leak", "coolant_leak"}
        _KNOCK_LABELS  = {"rod_knock_suspected", "exhaust_leak_suspected"}
        _WARN_LABELS   = {"valve_tick_suspected", "belt_squeal_suspected", "rough_idle_suspected"}

        has_underbody_leak_cv = any(
            f.get("label", "").lower() in _LEAK_LABELS
            for f in underbody_findings
        )
        has_knock  = any(f.label in _KNOCK_LABELS for f in audio_findings)
        has_warn   = any(f.label in _WARN_LABELS  for f in audio_findings)

        _mech_issues = []
        if has_underbody_leak_cv:
            _mech_issues.append({
                "severity": "high",
                "title": TR[lang_now].get("leak_oil", "Oil leak suspected") if lang_now == "he"
                         else "Oil / fluid leak detected in underbody image",
            })
            if decision.recommendation in ("go", "inconclusive"):
                decision.recommendation = "no_go"
            decision.confidence = "high"

        if has_knock:
            _mech_issues.append({
                "severity": "high",
                "title": "חשד לדפיקות מנוע | נדרשת בדיקה דחופה" if lang_now == "he"
                         else "Engine knock detected | urgent inspection required",
            })
            if decision.recommendation == "go":
                decision.recommendation = "no_go"

        if has_warn and not has_knock:
            _mech_issues.append({
                "severity": "medium",
                "title": "רעש חריג זוהה בהקלטת המנוע | מומלץ לבדוק" if lang_now == "he"
                         else "Abnormal engine noise detected | inspection recommended",
            })
            # Engine noise is a real finding — flag it as a risk, not "unclear"
            if decision.recommendation in ("go", "inconclusive"):
                decision.recommendation = "no_go"

        if _mech_issues:
            # Filter out the "no red flags" fallback — real findings are present
            _filtered_reasons = [
                r for r in (decision.top_reasons or [])
                if not any(kw in r.get("title", "").lower()
                           for kw in ("red flag", "דגלים אדומים", "no clear", "לא זוהו דגלים"))
            ]
            decision.top_reasons = _mech_issues + _filtered_reasons

        # ── Paint consistency (OpenCV LAB histogram comparison) ───────────────
        paint_data = _analyze_paint_consistency(photo_paths + video_frame_paths)

        # ── NHTSA recall & complaint data ─────────────────────────────────────
        nhtsa_data = _fetch_nhtsa_data(
            car_details.get("manufacturer", ""),
            car_details.get("model_name",   ""),
            int(car_details.get("year", 0)),
        )

        # ── Yad2 pricelist ────────────────────────────────────────────────────
        yad2_price_data = _fetch_yad2_price(
            car_details.get("manufacturer", ""),
            car_details.get("model_name",   ""),
            int(car_details.get("year", 0)),
        )

        # ── Comprehensive AI report (scores + translation + 10-sentence report)
        lang = _get_lang()
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        # Build underbody CV hint for Claude — always send even without CV signal
        # so Claude knows to look hard regardless
        if has_underbody_leak_cv:
            _cv_hint = (
                "ALERT — Computer vision pre-scan DETECTED a fluid stain signal in the underbody image "
                "(oil-colour, coolant-colour, dark-wet patch, or shiny reflective area). "
                "This is a HIGH-PRIORITY finding. Examine every underbody/engine-bay image for: "
                "wet patches, oily residue, discolouration, pooling fluid, fresh or dried stains, "
                "drips on components, or any surface that looks wetter/shinier than surrounding metal. "
                "Even a small or ambiguous stain MUST be reported as a potential leak. "
                "Set leak_assessment to a descriptive finding — do NOT return 'none detected'."
            )
        else:
            _cv_hint = (
                "No strong colour signal detected by CV pre-scan. However, you MUST still "
                "carefully examine all underbody and engine-bay images for any signs of fluid: "
                "wet/shiny patches, oil residue, discolouration, drips, or staining on components. "
                "Leaks are a critical safety issue — report any suspicious area, even if minor."
            )

        ai_report = _generate_comprehensive_report(
            car_details, decision, photo_paths + video_frame_paths, [], lang, api_key,
            nhtsa_data=nhtsa_data,
            audio_metrics=audio_metrics,
            paint_data=paint_data,
            interior_paths=interior_paths,
            underbody_cv_hints=_cv_hint,   # always passed — no-leak hint also sent
        )

        # ── Combine CV + Claude for final leak determination ─────────────────
        # Policy: CV OR Claude either one is enough to flag a leak.
        # CV detection is NOT overrideable by Claude saying "none detected".
        # Both signals are treated as independent evidence — union, not intersection.
        _ai_leak_str = ai_report.get("leak_assessment", "none detected").lower()
        _ai_says_leak = (
            _ai_leak_str not in ("none detected", "none", "no leak", "no leaks", "clean", "")
            and "none" not in _ai_leak_str
        )
        has_underbody_leak = has_underbody_leak_cv or _ai_says_leak
        # If CV caught it but Claude didn't mention it, inject a clear leak assessment
        if has_underbody_leak_cv and not _ai_says_leak:
            ai_report["leak_assessment"] = (
                "Possible fluid stain detected by computer vision analysis — "
                "oil-colour, dark-wet patch, or reflective area found in underbody image."
            )

        # ── Reject code computation ──────────────────────────────────────────────────────────────────────
        _ai_codes = set(ai_report.get("reject_codes") or [])
        # Keep only known codes Claude was asked to return
        _ai_codes &= {"R01", "R04", "R05", "R06", "R07", "R08", "R09", "R11"}

        # Programmatic codes based on CV / data signals
        if has_underbody_leak:
            _ai_codes.add("R03")
        if (paint_data or {}).get("suspicion") in ("medium", "high"):
            _ai_codes.add("R02")
        if len(photo_paths) < 4:
            _ai_codes.add("R10")

        # R12: auto-trigger when 2+ hard codes present (excluding R12 itself)
        _hard_triggered = _ai_codes & (_REJECT_HARD - {"R12"})
        if len(_hard_triggered) >= 2:
            _ai_codes.add("R12")

        # Apply verdict overrides based on reject severity
        if _ai_codes & _REJECT_HARD:
            decision.recommendation = "no_go"
            decision.confidence     = "high"
        elif _ai_codes & _REJECT_SOFT:
            if decision.recommendation == "go":
                decision.recommendation = "no_go"
        elif _ai_codes & _REJECT_TECH:
            if decision.recommendation == "go":
                decision.recommendation = "inconclusive"

        reject_codes = sorted(_ai_codes)

        # Apply Claude translations back (may still be English if Claude failed)
        tr_reasons = ai_report.get("translated_reasons", []) or []
        tr_steps   = ai_report.get("translated_steps",   []) or []
        if tr_reasons:
            for i, r in enumerate(decision.top_reasons or []):
                if i < len(tr_reasons) and tr_reasons[i]:
                    r["title"] = tr_reasons[i]
        if tr_steps:
            for i, s in enumerate(decision.next_steps or []):
                if i < len(tr_steps) and tr_steps[i]:
                    s["text"] = tr_steps[i]

        # ── Final pass: _tr_backend catches any English that Claude missed ────
        for r in (decision.top_reasons or []):
            r["title"] = _tr_backend(r.get("title", ""))
        for s in (decision.next_steps or []):
            if isinstance(s, dict):
                s["text"] = _tr_backend(s.get("text", ""))
        for e in (decision.education or []):
            if isinstance(e, dict):
                e["title"] = _tr_backend(e.get("title", ""))

        # Rebuild next_steps as clean dicts
        _tr_steps_final = [
            {"text": (s.get("text", "") if isinstance(s, dict) else str(s))}
            for s in (decision.next_steps or [])
        ]
        decision.next_steps = _tr_steps_final or decision.next_steps

        # Prepend urgent action items in the correct language for critical findings
        _urgent_steps = []
        if has_underbody_leak:
            _urgent_steps.append({"text":
                "דליפת שמן/נוזל זוהתה | יש לבדוק במוסך מורשה לפני כל שיקול של רכישה." if lang_now == "he"
                else "Fluid leak detected | have the vehicle inspected by a certified mechanic before any purchase decision."
            })
        if has_knock:
            _urgent_steps.append({"text":
                "זוהו סימנים לדפיקות מנוע | נדרשת בדיקת לחץ צילינדרים ומסב בדחיפות." if lang_now == "he"
                else "Engine knock detected | compression test and bearing inspection required urgently."
            })
        if has_warn and not has_knock:
            _urgent_steps.append({"text":
                "זוהה רעש חריג במנוע | מומלץ לבדוק שסתומים ורצועות הנע לפני רכישה." if lang_now == "he"
                else "Abnormal engine noise detected | have valves and drive belts checked before purchase."
            })
        if _urgent_steps:
            decision.next_steps = _urgent_steps + (decision.next_steps or [])

        # Serialise audio findings for display
        audio_findings_raw = [
            {"label": f.label, "confidence": f.confidence, "details": f.details}
            for f in audio_findings
        ]

        return decision, audio_dur, ai_report, nhtsa_data, audio_metrics, audio_findings_raw, paint_data, yad2_price_data, reject_codes

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
    <div class='mobile-card' style='background:{bg};border:1px solid {color};border-radius:6px;
                padding:2.5rem 2rem;text-align:{align};margin:1rem 0;
                box-shadow:0 0 40px {color}22;'>
        <div class='verdict-icon' style='font-size:2.5rem;color:{color};margin-bottom:0.5rem;'>{icon}</div>
        <div class='verdict-label' style='font-family:Cormorant Garamond,serif;font-size:3.5rem;
                    font-weight:600;letter-spacing:0.22em;color:{color};
                    line-height:1;'>{label}</div>
        <div style='height:1px;width:60px;background:{color};margin:1rem {"0 1rem auto" if not is_rtl else "0 auto 1rem 0"};opacity:0.6;'></div>
        <div class='verdict-car' style='font-size:1.17rem;letter-spacing:0.2em;color:var(--muted);
                    text-transform:uppercase;'>{car_label}</div>
        <div style='font-size:1.04rem;color:var(--muted);margin-top:0.3rem;'>{date}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Original report expander (shown when user has refined the analysis) ─────
    _orig = st.session_state.get("original_result")
    if _orig:
        with st.expander(f"📋 {t('view_original_btn')}", expanded=False):
            _orig_rec = _orig.get("recommendation", "inconclusive")
            _orig_label, _orig_color, _ = verdict_meta(_orig_rec)
            st.markdown(
                f"<div style='margin-bottom:0.5rem;{rtl_css}'>"
                f"<span style='font-size:1.5rem;font-weight:700;color:{_orig_color};'>{_orig_label}</span>"
                f"<span style='font-size:0.95rem;color:var(--muted);margin-{'right' if is_rtl else 'left'}:0.8rem;'>"
                f"{_orig.get('created_at','')[:10]}</span></div>",
                unsafe_allow_html=True,
            )
            _orig_report = _orig.get("detailed_report", "")
            if _orig_report:
                st.markdown(f"<p style='font-size:1.1rem;color:var(--text);{rtl_css}'>{_orig_report}</p>",
                            unsafe_allow_html=True)

    # ── Reject code findings ──────────────────────────────────────────────────────────────────
    _rc_list = result.get("reject_codes") or []
    if _rc_list:
        _lang_now = st.session_state.get("lang", "he")
        _severity_colors = {
            "hard": ("#B04040", "rgba(176,64,64,0.13)"),
            "soft": ("#C8A96A", "rgba(200,169,106,0.10)"),
            "tech": ("#6A8FAA", "rgba(106,143,170,0.10)"),
        }
        _severity_labels = {
            "hard": {"he": "חמור", "en": "Critical"},
            "soft": {"he": "אזהרה", "en": "Caution"},
            "tech": {"he": "מידע חסר", "en": "Retry"},
        }
        gold_divider()
        _rc_title = "ממצאי בדיקה" if _lang_now == "he" else "Inspection Findings"
        st.markdown(
            f"<p style='font-size:1.07rem;color:var(--muted);letter-spacing:0.18em;"
            f"text-transform:uppercase;margin-bottom:0.6rem;{rtl_css}'>{_rc_title}</p>",
            unsafe_allow_html=True,
        )
        for _code in _rc_list:
            _info = REJECT_TABLE.get(_code)
            if not _info:
                continue
            _sev   = _info["severity"]
            _color, _bg = _severity_colors.get(_sev, ("#888", "rgba(136,136,136,0.08)"))
            _title = _info["title_he"] if _lang_now == "he" else _info["title_en"]
            _expl  = _info["expl_he"]  if _lang_now == "he" else _info["expl_en"]
            _slabel = _severity_labels.get(_sev, {}).get(_lang_now, _sev)
            st.markdown(
                f"<div style='background:{_bg};border:1px solid {_color}33;"
                f"border-right:3px solid {_color};border-radius:6px;"
                f"padding:0.7rem 1rem;margin:0.35rem 0;{rtl_css}'>"
                f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;'>"
                f"<span style='font-size:0.72rem;font-weight:700;color:{_color};"
                f"background:{_color}22;border-radius:4px;padding:0.1rem 0.4rem;"
                f"letter-spacing:0.05em;'>{_code}</span>"
                f"<span style='font-size:0.72rem;color:{_color};letter-spacing:0.08em;"
                f"text-transform:uppercase;'>{_slabel}</span>"
                f"<span style='font-size:1rem;font-weight:600;color:var(--text);'>{_title}</span>"
                f"</div>"
                f"<div style='font-size:0.95rem;color:var(--muted);'>{_expl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Data quality notice ───────────────────────────────────────────────────
    _dq = result.get("data_quality") or {}
    _dq_issues = []
    _ext_n = _dq.get("exterior_count", 0)
    _int_n = _dq.get("interior_count", 0)
    _aud_s = _dq.get("audio_duration", 99)
    if _ext_n and _ext_n < 6:
        _dq_issues.append(t("data_few_exterior").format(n=_ext_n))
    if not _int_n:
        _dq_issues.append(t("data_missing_interior"))
    if not _dq.get("has_underbody"):
        _dq_issues.append(t("data_missing_underbody"))
    if _aud_s and _aud_s < 15:
        _dq_issues.append(t("data_short_audio").format(s=_aud_s))
    if _dq_issues:
        _items_html = "".join(
            f"<li style='margin:0.25rem 0;'>{_i}</li>" for _i in _dq_issues
        )
        st.markdown(
            f"<div style='background:rgba(200,169,106,0.08);border:1px solid rgba(200,169,106,0.45);"
            f"border-radius:8px;padding:1rem 1.4rem;margin:0.6rem 0 1rem;{rtl_css}'>"
            f"<div style='font-size:1.05rem;font-weight:600;color:var(--gold);margin-bottom:0.5rem;'>"
            f"{t('data_quality_title')}</div>"
            f"<div style='font-size:1rem;color:var(--muted);margin-bottom:0.5rem;'>"
            f"{t('data_quality_msg')}</div>"
            f"<ul style='font-size:0.97rem;color:var(--text);margin:0.3rem 0 0.5rem "
            f"{'1.2rem' if not is_rtl else '0'};padding-{'left' if not is_rtl else 'right'}:1.2rem;'>"
            f"{_items_html}</ul>"
            f"<div style='font-size:0.9rem;color:var(--muted);font-style:italic;'>"
            f"💡 {t('data_refine_hint')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Confidence + Audio meta ───────────────────────────────────────────────
    _conf_levels = {"high": {"he": "גבוה", "en": "High"},
                    "medium": {"he": "בינוני", "en": "Medium"},
                    "low": {"he": "נמוך", "en": "Low"}}
    _lang_now = st.session_state.get("lang", "he")
    _conf_word = _conf_levels.get(conf, {}).get(_lang_now, conf.upper())
    if rec == "go":
        _ctx_key = "conf_ctx_go"
    elif rec == "no_go":
        _ctx_key = "conf_ctx_nogo"
    else:
        _ctx_key = "conf_ctx_inc"
    _conf_sentence = t(_ctx_key).format(_conf_word)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<span style='font-size:1.11rem;color:var(--muted);letter-spacing:0.08em;'>{t('confidence_label')}</span>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1.05rem;color:var(--gold);margin:0.2rem 0 0;{rtl_css}'>{_conf_sentence}</p>", unsafe_allow_html=True)
    with col2:
        dur = result.get("audio_duration_seconds")
        if dur:
            st.markdown(f"<span style='font-size:1.11rem;color:var(--muted);'>🎙 {t('audio_analysed').format(dur)}</span>", unsafe_allow_html=True)

    gold_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION RECOMMENDATION BOX — derived from combined signals
    # ═══════════════════════════════════════════════════════════════════════════
    _paint_susp  = (result.get("paint_data") or {}).get("suspicion", "none")
    _leak_raw    = (result.get("leak_assessment") or "none detected").lower()
    _audio_labels = {f.get("label","") for f in (result.get("audio_findings_raw") or [])}
    _bad_audio   = {"rod_knock_suspected", "valve_tick_suspected",
                    "belt_squeal_suspected", "exhaust_leak_suspected", "rough_idle_suspected"}
    _severe_audio = {"rod_knock_suspected"}

    _has_paint_issue = _paint_susp in ("medium", "high")
    _has_leak        = _leak_raw != "none detected"
    _has_any_audio   = bool(_audio_labels & _bad_audio)
    _has_severe_audio = bool(_audio_labels & _severe_audio)

    # Determine tier
    if _has_leak:
        _action_key  = "action_leak"
        _action_bg   = "rgba(176,64,64,0.13)"
        _action_border = "#B04040"
        _action_icon = "⛔"
    elif _has_paint_issue and _has_any_audio:
        # Multiple risk factors — avoid
        _action_key  = "action_red"
        _action_bg   = "rgba(176,64,64,0.13)"
        _action_border = "#B04040"
        _action_icon = "🚫"
    elif _has_any_audio and not _has_paint_issue:
        # Engine risk only — caution, must test
        _action_key  = "action_yellow"
        _action_bg   = "rgba(200,169,106,0.12)"
        _action_border = "#C8A96A"
        _action_icon = "⚠️"
    else:
        # All clear visually — potential good buy, still must test
        _action_key  = "action_green"
        _action_bg   = "rgba(74,122,74,0.13)"
        _action_border = "#4A7A4A"
        _action_icon = "✅"

    st.markdown(
        f"<div class='mobile-card' style='background:{_action_bg};border:1.5px solid {_action_border};"
        f"border-radius:8px;padding:1.1rem 1.4rem;margin:0.4rem 0 1.2rem;{rtl_css}'>"
        f"<div style='font-size:1rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;"
        f"margin-bottom:0.4rem;'>{_action_icon}&nbsp; {t('action_label')}</div>"
        f"<div style='font-size:1.2rem;line-height:1.65;font-weight:500;color:var(--text);'>"
        f"{t(_action_key)}</div></div>",
        unsafe_allow_html=True,
    )

    gold_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # REGISTRY DATA CARD — Ministry of Transport official data
    # ═══════════════════════════════════════════════════════════════════════════
    gold_divider()
    _reg = result.get("plate_registry")
    if _reg and _reg.get("plate"):
        _own_key  = _reg.get("ownership_key", "private")
        _baalut   = _reg.get("baalut_he", "")

        # Ownership type display + risk colour
        _own_labels = {
            "private": t("ownership_private"),
            "lease":   t("ownership_lease"),
            "rental":  t("ownership_rental"),
            "govt":    t("ownership_govt"),
            "company": t("ownership_company"),
            "other":   _baalut or "—",
        }
        _own_colors = {
            "private": ("#4A7A4A", "rgba(74,122,74,0.12)"),
            "lease":   ("#C8A96A", "rgba(200,169,106,0.12)"),
            "rental":  ("#B04040", "rgba(176,64,64,0.12)"),
            "govt":    ("#C8A96A", "rgba(200,169,106,0.12)"),
            "company": ("#C8A96A", "rgba(200,169,106,0.12)"),
            "other":   ("#C8A96A", "rgba(200,169,106,0.12)"),
        }
        _own_text  = _own_labels.get(_own_key, _baalut or "—")
        _own_fc, _own_bg = _own_colors.get(_own_key, ("#C8A96A", "rgba(200,169,106,0.12)"))

        # Warning if not private
        _own_warn_html = ""
        if _own_key != "private":
            _own_warn_html = (
                f"<div style='background:rgba(176,64,64,0.10);border:1px solid #B04040;"
                f"border-radius:6px;padding:0.6rem 1rem;margin-top:0.7rem;font-size:1.05rem;"
                f"color:#e87a7a;line-height:1.5;{rtl_css}'>"
                f"{t('ownership_warn').format(type=_own_text)}</div>"
            )

        # Helper: format a date string YYYY-MM-DD → DD/MM/YYYY
        def _fmt_date(s):
            if not s: return "—"
            p = str(s).split("-")
            if len(p) == 3: return f"{p[2]}/{p[1]}/{p[0]}"
            if len(p) == 2: return f"{p[1]}/{p[0]}"
            return s

        _first_road_fmt = _fmt_date(_reg.get("first_road", ""))
        _last_test_fmt  = _fmt_date(_reg.get("last_test", ""))
        _valid_fmt      = _fmt_date(_reg.get("valid_until", ""))
        _color_he       = _reg.get("color_he", "") or "—"
        _fuel_he        = _reg.get("fuel_he",  "") or "—"
        _trim_val       = _reg.get("trim",     "") or ""

        def _reg_row(label, value, highlight=False):
            _vc = "var(--gold)" if highlight else "var(--text)"
            return (
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"padding:0.32rem 0;border-bottom:1px solid rgba(255,255,255,0.05);{rtl_css}'>"
                f"<span style='color:var(--muted);font-size:0.97rem;'>{label}</span>"
                f"<span style='color:{_vc};font-size:1rem;font-weight:500;'>{value}</span>"
                f"</div>"
            )

        _rows_html = (
            _reg_row(t("ownership_type"),
                     f"<span style='color:{_own_fc};font-weight:700;'>{_own_text}</span>")
            + _reg_row(t("first_road_reg"),  _first_road_fmt)
            + _reg_row(t("last_inspection"), _last_test_fmt)
            + _reg_row(t("reg_valid_until"), _valid_fmt)
            + _reg_row(t("car_color"),       _color_he)
            + _reg_row(t("fuel_type"),       _fuel_he)
            + (_reg_row("Trim", _trim_val) if _trim_val else "")
        )

        st.markdown(
            f"<div class='mobile-card' style='background:rgba(200,169,106,0.05);border:1px solid rgba(200,169,106,0.22);"
            f"border-radius:8px;padding:1rem 1.4rem;margin:0.4rem 0;{rtl_css}'>"
            f"<div style='font-size:1rem;color:var(--gold);letter-spacing:0.1em;text-transform:uppercase;"
            f"margin-bottom:0.6rem;'>🏛 {t('registry_title')}</div>"
            f"{_rows_html}"
            f"{_own_warn_html}"
            f"<div style='font-size:0.85rem;color:var(--muted);margin-top:0.6rem;'>◦ {t('registry_source')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:rgba(44,44,44,0.25);border:1px dashed rgba(200,169,106,0.2);"
            f"border-radius:8px;padding:0.8rem 1.4rem;margin:0.4rem 0;{rtl_css}'>"
            f"<span style='font-size:0.97rem;color:var(--muted);'>🏛 {t('no_plate_data')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Visual Assessment (chassis, paint, appearance)
    # ═══════════════════════════════════════════════════════════════════════════
    def _section_header(icon: str, key: str, color: str = "var(--gold)"):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin:1.2rem 0 0.6rem;{rtl_css}'>"
            f"<span style='font-size:1.4rem;'>{icon}</span>"
            f"<span style='font-family:Cormorant Garamond,serif;font-size:1.45rem;font-weight:600;"
            f"letter-spacing:0.12em;text-transform:uppercase;color:{color};'>{t(key)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _section_header("🏛", "sec_visual")

    # ── Scores ────────────────────────────────────────────────────────────────
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

    # ── Visual leak assessment (from AI image inspection) ─────────────────────
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
    <div style='display:flex;align-items:center;gap:0.7rem;margin:0.8rem 0 0.4rem;{rtl_css}'>
        <span style='font-size:1.07rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;'>{t("leaks_title")}:</span>
        <span style='font-size:1.1rem;'>{leak_icon}</span>
        <span style='font-size:1.17rem;color:{leak_color};font-weight:500;'>{leak_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Paint Analysis & Accident Indicator ──────────────────────────────────
    paint_data     = result.get("paint_data") or {}
    paint_findings = result.get("paint_findings") or []
    suspicion      = paint_data.get("suspicion", "none")

    section_label("paint_title")

    # Overall suspicion pill
    if suspicion == "high":
        sev_label, sev_color = t("paint_severity_high"),   "#B04040"
    elif suspicion in ("medium", "low"):
        sev_label, sev_color = t("paint_severity_medium"), "#C8A96A"
    else:
        sev_label, sev_color = t("paint_severity_low"),    "#4A7A4A"

    anomaly_count = len(paint_data.get("anomalies", []))
    summary_text  = t("paint_consistent") if suspicion == "none" else t("paint_suspect").format(count=anomaly_count)
    st.markdown(
        f"<div style='margin:0.3rem 0 0.6rem;{rtl_css}'>"
        f"<span style='font-size:1.15rem;font-weight:600;color:{sev_color};'>{sev_label}</span>"
        f"<span style='font-size:1.07rem;color:var(--muted);margin-{'right' if is_rtl else 'left'}:0.8rem;'> | {summary_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # OpenCV histogram anomalies (raw panel comparison)
    cv_anomalies = paint_data.get("anomalies", [])
    if cv_anomalies:
        for a in cv_anomalies[:5]:
            a_color = "#B04040" if a["severity"] == "high" else "#C8A96A"
            st.markdown(
                f"<div style='background:var(--elevated);border-left:3px solid {a_color};"
                f"padding:0.5rem 1rem;margin:0.25rem 0;border-radius:0 4px 4px 0;{rtl_css}'>"
                f"<span style='font-size:1rem;color:{a_color};'>◉ </span>"
                f"<span style='font-size:1.07rem;'>{t('paint_panel')}: <strong>{a['pair']}</strong>"
                f" | {t('paint_diff')}: {a['score']:.2f}</span></div>",
                unsafe_allow_html=True,
            )

    # Claude visual paint findings (overspray, orange peel, flake, hue, gaps)
    if paint_findings:
        for pf in paint_findings:
            sev = pf.get("severity", "low")
            fc  = {"high": "#B04040", "medium": "#C8A96A", "low": "#9A9080"}.get(sev, "#9A9080")
            st.markdown(
                f"<div style='background:var(--elevated);border-left:3px solid {fc};"
                f"padding:0.5rem 1rem;margin:0.25rem 0;border-radius:0 4px 4px 0;{rtl_css}'>"
                f"<span style='font-size:1rem;color:{fc};'>◎ </span>"
                f"<span style='font-size:1.07rem;'><strong>{pf.get('panel','')}</strong>"
                f": {pf.get('indicator','')}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<p style='font-size:0.95rem;color:var(--muted);margin-top:0.4rem;{rtl_css}'>◦ {t('paint_note')}</p>",
        unsafe_allow_html=True,
    )

    gold_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Mechanical Check (audio + underbody leaks + recalls)
    # ═══════════════════════════════════════════════════════════════════════════
    _section_header("🔧", "sec_mechanical")

    # ── Mechanical findings from top_reasons (audio / underbody / dashboard) ──
    sev_colors = {"high": "#B04040", "medium": "#C8A96A", "low": "#4A7A4A"}
    sev_icons  = {"high": "⚠", "medium": "◉", "low": "◎"}
    mech_evidence_types = {"audio", "underbody", "dashboard"}
    reasons = result.get("top_reasons", [])
    mech_reasons  = [r for r in reasons if r.get("evidence", {}).get("type") in mech_evidence_types
                     or r.get("severity") in ("high", "medium")]
    # Fall back to showing all if none classified
    if not mech_reasons:
        mech_reasons = reasons

    if mech_reasons:
        for r in mech_reasons:
            sev = r.get("severity", "low")
            bc  = sev_colors.get(sev, "#9A9080")
            ic  = sev_icons.get(sev, "◦")
            st.markdown(
                f"<div style='background:var(--elevated);border-left:3px solid {bc};"
                f"padding:0.75rem 1rem;margin:0.4rem 0;border-radius:0 4px 4px 0;{rtl_css}'>"
                f"<span style='color:{bc};margin-{'left' if is_rtl else 'right'}:0.5rem;'>{ic}</span>"
                f"<span style='font-size:1.24rem;'>{_tr_for_display(r.get('title',''))}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Audio Diagnosis Detail ────────────────────────────────────────────────
    audio_raw = result.get("audio_findings_raw", [])
    if audio_raw:
        st.markdown(
            f"<p style='font-size:1.07rem;color:var(--muted);letter-spacing:0.08em;margin-top:0.8rem;{rtl_css}'>"
            f"{t('audio_diagnosis')}</p>",
            unsafe_allow_html=True,
        )
        _finding_style = {
            "rod_knock_suspected":    ("🔴", "#B04040"),
            "valve_tick_suspected":   ("🟠", "#C8803A"),
            "belt_squeal_suspected":  ("🟠", "#C8803A"),
            "exhaust_leak_suspected": ("🟠", "#C8803A"),
            "rough_idle_suspected":   ("🟡", "#C8A96A"),
            "engine_sounds_normal":   ("🟢", "#4A7A4A"),
            "unknown":                ("⚪", "#9A9080"),
        }
        for f in audio_raw:
            lbl       = f.get("label", "unknown")
            icon_a, clr = _finding_style.get(lbl, ("◦", "#9A9080"))
            title     = _audio_label(lbl)
            details   = f.get("details", {})
            det_str   = "  ·  ".join(
                f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in details.items() if k not in ("reason",)
            )
            st.markdown(
                f"<div style='background:var(--elevated);border-left:3px solid {clr};"
                f"padding:0.6rem 1rem;margin:0.3rem 0;border-radius:0 4px 4px 0;{rtl_css}'>"
                f"<span style='font-size:1.15rem;'>{icon_a}&nbsp; {title}</span>"
                + (f"<div style='font-size:0.95rem;color:var(--muted);margin-top:0.15rem;'>{det_str}</div>" if det_str else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── NHTSA Recalls (inside mechanical section) ─────────────────────────────
    nhtsa = result.get("nhtsa_data") or {}
    recall_count     = nhtsa.get("recall_count", 0)
    total_complaints = nhtsa.get("total_complaints", 0)
    if recall_count > 0 or total_complaints > 0:
        st.markdown(
            f"<p style='font-size:1.07rem;color:var(--muted);letter-spacing:0.08em;margin-top:0.8rem;{rtl_css}'>"
            f"{t('recalls_title')}</p>",
            unsafe_allow_html=True,
        )
        if recall_count > 0:
            st.markdown(
                f"<div style='margin:0.3rem 0 0.4rem;{rtl_css}'>"
                f"<span style='color:#B04040;font-size:1.1rem;font-weight:600;'>⚠ {t('open_recalls')}: {recall_count}</span></div>",
                unsafe_allow_html=True,
            )
            for r in nhtsa.get("recalls", [])[:5]:
                comp    = r.get("component", "")
                summary = r.get("summary", "")[:200]
                st.markdown(
                    f"<div style='background:var(--elevated);border-left:3px solid #B04040;"
                    f"padding:0.5rem 1rem;margin:0.2rem 0;border-radius:0 4px 4px 0;{rtl_css}'>"
                    f"<span style='color:#B04040;font-size:0.97rem;font-weight:600;'>{comp}</span>"
                    f"<div style='font-size:1rem;margin-top:0.15rem;'>{summary}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<p style='color:#4A7A4A;font-size:1.05rem;{rtl_css}'>✓ {t('no_recalls_found')}</p>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='font-size:0.9rem;color:var(--muted);margin-top:0.3rem;{rtl_css}'>◦ {t('nhtsa_source')}</p>",
            unsafe_allow_html=True,
        )

    gold_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Summary & Conclusion (3-part verdict)
    # ═══════════════════════════════════════════════════════════════════════════
    _section_header("📋", "sec_conclusion")

    def _conclusion_row(icon: str, label_key: str, text: str, border_color: str = "var(--gold-dark)"):
        if not text:
            return
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:0.8rem;margin:0.5rem 0;{rtl_css}'>"
            f"<div style='min-width:28px;font-size:1.3rem;padding-top:0.1rem;'>{icon}</div>"
            f"<div style='flex:1;background:var(--elevated);border-left:3px solid {border_color};"
            f"border-radius:0 6px 6px 0;padding:0.6rem 1rem;'>"
            f"<div style='font-size:0.97rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;"
            f"margin-bottom:0.25rem;'>{t(label_key)}</div>"
            f"<div style='font-size:1.17rem;line-height:1.55;'>{text}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    conc_ext  = result.get("conclusion_external", "")
    conc_int  = result.get("conclusion_internal", "")
    conc_mech = result.get("conclusion_mechanical", "")

    _conclusion_row("🏠", "conc_external_label",   conc_ext,  "#4A7A4A" if ext_score and ext_score >= 7 else "#C8A96A")
    _conclusion_row("🪑", "conc_internal_label",   conc_int,  "#4A7A4A" if int_score and int_score >= 7 else "#C8A96A")
    _conclusion_row("⚙️", "conc_mechanical_label", conc_mech, color)  # color = verdict color

    # Fallback: show detailed report in expander if no conclusions
    report_text = result.get("detailed_report", "")
    if report_text and not (conc_ext or conc_int or conc_mech):
        with st.expander(t("detailed_report")):
            import re as _re
            sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', report_text) if s.strip()]
            for sentence in sentences:
                st.markdown(
                    f"<div style='border-left:2px solid rgba(200,169,106,0.3);padding:0.4rem 0.9rem;margin:0.3rem 0;{rtl_css}'>"
                    f"<span style='font-size:1.1rem;line-height:1.6;'>{sentence}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Recommended Next Steps ────────────────────────────────────────────────
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
                <div style='font-size:1.24rem;color:var(--text);padding-top:0.2rem;'>{_tr_for_display(s.get("text",""))}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── WhatsApp Share + HTML Report Download ────────────────────────────────
    import urllib.parse as _urllibparse
    import html as _html_mod
    _cd_share   = result.get("car_details", {})
    _make_s     = _cd_share.get("manufacturer", "")
    _model_s    = _cd_share.get("model_name", "")
    _year_s     = str(_cd_share.get("year", ""))
    _km_s       = _cd_share.get("odometer", "")
    _plate_s    = _cd_share.get("plate", "")
    _verdict_label_s, _verdict_hex_s, _ = verdict_meta(rec)

    # ── Pull all report text sections ────────────────────────────────────────
    _report_text  = result.get("detailed_report", "")
    _conc_ext     = result.get("conclusion_external", "")
    _conc_int     = result.get("conclusion_internal", "")
    _conc_mech    = result.get("conclusion_mechanical", "")
    _leak_txt     = result.get("leak_assessment", "")
    _rc_list_rep  = result.get("reject_codes") or []
    _yad2_rep     = result.get("yad2_price_data")
    _audio_raw    = result.get("audio_findings_raw") or []
    _paint_data   = result.get("paint_data") or {}
    _lang_rep     = st.session_state.get("lang", "he")
    _dir_attr     = "rtl" if is_rtl else "ltr"
    _verdict_colors_map = {"go": "#1b7c2e", "no_go": "#c62828", "inconclusive": "#b56a00"}
    _verdict_hex  = _verdict_colors_map.get(rec, "#555")

    # ── Reject codes rows ─────────────────────────────────────────────────────
    _rc_rows = ""
    for _rcode in _rc_list_rep:
        _ri = REJECT_TABLE.get(_rcode, {})
        _rt = _ri.get("title_he" if _lang_rep == "he" else "title_en", _rcode)
        _re = _ri.get("expl_he"  if _lang_rep == "he" else "expl_en",  "")
        _sev_colors = {"hard": "#c62828", "soft": "#b56a00", "tech": "#1565c0"}
        _sev_col = _sev_colors.get(_ri.get("severity", "soft"), "#555")
        _rc_rows += (
            f"<tr><td style='color:{_sev_col};font-weight:700;padding:4px 8px;'>{_html_mod.escape(_rcode)}</td>"
            f"<td style='padding:4px 8px;font-weight:600;'>{_html_mod.escape(_rt)}</td>"
            f"<td style='padding:4px 8px;color:#555;'>{_html_mod.escape(_re)}</td></tr>"
        )

    # ── Price HTML ────────────────────────────────────────────────────────────
    _price_html = ""
    if _yad2_rep and _yad2_rep.get("min_price") and _yad2_rep.get("max_price"):
        _min_rep = f"&#x20AA;{_yad2_rep['min_price']:,.0f}"
        _max_rep = f"&#x20AA;{_yad2_rep['max_price']:,.0f}"
        _price_lbl = "טווח מחיר שוק" if is_rtl else "Market Price Range"
        _price_html = f"<p style='font-size:1.15rem;'><strong>{_price_lbl}:</strong> {_min_rep} &ndash; {_max_rep}</p>"

    # ── Audio findings ────────────────────────────────────────────────────────
    _audio_html = ""
    if _audio_raw:
        _aud_lbl = "ממצאי שמע" if is_rtl else "Audio Findings"
        _aud_rows = "".join(
            f"<li>{_html_mod.escape(str(f.get('label','')))} "
            f"<span style='color:#888;font-size:0.88em;'>({f.get('confidence','')}) {_html_mod.escape(str(f.get('details','')))}</span></li>"
            for f in _audio_raw if f.get("label")
        )
        if _aud_rows:
            _audio_html = f"<h2>{_aud_lbl}</h2><ul>{_aud_rows}</ul>"

    # ── Paint section ─────────────────────────────────────────────────────────
    _paint_html = ""
    _paint_susp = _paint_data.get("suspicion", "none")
    if _paint_susp != "none":
        _paint_lbl  = "ניתוח צבע" if is_rtl else "Paint Analysis"
        _susp_map   = {"none": ("תקין","OK"), "low": ("חשד נמוך","Low suspicion"),
                       "medium": ("חשד בינוני","Medium suspicion"), "high": ("חשד גבוה","High suspicion")}
        _susp_he, _susp_en = _susp_map.get(_paint_susp, (_paint_susp, _paint_susp))
        _susp_txt = _susp_he if is_rtl else _susp_en
        _paint_html = f"<h2>{_paint_lbl}</h2><p>{_html_mod.escape(_susp_txt)}</p>"

    # ── Build full HTML report ─────────────────────────────────────────────────
    _date_str = datetime.now().strftime("%Y-%m-%d")
    _veh_lbl  = "רכב" if is_rtl else "Vehicle"
    _km_lbl   = "ק\"מ" if is_rtl else "km"
    _pl_lbl   = "לוחית" if is_rtl else "Plate"
    _own_lbl  = "בעלים" if is_rtl else "Owners"
    _prev_own = _cd_share.get("prev_owners", "")

    def _sec(title, body):
        if not body: return ""
        return f"<h2 style='margin-top:1.4rem;padding-bottom:0.3rem;border-bottom:2px solid #e0b84a;color:#333;'>{title}</h2>{body}"

    _body_sections = ""
    if _report_text:
        _body_sections += _sec("סיכום" if is_rtl else "Summary", f"<p>{_html_mod.escape(_report_text)}</p>")
    if _conc_ext:
        _body_sections += _sec("מצב חיצוני" if is_rtl else "Exterior", f"<p>{_html_mod.escape(_conc_ext)}</p>")
    if _conc_int:
        _body_sections += _sec("מצב פנים" if is_rtl else "Interior", f"<p>{_html_mod.escape(_conc_int)}</p>")
    if _conc_mech:
        _body_sections += _sec("מצב מכאני" if is_rtl else "Mechanical", f"<p>{_html_mod.escape(_conc_mech)}</p>")
    if _leak_txt and _leak_txt not in ("none detected", ""):
        _body_sections += _sec("דליפות" if is_rtl else "Leak Assessment", f"<p>{_html_mod.escape(_leak_txt)}</p>")
    _body_sections += _audio_html
    _body_sections += _paint_html
    if _rc_rows:
        _find_lbl = "ממצאים שזוהו" if is_rtl else "Issues Detected"
        _body_sections += _sec(_find_lbl,
            f"<table style='width:100%;border-collapse:collapse;font-size:0.95rem;'>"
            f"<thead><tr style='background:#f5f5f5;'>"
            f"<th style='padding:4px 8px;text-align:{'right' if is_rtl else 'left'};'>קוד</th>"
            f"<th style='padding:4px 8px;text-align:{'right' if is_rtl else 'left'};'>כותרת</th>"
            f"<th style='padding:4px 8px;text-align:{'right' if is_rtl else 'left'};'>הסבר</th>"
            f"</tr></thead><tbody>{_rc_rows}</tbody></table>")
    if _price_html:
        _body_sections += _sec("מחירון שוק" if is_rtl else "Market Price", _price_html)

    # ── PDF report — always generated in English for font compatibility ──────
    # (xhtml2pdf uses standard Latin fonts; Hebrew fonts are not bundled)
    _e = _html_mod.escape   # shorthand
    _pdf_verdict_map = {"go": "GO — Recommended", "no_go": "NO-GO — Not Recommended", "inconclusive": "INCONCLUSIVE — Further Inspection Needed"}
    _pdf_verdict_lbl = _pdf_verdict_map.get(rec, _verdict_label_s)
    _pdf_rc_rows = ""
    for _rcode in _rc_list_rep:
        _ri = REJECT_TABLE.get(_rcode, {})
        _rt = _ri.get("title_en", _rcode)
        _re = _ri.get("expl_en", "")
        _sev_colors = {"hard": "#c62828", "soft": "#b56a00", "tech": "#1565c0"}
        _sc = _sev_colors.get(_ri.get("severity", "soft"), "#555")
        _pdf_rc_rows += f"<tr><td style='color:{_sc};font-weight:bold;padding:4px 8px;'>{_e(_rcode)}</td><td style='padding:4px 8px;font-weight:600;'>{_e(_rt)}</td><td style='padding:4px 8px;color:#555;font-size:11px;'>{_e(_re)}</td></tr>"

    def _pdf_sec(title, body):
        if not body: return ""
        return f"<h2 style='font-size:13px;margin-top:14px;padding-bottom:3px;border-bottom:1px solid #c8a96a;color:#333;'>{title}</h2>{body}"

    _pdf_body = ""
    if _report_text:  _pdf_body += _pdf_sec("Summary", f"<p style='font-size:11px;'>{_e(_report_text)}</p>")
    if _conc_ext:     _pdf_body += _pdf_sec("Exterior", f"<p style='font-size:11px;'>{_e(_conc_ext)}</p>")
    if _conc_int:     _pdf_body += _pdf_sec("Interior", f"<p style='font-size:11px;'>{_e(_conc_int)}</p>")
    if _conc_mech:    _pdf_body += _pdf_sec("Mechanical", f"<p style='font-size:11px;'>{_e(_conc_mech)}</p>")
    if _leak_txt and _leak_txt not in ("none detected", "none", ""):
        _pdf_body += _pdf_sec("Leak Assessment", f"<p style='font-size:11px;color:#c62828;font-weight:600;'>{_e(_leak_txt)}</p>")
    if _audio_raw:
        _au = "".join(f"<li style='font-size:11px;'>{_e(str(f.get('label','')))} ({f.get('confidence','')}) {_e(str(f.get('details','')))}</li>" for f in _audio_raw if f.get("label"))
        if _au: _pdf_body += _pdf_sec("Audio Findings", f"<ul>{_au}</ul>")
    if _paint_susp != "none":
        _pdf_body += _pdf_sec("Paint Analysis", f"<p style='font-size:11px;'>Suspicion level: {_e(_paint_susp)}</p>")
    if _pdf_rc_rows:
        _pdf_body += _pdf_sec("Issues Detected",
            f"<table style='width:100%;border-collapse:collapse;'><thead><tr style='background:#f5f5f5;'>"
            f"<th style='padding:4px 8px;text-align:left;font-size:11px;'>Code</th>"
            f"<th style='padding:4px 8px;text-align:left;font-size:11px;'>Issue</th>"
            f"<th style='padding:4px 8px;text-align:left;font-size:11px;'>Details</th>"
            f"</tr></thead><tbody>{_pdf_rc_rows}</tbody></table>")
    if _yad2_rep and _yad2_rep.get("min_price"):
        _pdf_body += _pdf_sec("Market Price",
            f"<p style='font-size:12px;'>ILS {_yad2_rep['min_price']:,.0f} &ndash; {_yad2_rep['max_price']:,.0f}</p>")

    _pdf_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 1.8cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; color: #222; font-size: 12px; line-height: 1.55; }}
  .header {{ border-bottom: 2px solid #c8a96a; padding-bottom: 8px; margin-bottom: 10px; }}
  .app-name {{ font-size: 10px; color: #c8a96a; letter-spacing: 2px; text-transform: uppercase; }}
  .car-name {{ font-size: 18px; font-weight: bold; margin: 2px 0 6px; }}
  .meta {{ font-size: 10px; color: #666; margin-bottom: 4px; }}
  .verdict-box {{ padding: 10px 14px; margin: 10px 0; border-left: 5px solid {_verdict_hex}; background: {_verdict_hex}12; }}
  .verdict-lbl {{ font-size: 16px; font-weight: bold; color: {_verdict_hex}; }}
  .footer {{ margin-top: 18px; padding-top: 6px; border-top: 1px solid #eee; font-size: 9px; color: #aaa; text-align: center; }}
</style>
</head>
<body>
<div class="header">
  <div class="app-name">UsedCar Check — Inspection Report</div>
  <div class="car-name">{_e(_make_s)} {_e(_model_s)} {_e(_year_s)}</div>
  <div class="meta">Plate: {_e(str(_plate_s or "—"))} &nbsp;|&nbsp; Odometer: {_e(str(_km_s) if _km_s else "—")} km &nbsp;|&nbsp; Date: {_date_str}</div>
</div>
<div class="verdict-box">
  <div class="verdict-lbl">{_e(_pdf_verdict_lbl)}</div>
</div>
{_pdf_body}
<div class="footer">UsedCar Check &mdash; usedcar-check-if-the-car-is-worth-it.streamlit.app</div>
</body>
</html>"""

    # Convert HTML to PDF using xhtml2pdf
    try:
        import io as _io
        from xhtml2pdf import pisa as _pisa
        _pdf_buf = _io.BytesIO()
        _pisa.CreatePDF(_pdf_html, dest=_pdf_buf, encoding="utf-8")
        _report_bytes = _pdf_buf.getvalue()
        _report_mime = "application/pdf"
        _report_ext  = "pdf"
    except Exception:
        # Fallback to HTML if xhtml2pdf unavailable
        _report_bytes = _pdf_html.encode("utf-8")
        _report_mime = "text/html"
        _report_ext  = "html"

    # ── WhatsApp message (rich summary) ──────────────────────────────────────
    _wa_lines = []
    if is_rtl:
        _wa_lines += [f"✅ בדיקת רכב | {_make_s} {_model_s} {_year_s}",
                      f"🔑 תוצאה: {_verdict_label_s}"]
        if _km_s: _wa_lines.append(f"🛣 קילומטראז': {_km_s:,} ק\"מ" if isinstance(_km_s, int) else f"🛣 קילומטראז': {_km_s} ק\"מ")
        if _conc_ext:   _wa_lines.append(f"🚗 חוץ: {_conc_ext[:120]}")
        if _conc_mech:  _wa_lines.append(f"🔧 מכאני: {_conc_mech[:120]}")
        if _rc_list_rep:
            _codes_txt = ", ".join(_rc_list_rep)
            _wa_lines.append(f"⚠️ קודי ממצא: {_codes_txt}")
        _wa_lines.append("📋 הורד את הדוח המלא מהאפליקציה")
        _wa_lines.append("🔗 usedcar-check-if-the-car-is-worth-it.streamlit.app")
    else:
        _wa_lines += [f"✅ Car Check | {_make_s} {_model_s} {_year_s}",
                      f"🔑 Result: {_verdict_label_s}"]
        if _km_s: _wa_lines.append(f"🛣 Odometer: {_km_s} km")
        if _conc_ext:   _wa_lines.append(f"🚗 Exterior: {_conc_ext[:120]}")
        if _conc_mech:  _wa_lines.append(f"🔧 Mechanical: {_conc_mech[:120]}")
        if _rc_list_rep:
            _wa_lines.append(f"⚠️ Issues: {', '.join(_rc_list_rep)}")
        _wa_lines.append("📋 Download full report from the app")
        _wa_lines.append("🔗 usedcar-check-if-the-car-is-worth-it.streamlit.app")
    _wa_url = "https://wa.me/?text=" + _urllibparse.quote("\n".join(_wa_lines))

    gold_divider()
    _wa_col, _dl_col = st.columns([1, 1])
    with _wa_col:
        st.markdown(
            f"<a href='{_wa_url}' target='_blank' rel='noopener' "
            f"style='background:#25D366;color:white;border-radius:8px;padding:0.55rem 1.2rem;"
            f"text-decoration:none;display:inline-block;font-size:1rem;font-weight:600;width:100%;text-align:center;box-sizing:border-box;'>"
            f"{t('whatsapp_share')}</a>",
            unsafe_allow_html=True,
        )
    with _dl_col:
        st.download_button(
            label=t("download_report"),
            data=_report_bytes,
            file_name=f"car_check_{_year_s}_{_model_s}.{_report_ext}",
            mime=_report_mime,
            use_container_width=True,
        )

    # ── Yad2 Market Price Reference ───────────────────────────────────────────
    gold_divider()
    car_d    = result.get("car_details", {})
    _yr      = car_d.get("year", "")
    _mk      = car_d.get("manufacturer", "")
    _mdl     = car_d.get("model_name", "")
    _search  = f"{_yr} {_mk} {_mdl}".strip()

    import urllib.parse as _up
    _yad2pd = result.get("yad2_price_data")

    # Deep-link to the Yad2 pricelist page (not the vehicle search listings)
    _pricelist_base = "https://www.yad2.co.il/price-list"
    _mfr_id = (_yad2pd or {}).get("mfr_id")
    _yad2_url = f"{_pricelist_base}?manufacturer={_mfr_id}" if _mfr_id else _pricelist_base
    _google_url = "https://www.google.com/search?q=" + _up.quote(f'site:yad2.co.il מחירון {_search}')

    if _yad2pd and _yad2pd.get("min_price") and _yad2pd.get("max_price"):
        _min_p = f"₪{_yad2pd['min_price']:,.0f}"
        _max_p = f"₪{_yad2pd['max_price']:,.0f}"
        _matched = _yad2pd.get("matched_model", _search)
        _exact   = _yad2pd.get("exact_match", False)
        _match_label = ("✓ " if _exact else "~") + _matched
        st.markdown(
            f"<div class='mobile-card' style='background:rgba(200,169,106,0.08);border:1px solid rgba(200,169,106,0.35);"
            f"border-radius:8px;padding:1rem 1.4rem;margin:0.4rem 0;{rtl_css}'>"
            f"<div style='font-size:1rem;color:var(--gold);letter-spacing:0.1em;text-transform:uppercase;"
            f"margin-bottom:0.6rem;'>💰 {t('yad2_price_label')}</div>"
            f"<div style='font-size:2rem;font-weight:700;color:var(--text);letter-spacing:0.03em;'>"
            f"{_min_p} &ndash; {_max_p}</div>"
            f"<div style='font-size:0.95rem;color:var(--muted);margin-top:0.3rem;'>"
            f"{t('yad2_price_model')} <span style='color:var(--text);'>{_match_label}</span></div>"
            f"<div style='font-size:0.88rem;color:var(--muted);margin-top:0.25rem;'>{t('yad2_price_note')}</div>"
            f"<div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.8rem;'>"
            f"<a href='{_yad2_url}' target='_blank' rel='noopener' "
            f"style='display:inline-block;background:rgba(200,169,106,0.12);color:var(--gold);"
            f"border:1px solid var(--gold-dark);border-radius:5px;padding:0.4rem 1rem;"
            f"font-size:0.97rem;text-decoration:none;'>🔍 {t('yad2_ref_btn')}</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:rgba(200,169,106,0.06);border:1px solid rgba(200,169,106,0.25);"
            f"border-radius:8px;padding:1rem 1.4rem;margin:0.4rem 0;{rtl_css}'>"
            f"<div style='font-size:1rem;color:var(--gold);letter-spacing:0.1em;text-transform:uppercase;"
            f"margin-bottom:0.5rem;'>💰 {t('yad2_ref_label')}</div>"
            f"<div style='font-size:1.07rem;color:var(--muted);margin-bottom:0.7rem;'>{t('yad2_ref_hint')}</div>"
            f"<div style='font-size:1.05rem;color:var(--muted);margin-bottom:0.6rem;'>"
            f"<span style='color:var(--gold);'>{t('yad2_search_hint')}</span> "
            f"<strong style='color:var(--text);'>{_search}</strong></div>"
            f"<div style='display:flex;gap:0.8rem;flex-wrap:wrap;'>"
            f"<a href='{_yad2_url}' target='_blank' rel='noopener' "
            f"style='display:inline-block;background:rgba(200,169,106,0.12);color:var(--gold);"
            f"border:1px solid var(--gold-dark);border-radius:5px;padding:0.45rem 1.1rem;"
            f"font-size:1rem;text-decoration:none;letter-spacing:0.05em;'>"
            f"🔍 {t('yad2_ref_btn')}</a>"
            f"<a href='{_google_url}' target='_blank' rel='noopener' "
            f"style='display:inline-block;background:transparent;color:var(--muted);"
            f"border:1px solid var(--border);border-radius:5px;padding:0.45rem 1.1rem;"
            f"font-size:1rem;text-decoration:none;'>"
            f"🌐 {'חיפוש Google' if is_rtl else 'Google Search'}</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Damage History ───────────────────────────────────────────────────────
    gold_divider()
    _r02_flagged = "R02" in (result.get("reject_codes") or [])
    _r02_note_he = "⚠️ זוהתה אי-עקביות בצבע (R02) — ייתכן שהרכב עבר תיקון גוף או צביעה מחדש." if _r02_flagged else "לא זוהתה אי-עקביות בצבע בבדיקה הנוכחית."
    _r02_note_en = "⚠️ Paint inconsistency detected (R02) — the vehicle may have had body repair or repainting." if _r02_flagged else "No paint inconsistency detected in this check."
    _r02_note    = _r02_note_he if is_rtl else _r02_note_en
    st.markdown(
        f"<div class='mobile-card' style='background:rgba(106,143,170,0.07);border:1px solid rgba(106,143,170,0.3);"
        f"border-radius:8px;padding:1rem 1.4rem;margin:0.4rem 0;{rtl_css}'>"
        f"<div style='font-size:1rem;color:#6A8FAA;letter-spacing:0.1em;text-transform:uppercase;"
        f"margin-bottom:0.6rem;'>🔍 {t('damage_history_title')}</div>"
        f"<div style='font-size:1rem;color:var(--text);margin-bottom:0.6rem;'>{_r02_note}</div>"
        f"<div style='font-size:0.97rem;color:var(--muted);margin-bottom:0.5rem;'>{t('damage_history_hint')}</div>"
        f"<div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;'>"
        f"<a href='https://www.check-car.co.il' target='_blank' rel='noopener' "
        f"style='display:inline-block;background:rgba(106,143,170,0.12);color:#6A8FAA;"
        f"border:1px solid rgba(106,143,170,0.4);border-radius:5px;padding:0.4rem 1rem;"
        f"font-size:0.97rem;text-decoration:none;'>check-car.co.il</a>"
        f"</div>"
        f"<div style='font-size:0.88rem;color:var(--muted);font-style:italic;'>"
        f"◦ {t('damage_history_external')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Refine Analysis ───────────────────────────────────────────────────────
    gold_divider()
    st.markdown(
        f"<div style='font-size:1.3rem;font-family:Cormorant Garamond,serif;font-weight:600;"
        f"letter-spacing:0.12em;color:var(--gold);text-transform:uppercase;margin-bottom:0.5rem;{rtl_css}'>"
        f"🔄 {t('refine_title')}</div>"
        f"<p style='font-size:1.05rem;color:var(--muted);margin-bottom:0.8rem;{rtl_css}'>"
        f"{t('refine_banner')}</p>",
        unsafe_allow_html=True,
    )
    _rcol1, _rcol2, _rcol3 = st.columns(3)
    with _rcol1:
        if st.button(f"🚗 {t('refine_details_btn')}", key="refine_details", use_container_width=True):
            st.session_state.original_result = result
            st.session_state.refine_mode     = True
            st.session_state.step            = 1
            st.rerun()
    with _rcol2:
        if st.button(f"📸 {t('refine_photos_btn')}", key="refine_photos", use_container_width=True):
            st.session_state.original_result = result
            st.session_state.refine_mode     = True
            st.session_state.step            = 2
            st.rerun()
    with _rcol3:
        if st.button(f"🎙 {t('refine_audio_btn')}", key="refine_audio", use_container_width=True):
            st.session_state.original_result = result
            st.session_state.refine_mode     = True
            st.session_state.step            = 3
            st.rerun()

# ─── Step indicator ───────────────────────────────────────────────────────────
STEP_ICONS = ["🚗", "📸", "🎙", "✓"]

def step_indicator(current: int):
    """Pure-HTML step bar — never uses st.columns so mobile CSS can't stack it."""
    labels = [t("step_details"), t("step_photos"), t("step_audio"), t("step_result")]
    items  = ""
    for i, (name, icon) in enumerate(zip(labels, STEP_ICONS), 1):
        active = i == current
        done   = i < current
        bg_c   = "rgba(200,169,106,0.15)" if active else ("rgba(74,122,74,0.1)" if done else "rgba(44,44,44,0.4)")
        border = "var(--gold)" if active else ("#4A7A4A" if done else "var(--border)")
        txt    = "var(--gold)" if active else ("#4A7A4A" if done else "var(--muted)")
        items += (
            f"<div style='flex:1;text-align:center;min-width:0;'>"
            f"<div style='width:36px;height:36px;border-radius:50%;background:{bg_c};"
            f"border:1px solid {border};display:flex;align-items:center;justify-content:center;"
            f"margin:0 auto 0.3rem;font-size:1.1rem;'>"
            f"<span style='color:{border};'>{icon}</span></div>"
            f"<div style='font-size:0.78rem;letter-spacing:0.06em;text-transform:uppercase;"
            f"color:{txt};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{name}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='display:flex;gap:4px;margin:0.6rem 0 0.4rem;align-items:flex-start;'>{items}</div>",
        unsafe_allow_html=True,
    )
    gold_divider()

# ─── Login screen ─────────────────────────────────────────────────────────────
def login_screen():
    # ── Hero banner with car image (flags embedded inside) ───────────────────
    st.markdown(f"""
    <div class='login-hero-wrap' style="
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
        <div class='login-hero-title' style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:5.5rem;
                    letter-spacing:0.22em;color:#C8A96A;text-transform:uppercase;
                    text-shadow:0 2px 30px rgba(0,0,0,0.9);line-height:1.05;'>
            {t("app_title")}
        </div>
        <div style='height:1px;width:80px;background:linear-gradient(90deg,transparent,#C8A96A,transparent);
                    margin:1.2rem auto;'></div>
        <div class='login-hero-subtitle' style='font-size:1.7rem;letter-spacing:0.1em;color:rgba(240,235,224,0.85);font-style:italic;'>
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

        # Language toggle — flag radio (markdown images work on all OS)
        _FLAG_IL = "![IL](https://flagcdn.com/w40/il.png)"
        _FLAG_US = "![US](https://flagcdn.com/w40/us.png)"
        _login_lang = st.radio("", [_FLAG_IL, _FLAG_US],
                               index=0 if st.session_state.lang == "he" else 1,
                               horizontal=True, label_visibility="collapsed",
                               key="login_lang_radio")
        if _login_lang == _FLAG_IL and st.session_state.lang != "he":
            st.session_state.lang = "he"; st.rerun()
        elif _login_lang == _FLAG_US and st.session_state.lang != "en":
            st.session_state.lang = "en"; st.rerun()

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # Custom labels rendered ABOVE the field — label_visibility="collapsed" prevents
        # Streamlit's own label from sitting inside or on top of the typed text
        st.markdown(f"<p style='font-size:1rem;color:var(--muted);margin:0 0 0.2rem;{rtl_css}'>{t('email_label')}</p>", unsafe_allow_html=True)
        email = st.text_input("", placeholder="your@email.com", key="login_email", label_visibility="collapsed")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        if st.button(t("enter_btn"), use_container_width=True):
            if not email or "@" not in email or "." not in email.split("@")[-1]:
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

        _disc_rtl = "direction:rtl;text-align:right;" if st.session_state.lang == "he" else "direction:ltr;text-align:left;"
        st.markdown(f"""
        <div style='margin-top:1.4rem;padding:0.9rem 1.1rem;
                    border:1px solid var(--gold);border-radius:8px;
                    background:rgba(212,175,55,0.06);'>
            <p style='font-size:0.78rem;color:var(--muted);line-height:1.55;
                      margin:0 0 0.55rem;{_disc_rtl}'>
                {t('disclaimer')}
            </p>
            <p style='font-size:0.76rem;color:var(--gold);margin:0;
                      font-style:italic;{_disc_rtl}'>
                {t('disclaimer_accept')}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        _FLAG_IL = "![IL](https://flagcdn.com/w40/il.png)"
        _FLAG_US = "![US](https://flagcdn.com/w40/us.png)"
        _sb_lang = st.radio("", [_FLAG_IL, _FLAG_US],
                            index=0 if st.session_state.lang == "he" else 1,
                            horizontal=True, label_visibility="collapsed",
                            key="sb_lang_radio")
        if _sb_lang == _FLAG_IL and st.session_state.lang != "he":
            st.session_state.lang = "he"; st.rerun()
        elif _sb_lang == _FLAG_US and st.session_state.lang != "en":
            st.session_state.lang = "en"; st.rerun()

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
            st.session_state.step            = 1
            st.session_state.car_details     = {}
            st.session_state.photos          = []
            st.session_state.underbody       = None
            st.session_state.vehicle_video   = None
            st.session_state.audio           = None
            st.session_state.result          = None
            st.session_state.original_result = None
            st.session_state.refine_mode     = False
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

# ─── Ministry of Transport plate lookup ──────────────────────────────────────
# Hebrew make names as returned by data.gov.il → English keys in CAR_MAKES_MODELS
_HE_MAKE_MAP: dict[str, str] = {
    # ── Base names ──────────────────────────────────────────────────────────────
    "טויוטה": "Toyota", "הונדה": "Honda", "מזדה": "Mazda", "ניסאן": "Nissan",
    "יונדאי": "Hyundai", "קיה": "Kia", "סובארו": "Subaru", "מיצובישי": "Mitsubishi",
    "פולקסווגן": "Volkswagen", "אאודי": "Audi", "בי.מ.וו": "BMW", "ב.מ.וו": "BMW",
    "מרצדס": "Mercedes-Benz", "מרצדס בנץ": "Mercedes-Benz", "פורד": "Ford",
    # ── Country-suffixed variants (Ministry of Transport format) ────────────────
    "סקודה צ'כיה": "Škoda", "סקודה": "Škoda", "שקודה": "Škoda",
    "טויוטה יפן": "Toyota", "הונדה יפן": "Honda", "מזדה יפן": "Mazda",
    "ניסאן יפן": "Nissan", "מיצובישי יפן": "Mitsubishi", "סובארו יפן": "Subaru",
    "יונדאי קוריאה": "Hyundai", "קיה קוריאה": "Kia",
    "פולקסווגן גרמניה": "Volkswagen", "אאודי גרמניה": "Audi",
    "מרצדס גרמניה": "Mercedes-Benz", "מרצדס בנץ גרמניה": "Mercedes-Benz",
    "בי.מ.וו גרמניה": "BMW", "ב.מ.וו גרמניה": "BMW", "אופל גרמניה": "Opel",
    "פורד אמריקה": "Ford", "פורד ארה\"ב": "Ford",
    "וולוו שבדיה": "Volvo", "וולוו": "Volvo",
    "רנו צרפת": "Renault", "סיטרואן צרפת": "Citroën", "פיג'ו צרפת": "Peugeot",
    "פיאט איטליה": "Fiat", "אלפא רומאו איטליה": "Alfa Romeo",
    "ג'יפ אמריקה": "Jeep", "ג'יפ": "Jeep",
    "שברולט אמריקה": "Chevrolet", "דאצ'יה רומניה": "Dacia",
    "שברולט": "Chevrolet", "אופל": "Opel", "פיאט": "Fiat", "פיג'ו": "Peugeot",
    "רנו": "Renault", "סיטרואן": "Citroën", "שקודה": "Škoda", "סיאט": "Seat",
    "וולוו": "Volvo", "ג'יפ": "Jeep", "סוזוקי": "Suzuki", "מיני": "MINI",
    "פורשה": "Porsche", "לנד רובר": "Land Rover", "ג'אגואר": "Jaguar",
    "לקסוס": "Lexus", "אינפיניטי": "Infiniti", "טסלה": "Tesla",
    "סיאט": "Seat", "דאצ'יה": "Dacia", "סקודה": "Škoda", "קאדילק": "Cadillac",
    "ב.מ.וו.": "BMW", "אי.מ.ג'י": "MG", "מ.ג.": "MG",
}

def _fetch_vehicle_by_plate(plate: str) -> dict | None:
    """
    Query the Israeli Ministry of Transport open data API.
    Returns a dict with year, manufacturer (English), model_name on success,
    or None if not found / API unavailable.
    """
    import requests as _req
    plate = plate.strip().replace("-", "").replace(" ", "")
    if not plate:
        return None
    try:
        url = (
            "https://data.gov.il/api/3/action/datastore_search"
            "?resource_id=053cea08-09bc-40ec-8f7a-156f0677aff3"
            f"&q={plate}&limit=5"
        )
        resp = _req.get(url, timeout=6)
        data = resp.json()
        records = data.get("result", {}).get("records", [])
        # Find exact plate match (q= does substring; verify exact)
        match = next(
            (r for r in records if str(r.get("mispar_rechev", "")).strip() == plate),
            records[0] if records else None,
        )
        if not match:
            return None

        he_make   = (match.get("tozeret_nm") or "").strip()
        he_model  = (match.get("kinuy_mishari") or match.get("degem_nm") or "").strip()
        year_raw  = match.get("shnat_yitzur") or match.get("shnat_yitzur_dt", "")
        try:
            year = int(str(year_raw)[:4])
        except Exception:
            year = None

        # Try full name first; if unmapped, strip trailing country word and retry
        en_make = _HE_MAKE_MAP.get(he_make)
        if en_make is None:
            _words = he_make.split()
            if len(_words) > 1:
                en_make = _HE_MAKE_MAP.get(" ".join(_words[:-1]))
            if en_make is None:
                en_make = _HE_MAKE_MAP.get(_words[0], he_make)

        # ── Extra registry fields ──────────────────────────────────────────────
        baalut_raw    = (match.get("baalut") or "").strip()
        color_he      = (match.get("tzeva_rechev") or "").strip()
        fuel_he       = (match.get("sug_delek_nm") or "").strip()
        last_test_raw = (match.get("mivchan_acharon_dt") or "")[:10]  # YYYY-MM-DD
        valid_until   = (match.get("tokef_dt") or "")[:10]
        first_road    = (match.get("moed_aliya_lakvish") or "")       # "2011-1"
        trim_val      = (match.get("ramat_gimur") or "").strip()

        # Normalise baalut to a canonical English key
        _baalut_map = {
            "פרטי": "private", "ליסינג": "lease", "השכרה": "rental",
            "ממשלתי": "govt", "עסקי": "company", "חברה": "company",
            "מסחרי": "company",
        }
        ownership_key = _baalut_map.get(baalut_raw, "private" if not baalut_raw else "other")

        # Map ownership → usage_type index (0=private,1=rental/lease,2=company,3=unknown)
        _usage_map = {"private": 0, "lease": 1, "rental": 1, "govt": 2, "company": 2, "other": 3}
        usage_type_from_plate = _usage_map.get(ownership_key, 0)

        return {
            "manufacturer":    en_make,
            "model_name":      he_model,
            "year":            year,
            "plate":           plate,
            "trim":            trim_val,           # fills the Trim field directly
            "usage_type":      usage_type_from_plate,  # pre-selects usage dropdown
            "_he_make":        he_make,
            # Registry extras (kept through continue → results)
            "_baalut_he":      baalut_raw,
            "_ownership_key":  ownership_key,
            "_color_he":       color_he,
            "_fuel_he":        fuel_he,
            "_last_test":      last_test_raw,
            "_valid_until":    valid_until,
            "_first_road":     first_road,
        }
    except Exception:
        return None


# ─── Yad2 Pricelist fetch ─────────────────────────────────────────────────────
# Module-level cache: (build_id, {he_make: yad2_id}, fetched_at)
_YAD2_CACHE: dict = {}

# English make → Hebrew as Yad2 spells it (for reverse-lookup against their manufacturer list)
_EN_TO_YAD2_HE: dict[str, str] = {
    "Audi": "אאודי", "BMW": "ב מ וו", "Honda": "הונדה", "Toyota": "טויוטה",
    "Hyundai": "יונדאי", "Mazda": "מאזדה", "Mercedes-Benz": "מרצדס-בנץ",
    "Nissan": "ניסאן", "Volkswagen": "פולקסווגן", "Ford": "פורד", "Kia": "קיה",
    "Tesla": "טסלה", "Subaru": "סובארו", "Mitsubishi": "מיצובישי",
    "Volvo": "וולוו", "Jeep": "ג'יפ", "Suzuki": "סוזוקי", "MINI": "מיני",
    "Porsche": "פורשה", "Land Rover": "לנד רובר", "Jaguar": "ג'אגואר",
    "Lexus": "לקסוס", "Infiniti": "אינפיניטי", "Seat": "סיאט",
    "Škoda": "סקודה", "Citroën": "סיטרואן", "Peugeot": "פיג'ו",
    "Renault": "רנו", "Fiat": "פיאט", "Opel": "אופל", "Chevrolet": "שברולט",
    "Dacia": "דאציה", "Alfa Romeo": "אלפא רומיאו", "BYD": "ב.י.ד",
    "MG": "מ.ג.", "Cupra": "קופרה", "Genesis": "ג'נסיס",
}


def _fetch_yad2_price(manufacturer_en: str, model_name: str, year: int) -> dict | None:
    """
    Fetch Yad2 book-value price range for a specific car.
    Returns dict: {min_price, max_price, matched_model, year, currency='ILS'} or None.
    Caches the build_id + manufacturer map for up to 6 hours.
    """
    import requests as _req, re as _re, json as _json, time as _time

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*",
    }

    # ── Step 1: resolve Yad2 manufacturer ID ─────────────────────────────────
    he_name = _EN_TO_YAD2_HE.get(manufacturer_en)
    if not he_name:
        return None   # unsupported make

    now = _time.time()
    cache = _YAD2_CACHE

    # Refresh build_id + make map if stale (> 6 h) or missing
    if not cache or (now - cache.get("fetched_at", 0)) > 21_600:
        try:
            r = _req.get("https://www.yad2.co.il/price-list",
                         headers=HEADERS, timeout=8)
            m = _re.search(r'"buildId":"([^"]+)"', r.text)
            nd_m = _re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                r.text, _re.DOTALL)
            if not m or not nd_m:
                return None
            build_id = m.group(1)
            nd = _json.loads(nd_m.group(1))
            # Manufacturers live under pageProps.manufacturers or similar path
            mfrs_raw = (
                nd.get("props", {}).get("pageProps", {}).get("manufacturers")
                or nd.get("props", {}).get("pageProps", {}).get("carManufacturers")
                or []
            )
            make_map: dict[str, int] = {}
            for entry in mfrs_raw:
                mfr_id  = entry.get("id") or entry.get("manufacturerId")
                mfr_he  = (entry.get("manufacturer") or entry.get("name") or "").strip()
                if mfr_id and mfr_he:
                    make_map[mfr_he] = int(mfr_id)
            cache.update({"build_id": build_id, "make_map": make_map,
                          "fetched_at": now})
        except Exception:
            return None

    build_id = cache.get("build_id")
    make_map  = cache.get("make_map", {})
    mfr_id    = make_map.get(he_name)
    if not mfr_id or not build_id:
        return None

    # ── Step 2: fetch models for this manufacturer ────────────────────────────
    try:
        feed_url = (f"https://www.yad2.co.il/price-list"
                    f"/_next/data/{build_id}/feed.json")
        resp = _req.get(feed_url, params={"manufacturer": mfr_id},
                        headers={**HEADERS, "Accept": "application/json"},
                        timeout=8)
        models = resp.json().get("pageProps", {}).get("models", [])
    except Exception:
        return None

    if not models:
        return None

    # ── Step 3: match by model name + year ───────────────────────────────────
    def _norm(s: str) -> str:
        """Lowercase, strip punctuation/spaces for fuzzy matching."""
        import unicodedata
        s = unicodedata.normalize("NFKD", str(s)).lower()
        return _re.sub(r"[^a-z0-9\u05d0-\u05ea]", "", s)

    model_norm = _norm(model_name)

    # Score each model: +2 for name match, +1 for year match (±1 tolerance)
    scored: list[tuple[int, dict]] = []
    for m in models:
        score = 0
        m_name_norm = _norm(m.get("model", ""))
        if model_norm and m_name_norm:
            if model_norm in m_name_norm or m_name_norm in model_norm:
                score += 2
        m_year = m.get("year")
        if m_year and year and abs(int(m_year) - int(year)) <= 1:
            score += 1
        if score > 0:
            scored.append((score, m))

    # Pick best match; fall back to year-only match across all models
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
    else:
        # Year fallback — collect all models within ±2 years
        year_matches = [m for m in models
                        if m.get("year") and abs(int(m.get("year")) - int(year)) <= 2]
        if not year_matches:
            return None
        best = None
        # Return aggregated range across year matches
        all_min = [m["minPrice"] for m in year_matches if m.get("minPrice")]
        all_max = [m["maxPrice"] for m in year_matches if m.get("maxPrice")]
        if not all_min:
            return None
        return {
            "min_price":     min(all_min),
            "max_price":     max(all_max) if all_max else min(all_min),
            "matched_model": f"{manufacturer_en} ({year})",
            "year":          year,
            "currency":      "ILS",
            "source":        "yad2",
            "exact_match":   False,
            "mfr_id":        mfr_id,
        }

    min_p = best.get("minPrice") or 0
    max_p = best.get("maxPrice") or min_p

    # Also consider sub-model prices for tighter range
    sub_prices = [s.get("price") for s in best.get("subModels", []) if s.get("price")]
    if sub_prices:
        min_p = min(sub_prices)
        max_p = max(sub_prices)

    if not min_p:
        return None

    return {
        "min_price":     min_p,
        "max_price":     max_p,
        "matched_model": f"{best.get('model', '')} {best.get('year', '')}".strip(),
        "year":          best.get("year", year),
        "currency":      "ILS",
        "source":        "yad2",
        "exact_match":   True,
        "mfr_id":        mfr_id,
    }


# ─── Step 1 — Vehicle Details ─────────────────────────────────────────────────
def step_vehicle_details():
    section_label("vehicle_details")
    d = st.session_state.car_details

    # ── Plate number auto-fill ────────────────────────────────────────────────
    # Outer column limits the TOTAL plate section to ~40% of page width.
    # Inner columns split that 40% between the input (60%) and button (40%).
    _plate_outer, _ = st.columns([5, 7])
    with _plate_outer:
        st.markdown(
            f"<p style='font-size:0.82rem;color:var(--muted);margin:0 0 0.2rem;{rtl_css}'>"
            f"🔍 {t('plate_label')}</p>",
            unsafe_allow_html=True,
        )
        with st.form(key="plate_form", border=False):
            _pi, _pb = st.columns([3, 2])
            with _pi:
                plate_input = st.text_input(
                    "", value=d.get("plate", ""),
                    placeholder="12-345-67",
                    key="plate_number", label_visibility="collapsed",
                )
            with _pb:
                plate_btn = st.form_submit_button(
                    t("plate_lookup_btn"), use_container_width=True
                )

    if plate_btn and plate_input.strip():
        with st.spinner("" if is_rtl else ""):
            _pdata = _fetch_vehicle_by_plate(plate_input.strip())
        if _pdata:
            # Merge into car_details — keep existing values that API doesn't cover
            d.update({k: v for k, v in _pdata.items() if v})
            st.session_state.car_details = d
            # Explicitly sync widget states — Streamlit ignores value= when a
            # key already exists in session_state, so we must update it directly
            if _pdata.get("trim"):
                st.session_state["trim_input"] = _pdata["trim"]
            if _pdata.get("model_name"):
                st.session_state["model_free"] = _pdata["model_name"]
            # Store a flash message that survives the rerun
            st.session_state["_plate_msg"] = (
                t("plate_found")
                + f"  |  {_pdata.get('_he_make','')} {_pdata.get('model_name','')} {_pdata.get('year','')}"
            )
            st.rerun()
        else:
            st.warning(t("plate_not_found"))

    # Show flash message set by previous run (survives rerun)
    if st.session_state.get("_plate_msg"):
        st.success(st.session_state.pop("_plate_msg"))

    gold_divider()

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
            # Case-insensitive lookup (MOT API returns UPPERCASE model names like "ENYAQ")
            # Falls back to startswith match for longer MOT names like "ENYAQ IV 80" → "Enyaq"
            model_idx = 0
            if saved_model:
                _sm_lower = saved_model.lower()
                for _mi, _mn in enumerate(models_list):
                    if _mn.lower() == _sm_lower:
                        model_idx = _mi; break
                else:
                    # Partial match: "ENYAQ IV 80" → "Enyaq", or "ENYAQ" → "Enyaq iV"
                    for _mi, _mn in enumerate(models_list):
                        if _mn and (_sm_lower.startswith(_mn.lower()) or _mn.lower().startswith(_sm_lower)):
                            model_idx = _mi; break
            model_name  = st.selectbox(
                t("model"),
                options=models_list,
                index=model_idx,
                format_func=lambda x: t("select_model") if x == "" else x,
            )
        else:
            st.markdown(f"<p style='font-size:1rem;color:var(--muted);margin:0 0 0.2rem;{rtl_css}'>{t('model')}</p>", unsafe_allow_html=True)
            model_name = st.text_input("", value=d.get("model_name", ""), key="model_free", label_visibility="collapsed")

        year     = st.number_input(t("year"), min_value=1990, max_value=2026, step=1,
                                   value=int(d.get("year", 2020)))
        odometer = st.number_input(t("odometer"), min_value=0, max_value=2_000_000,
                                   step=1000, value=int(d.get("odometer", 0)))

    with col2:
        st.markdown(f"<p style='font-size:1rem;color:var(--muted);margin:0 0 0.2rem;{rtl_css}'>{t('trim')}</p>", unsafe_allow_html=True)
        trim = st.text_input("", value=d.get("trim", ""), placeholder=t("trim_ph"), key="trim_input", label_visibility="collapsed")

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
    if st.button(t("continue_btn"), use_container_width=True):
        # Start from existing dict so registry fields (underscore-prefixed) are preserved
        new_details = {k: v for k, v in d.items() if k.startswith("_")}
        new_details.update({
            "manufacturer": manufacturer,
            "model_name":   model_name,
            "year":         int(year),
            "trim":         trim.strip(),
            "odometer":     int(odometer),
            "usage_type":   int(usage_type),
            "prev_owners":  int(prev_owners),
            "plate":        st.session_state.get("plate_number", d.get("plate", "")),
        })
        st.session_state.car_details = new_details
        st.session_state.step = 2
        st.rerun()

# ─── Image pre-validation ─────────────────────────────────────────────────────
def _validate_photos(photo_files: list, manufacturer: str, model_name: str) -> list[str]:
    """Quick Claude Haiku check: unrelated images, two vehicles, brand mismatch.
    Returns list of warning strings (empty = all clear)."""
    import base64, json, tempfile
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key or len(photo_files) == 0:
            return []
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Sample up to 8 photos — more coverage = better mixed-vehicle detection
        sample = photo_files[:8]
        img_blocks = []
        for f in sample:
            try:
                f.seek(0)
                raw = f.read()
                f.seek(0)
                ext = f.name.rsplit(".", 1)[-1].lower()
                media = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                         "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
                img_blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": media,
                    "data": base64.standard_b64encode(raw).decode()}})
            except Exception:
                continue
        if not img_blocks:
            return []
        mfr_hint = f"{manufacturer} {model_name}".strip() or "unknown"
        prompt = f"""You are a strict photo validator for a used-car inspection app. The user selected vehicle: {mfr_hint}.

Examine ALL uploaded images carefully. Answer ONLY with a valid JSON object (no markdown, no prose):
{{
  "all_are_cars": true/false,
  "same_vehicle": true/false,
  "brand_matches": true/false/null,
  "detected_brand": "<visible brand name, or null>",
  "notes": "<brief reason if any field is false, else empty string>"
}}

Rules:
- "all_are_cars": false if ANY image is clearly not a vehicle (e.g. a selfie, food, landscape, screenshot, document)
- "same_vehicle": false if the images show MORE THAN ONE DISTINCT VEHICLE — different colors, body styles, or clearly different cars mixed in the same set. Look carefully at body shape, color, trim, and interior differences across photos.
- "brand_matches": false ONLY if you can clearly see a brand logo (badge, emblem, steering wheel logo) that contradicts "{manufacturer}". Set to null if no logo is clearly visible.
- Do NOT be lenient on "same_vehicle" — if something looks like two different cars, flag it."""
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": img_blocks + [{"type": "text", "text": prompt}]}]
        )
        raw_json = resp.content[0].text.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1].lstrip("json").strip()
        data = json.loads(raw_json)
        warnings = []
        if not data.get("all_are_cars", True):
            warnings.append("unrelated")
        if not data.get("same_vehicle", True):
            warnings.append("two_vehicles")
        if data.get("brand_matches") is False and manufacturer:
            detected = data.get("detected_brand") or ""
            if detected and detected.lower() not in manufacturer.lower() and manufacturer.lower() not in detected.lower():
                warnings.append(f"brand_mismatch:{manufacturer}")
        return warnings
    except Exception as _ve:
        # Surface error as a gentle warning rather than silent fail
        return [f"validation_error:{str(_ve)[:80]}"]


# ─── Step 2 — Photos ──────────────────────────────────────────────────────────
def step_photos():
    # ── Refine mode banner ────────────────────────────────────────────────────
    if st.session_state.get("refine_mode") and st.session_state.get("original_result"):
        st.markdown(
            f"<div style='background:rgba(200,169,106,0.10);border:1px solid rgba(200,169,106,0.4);"
            f"border-radius:6px;padding:0.7rem 1rem;margin-bottom:0.8rem;{rtl_css}'>"
            f"🔄 {t('refine_banner')}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"📋 {t('view_original_btn')}", expanded=False):
            _orig2 = st.session_state.original_result
            _o_label, _o_color, _ = verdict_meta(_orig2.get("recommendation", "inconclusive"))
            st.markdown(f"<span style='color:{_o_color};font-weight:700;font-size:1.3rem;'>{_o_label}</span>",
                        unsafe_allow_html=True)
            _rpt = _orig2.get("detailed_report", "")
            if _rpt:
                st.markdown(f"<p style='font-size:1.05rem;{rtl_css}'>{_rpt}</p>", unsafe_allow_html=True)

    # ── Saved photos notice (refine mode) ────────────────────────────────────
    _saved_photos   = st.session_state.get("photos") or []
    _saved_interior = st.session_state.get("interior_photos") or []
    _saved_underbody= st.session_state.get("underbody")
    _saved_video    = st.session_state.get("vehicle_video")
    _is_refine      = st.session_state.get("refine_mode") and st.session_state.get("original_result")
    if _is_refine and _saved_photos:
        st.markdown(
            f"<div style='background:rgba(74,122,74,0.12);border:1px solid rgba(74,122,74,0.4);"
            f"border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.6rem;{rtl_css}'>"
            f"{t('prev_photos_kept').format(n=len(_saved_photos))}</div>",
            unsafe_allow_html=True,
        )

    # ── Exterior photos ───────────────────────────────────────────────────────
    section_label("vehicle_photos")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>"
                f"{'העלה 3–8 תמונות חיצוניות: כל הצדדים, תא המנוע, גלגלים. ודא תאורה טובה.' if is_rtl else 'Upload 3–8 exterior photos: all sides, engine bay, wheels. Ensure good lighting.'}"
                f"</p>", unsafe_allow_html=True)
    photos_new = st.file_uploader(t("vehicle_photos"), type=["jpg","jpeg","png","webp"],
                              accept_multiple_files=True, label_visibility="collapsed",
                              key="exterior_upload")
    # Merge: new uploads replace saved only if user uploaded something new
    photos = photos_new if photos_new else _saved_photos
    if photos:
        color = "#4A7A4A" if 3 <= len(photos) <= 8 else "#B04040"
        _src = f" ({'חדשות' if is_rtl else 'new'})" if photos_new else (f" ({'שמורות' if is_rtl else 'saved'})" if _is_refine and _saved_photos else "")
        st.markdown(f"<p style='font-size:1.01rem;color:{color};margin-top:0.4rem;'>📸 {len(photos)} {t('photos_count')}{_src}</p>", unsafe_allow_html=True)

    gold_divider()

    # ── Interior photos ───────────────────────────────────────────────────────
    section_label("interior_photos_title")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('interior_photos_hint')}</p>", unsafe_allow_html=True)
    interior_photos_new = st.file_uploader(t("interior_photos_title"), type=["jpg","jpeg","png","webp"],
                                       accept_multiple_files=True, label_visibility="collapsed",
                                       key="interior_upload")
    interior_photos = interior_photos_new if interior_photos_new else _saved_interior
    if interior_photos:
        color = "#4A7A4A" if 1 <= len(interior_photos) <= 6 else "#C8A96A"
        st.markdown(f"<p style='font-size:1.01rem;color:{color};margin-top:0.4rem;'>🪑 {len(interior_photos)} {t('interior_photos_count')}</p>", unsafe_allow_html=True)

    gold_divider()

    # ── Underbody ─────────────────────────────────────────────────────────────
    section_label("underbody_title")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('underbody_hint')}</p>", unsafe_allow_html=True)
    underbody_new = st.file_uploader(t("underbody_title"), type=["jpg","jpeg","png"],
                                 label_visibility="collapsed", key="underbody_upload")
    underbody = underbody_new if underbody_new else _saved_underbody

    gold_divider()

    # ── Video ─────────────────────────────────────────────────────────────────
    section_label("vehicle_video")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('video_hint')}</p>", unsafe_allow_html=True)
    vehicle_video_new = st.file_uploader(t("vehicle_video"), type=["mp4","mov","avi","mkv","webm"],
                                     label_visibility="collapsed", key="vehicle_video_upload")
    vehicle_video = vehicle_video_new if vehicle_video_new else _saved_video
    if vehicle_video:
        size_mb = len(vehicle_video.getvalue()) / (1024 * 1024)
        st.markdown(f"<p style='font-size:1.01rem;color:#4A7A4A;margin-top:0.3rem;'>🎬 {vehicle_video.name} ({size_mb:.1f} MB)</p>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn"), use_container_width=True):
            st.session_state.step = 1; st.rerun()
    with col2:
        if st.button(t("continue_btn"), key="photos_continue", use_container_width=True):
            if not photos or len(photos) < 3:
                st.error("אנא העלה לפחות 3 תמונות חיצוניות." if is_rtl else "Please upload at least 3 exterior photos.")
            elif len(photos) > 8:
                st.error("מקסימום 8 תמונות חיצוניות." if is_rtl else "Maximum 8 exterior photos.")
            else:
                # ── Image validation (quick Haiku check) ──────────────────
                _d = st.session_state.get("car_details", {})
                _mfr = _d.get("manufacturer", "")
                _mdl = _d.get("model_name", "")
                # Only validate newly uploaded photos (not saved ones from prior check)
                _photos_to_validate = photos_new if photos_new else photos
                with st.spinner(t("img_validation_running")):
                    _warnings = _validate_photos(_photos_to_validate, _mfr, _mdl)
                _block = False
                for _w in _warnings:
                    if _w == "unrelated":
                        st.warning(t("img_warn_unrelated"))
                    elif _w == "two_vehicles":
                        st.error(t("img_warn_two_vehicles"))
                        _block = True
                    elif _w.startswith("brand_mismatch:"):
                        _sel = _w.split(":", 1)[1]
                        st.warning(t("img_warn_brand_mismatch").format(selected=_sel))
                    elif _w.startswith("validation_error:"):
                        pass  # silent — don't block user on API error
                if not _block:
                    st.session_state.photos          = photos
                    st.session_state.interior_photos = interior_photos or []
                    st.session_state.underbody       = underbody
                    st.session_state.vehicle_video   = vehicle_video
                    st.session_state.step            = 3; st.rerun()

# ─── Step 3 — Audio ───────────────────────────────────────────────────────────
def step_audio():
    # ── Refine mode banner ────────────────────────────────────────────────────
    if st.session_state.get("refine_mode") and st.session_state.get("original_result"):
        st.markdown(
            f"<div style='background:rgba(200,169,106,0.10);border:1px solid rgba(200,169,106,0.4);"
            f"border-radius:6px;padding:0.7rem 1rem;margin-bottom:0.8rem;{rtl_css}'>"
            f"🔄 {t('refine_banner')}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"📋 {t('view_original_btn')}", expanded=False):
            _orig3 = st.session_state.original_result
            _oa_label, _oa_color, _ = verdict_meta(_orig3.get("recommendation", "inconclusive"))
            st.markdown(f"<span style='color:{_oa_color};font-weight:700;font-size:1.3rem;'>{_oa_label}</span>",
                        unsafe_allow_html=True)
            _rpt3 = _orig3.get("detailed_report", "")
            if _rpt3:
                st.markdown(f"<p style='font-size:1.05rem;{rtl_css}'>{_rpt3}</p>", unsafe_allow_html=True)

    section_label("engine_audio")
    st.markdown(f"<p style='font-size:1.24rem;color:var(--muted);{rtl_css}'>{t('audio_hint')}</p>", unsafe_allow_html=True)
    audio = st.file_uploader(t("engine_audio"), type=["mp3","wav","m4a","ogg","aac","flac","mp4","mov","avi","mkv","webm","3gp"], label_visibility="collapsed")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("back_btn"), key="audio_back", use_container_width=True):
            st.session_state.step = 2; st.rerun()
    with col2:
        if st.button(t("analyse_btn"), use_container_width=True):
            if not audio:
                st.error(t("audio_missing"))
            else:
                d = st.session_state.car_details
                car_label = f"{d.get('year','')} {d.get('manufacturer','')} {d.get('model_name','')}".strip()
                import threading, time as _time
                global _ACTIVE_LANG
                _lang = st.session_state.get("lang", "he")
                _ACTIVE_LANG = _lang   # expose to thread-safe getter
                _photos    = list(st.session_state.get("photos", []))
                _interior  = list(st.session_state.get("interior_photos", []))
                _underbody = st.session_state.get("underbody")
                _video     = st.session_state.get("vehicle_video")
                _stages = [
                    (0.10, "מנתח תמונות חיצוניות של הרכב..."        if _lang=="he" else "Analysing exterior photos..."),
                    (0.22, "בודק מצב הפנים והלוח..."                if _lang=="he" else "Checking interior & dashboard..."),
                    (0.36, "מעבד הקלטת קול המנוע..."                if _lang=="he" else "Processing engine audio..."),
                    (0.50, "מנתח עקביות צבע לוחות הרכב..."          if _lang=="he" else "Analysing paint panel consistency..."),
                    (0.63, "מאחזר נתוני ריקול ובטיחות (NHTSA)..."   if _lang=="he" else "Fetching safety & recall data (NHTSA)..."),
                    (0.78, "מייצר דוח מקצועי מבוסס AI..."           if _lang=="he" else "Generating AI professional report..."),
                    (0.92, "מסיים ומאמת תוצאות..."                  if _lang=="he" else "Finalising and validating results..."),
                ]
                _result_box  = [None]
                _error_box   = [None]
                def _worker():
                    try:
                        _result_box[0] = run_analysis(
                            d, _photos, audio, _underbody, _video, _interior,
                        )
                    except Exception as _e:
                        _error_box[0] = _e
                _thread = threading.Thread(target=_worker, daemon=True)
                _thread.start()
                _prog_bar  = st.progress(0.0)
                _prog_text = st.empty()
                _stage_i, _elapsed = 0, 0.0
                _MIN_SECS  = 15.0   # always show progress for at least this long
                _rtl = rtl_css if _lang == "he" else ""

                def _render_stage(elapsed: float):
                    """Update bar + stage label for the given elapsed time."""
                    pct = min(0.95, elapsed / _MIN_SECS)
                    _prog_bar.progress(pct)
                    si = 0
                    for _si2, (_thresh, _) in enumerate(_stages):
                        if pct >= _thresh:
                            si = _si2
                    _prog_text.markdown(
                        f"<p style='font-size:1.05rem;color:var(--gold);{_rtl}'>"
                        f"⚙️ &nbsp;{_stages[si][1]}</p>",
                        unsafe_allow_html=True,
                    )

                # Phase 1 — tick while analysis is running
                while _thread.is_alive():
                    _time.sleep(0.4)
                    _elapsed += 0.4
                    _render_stage(_elapsed)

                _thread.join()

                # Phase 2 — analysis finished but minimum time not yet reached
                while _elapsed < _MIN_SECS:
                    _time.sleep(0.4)
                    _elapsed += 0.4
                    _render_stage(_elapsed)

                _prog_bar.progress(1.0)
                _prog_text.empty()
                if _error_box[0]:
                    st.error(f"{t('analysis_failed')}: {_error_box[0]}")
                else:
                    decision, audio_dur, ai_report, nhtsa_data, audio_metrics, audio_findings_raw, paint_data, yad2_price_data, reject_codes = _result_box[0]
                    result = {
                        "recommendation":      decision.recommendation,
                        "confidence":          decision.confidence,
                        "top_reasons":         decision.top_reasons,
                        "breakdown":           decision.breakdown,
                        "education":           decision.education,
                        "next_steps":          decision.next_steps,
                        "audio_duration_seconds": audio_dur,
                        "car_details":         d,
                        "car_label":           car_label,
                        "exterior_score":      ai_report.get("exterior_score"),
                        "interior_score":      ai_report.get("interior_score"),
                        "leak_assessment":     ai_report.get("leak_assessment", "none detected"),
                        "detailed_report":     ai_report.get("report", ""),
                        "nhtsa_data":          nhtsa_data,
                        "audio_metrics":       audio_metrics,
                        "audio_findings_raw":  audio_findings_raw,
                        "paint_data":              paint_data,
                        "paint_findings":          ai_report.get("paint_findings", []),
                        "conclusion_external":     ai_report.get("conclusion_external", ""),
                        "conclusion_internal":     ai_report.get("conclusion_internal", ""),
                        "conclusion_mechanical":   ai_report.get("conclusion_mechanical", ""),
                        "yad2_price_data":         yad2_price_data,
                        "reject_codes":            reject_codes,
                        "plate_registry": {
                            "ownership_key":  d.get("_ownership_key", ""),
                            "baalut_he":      d.get("_baalut_he", ""),
                            "color_he":       d.get("_color_he", ""),
                            "fuel_he":        d.get("_fuel_he", ""),
                            "last_test":      d.get("_last_test", ""),
                            "valid_until":    d.get("_valid_until", ""),
                            "first_road":     d.get("_first_road", ""),
                            "trim":           d.get("_trim", ""),
                            "plate":          d.get("plate", ""),
                        } if d.get("plate") else None,
                        "data_quality": {
                            "exterior_count":  len(_photos),
                            "interior_count":  len(_interior),
                            "has_underbody":   _underbody is not None,
                            "has_video":       _video is not None,
                            "audio_duration":  audio_dur,
                        },
                    }
                    check_id = save_check(st.session_state.email, result)
                    result["check_id"]      = check_id
                    result["created_at"]    = result.get("created_at", "")
                    # Send results to user's email (silently — never block on failure)
                    _send_result_email(
                        st.session_state.email, result,
                        st.session_state.get("lang", "he")
                    )
                    st.session_state.result      = result
                    st.session_state.refine_mode = False   # refine complete
                    st.session_state.step        = 4; st.rerun()

# ─── Main app ─────────────────────────────────────────────────────────────────
def main_app():
    render_sidebar()

    # Compact header with animated car silhouette
    st.markdown(f"""
    <div style='text-align:center;padding:1.2rem 0 0.2rem;overflow:hidden;'>
        <div class='car-animated' style='margin-bottom:0.4rem;'>
            <img class='hero-img' src='data:image/png;base64,{_PORSCHE_B64}'
                 style='height:110px;width:auto;filter:drop-shadow(0 4px 14px rgba(200,169,106,0.45));'/>
        </div>
        <div class='hero-title' style='font-family:Cormorant Garamond,serif;font-weight:300;font-size:2.9rem;
                    letter-spacing:0.18em;color:var(--gold);text-transform:uppercase;line-height:1.1;'>
            {t("app_title")}
        </div>
        <div class='hero-subtitle' style='font-size:1.85rem;letter-spacing:0.08em;color:var(--muted);margin-top:0.4rem;font-style:italic;'>
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
