## UI Requirements (Android-first)

### Car Selection Form
1. **Make**: dropdown sourced from `GET /v1/reference/makes` (A-Z list).
2. **Year**: dropdown sourced from `GET /v1/reference/years` (1990-2026).
3. **Exact model**: free text input.
4. **Driven KM**: numeric input (manual) and required for better engine-risk context.
5. **Trim**: optional text input.

### Media Upload Screen
- Allow selecting **multiple photos in one action**.
- Add helper text near the upload control:
  - "Tip: In Windows file picker, hold Ctrl + left-click to select multiple images."
  - "You can also hold Shift to select a range."
- Required: minimum 4 photos, maximum 10.
- Optional slot selector per upload:
  - `front`, `rear`, `dashboard`, `engine_bay`, `tire`, `interior`, `underbody`.
- Special option:
  - **Underbody (bottom) photo** to screen for possible oil leaks/serious undercarriage damage clues.

### Engine Audio Capture
- Show minimum required duration: **10 seconds**.
- If uploaded audio is shorter than 10 seconds, show backend message:
  - `audio_too_short_min_10_seconds`

### Results Screen
- Always show:
  - Recommendation (`go`, `no_go`, `inconclusive`)
  - Confidence (`high`, `medium`, `low`)
  - Audio length in seconds (`audio_duration_seconds`)
  - Car details (`manufacturer`, `model_name`, `year`, `trim`, `driven_km`)
- Show category breakdown including:
  - `dashboard` (hook for warning-light detections)
  - `tires_suspension` (includes underbody findings)
