## Used Car Check (local MVP)

This is a local-first MVP for a used-car “GO / NO GO” preliminary assessment:

- Upload **4–10 photos** + **engine audio**
- Backend runs basic **quality checks** (blur/brightness) + **coarse audio heuristics**
- Returns **GO / INCONCLUSIVE** (and later we’ll add more “NO GO” detectors)
- Accepts car details (make/model/year/trim/driven_km)
- Supports optional **underbody (bottom)** photo slot for leak/damage hints

### Prereqs
- Docker Desktop (Windows)

### Run locally
From `C:\Users\ypola\Downloads\usedcar-check`:

```bash
docker compose up --build
```

API will be at `http://localhost:8000`.

### Quick smoke test (no app yet)
1) Create a check:

```bash
curl -X POST http://localhost:8000/v1/checks -H "Content-Type: application/json" -d "{\"nickname\":\"test\",\"make\":\"Toyota\",\"model_name\":\"Corolla\",\"year\":2017,\"trim\":\"LE\",\"driven_km\":120000}"
```

2) Upload at least 4 photos:

```bash
curl -X POST "http://localhost:8000/v1/checks/<CHECK_ID>/photos" -F "files=@photo1.jpg" -F "files=@photo2.jpg" -F "files=@photo3.jpg" -F "files=@photo4.jpg"
```

In the API docs file picker you can upload many images in one shot:
- Windows: hold **Ctrl + left-click** each image, then Open.
- Another common way: hold **Shift** to select a range of images.

Optional underbody/bottom image slot example:

```bash
curl -X POST "http://localhost:8000/v1/checks/<CHECK_ID>/photos?slot=underbody" -F "files=@underbody.jpg"
```

3) Upload audio:

```bash
curl -X POST "http://localhost:8000/v1/checks/<CHECK_ID>/audio" -F "file=@engine.wav"
```

Note: minimum audio length is **10 seconds**.

4) Start analysis:

```bash
curl -X POST "http://localhost:8000/v1/checks/<CHECK_ID>/analyze"
```

5) Poll job + fetch result:

```bash
curl http://localhost:8000/v1/jobs/<JOB_ID>
curl http://localhost:8000/v1/checks/<CHECK_ID>/result
```

### Reference data for UI dropdowns
- Makes A-Z list: `GET /v1/reference/makes`
- Years 1990-2026: `GET /v1/reference/years`

