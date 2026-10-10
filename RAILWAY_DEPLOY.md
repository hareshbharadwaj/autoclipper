# Deploying AutoClipper to Railway

This guide walks you through deploying AutoClipper to [Railway](https://railway.app). AutoClipper is structured with clean, separate `backend/` and `frontend/` folders, and supports two deployment options:

1. **Option A (Recommended — Single Service):** Runs both FastAPI backend and React frontend together in one container. Simplest setup, zero CORS configuration, uses half the RAM, and saves money on Railway.
2. **Option B (Separate Services):** Deploys `backend` and `frontend` as two distinct Railway services with independent scaling.

---

## Required Environment Variables

Configure these in your Railway service settings under **Variables**:

| Variable | Description | Example / Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (for viral hook and highlight detection) | `AIzaSy...` |
| `GROQ_API_KEY` | Groq API key (for Whisper transcription) | `gsk_...` |
| `AUTOCLIPPER_SECRET` | Secret key for JWT sessions and credential encryption at rest | `random_long_string_32_chars` |
| `OUTPUT_ROOT` | Directory where videos and clips are stored | `/app/output` |
| `AUTOCLIPPER_DB` | SQLite database file path | `/app/data/autoclipper.db` |
| `MAX_CONCURRENT_JOBS` | Maximum simultaneous video processing jobs | `1` (or `2` if plan has >= 2GB RAM) |

---

## Option A: Single Service Deployment (Recommended)

In this mode, Railway builds the React frontend in Stage 1, packages Python + FFmpeg + MediaPipe in Stage 2, and FastAPI serves both the API endpoints (`/api/...`) and the React dashboard (`/`) on a single port.

### Step-by-Step:
1. Log in to [Railway](https://railway.app) and click **New Project**.
2. Select **Deploy from GitHub repo** and choose your `autoclipper` repository.
3. Railway will detect the root `Dockerfile` and `railway.json`.
4. Go to **Variables** and add:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `AUTOCLIPPER_SECRET` (generate any random string)
   - `OUTPUT_ROOT=/app/output`
   - `AUTOCLIPPER_DB=/app/data/autoclipper.db`
5. *(Recommended)* Add Persistent Storage:
   - Click your service -> **Data** (or Settings -> Volumes) -> **Add Volume**.
   - Set the mount path to `/app/data` (to preserve user accounts, settings, and library).
   - Add a second volume mounted to `/app/output` if you want generated clips to persist across redeployments.
6. Click **Deploy**.
7. Under **Settings** -> **Networking**, click **Generate Domain** (e.g. `autoclipper-production.up.railway.app`).
8. Open the URL in your browser — your fullstack AutoClipper dashboard is live!

---

## Option B: Separate Services (Backend + Frontend)

If you prefer independent scaling or want to serve the frontend via Nginx CDN:

### 1. Deploy the Backend Service
1. In your Railway project, click **New Service** -> **GitHub Repo** -> choose `autoclipper`.
2. Name it `autoclipper-backend`.
3. In **Settings** -> **Source**:
   - Set **Root Directory** to `/backend`.
   - Railway will build `backend/Dockerfile`.
4. In **Variables**, add:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `AUTOCLIPPER_SECRET`
   - `OUTPUT_ROOT=/app/output`
   - `AUTOCLIPPER_DB=/app/data/autoclipper.db`
5. Under **Networking**, click **Generate Domain** and copy your backend URL (e.g. `https://autoclipper-backend-production.up.railway.app`).
6. Attach volumes for `/app/data` and `/app/output`.

### 2. Deploy the Frontend Service
1. Click **New Service** -> **GitHub Repo** -> choose `autoclipper`.
2. Name it `autoclipper-frontend`.
3. In **Settings** -> **Source**:
   - Set **Root Directory** to `/frontend`.
   - Railway will build `frontend/Dockerfile`.
4. In **Variables**, add:
   - `VITE_API_URL=https://autoclipper-backend-production.up.railway.app` (use your backend URL from Step 1).
5. Under **Networking**, click **Generate Domain**.
6. Open your frontend domain in the browser!

---

## Railway Resource Recommendations

- **RAM:** Video processing with FFmpeg + MediaPipe face tracking works best with at least **1.5 GB to 2 GB RAM**. On Railway's free/starter plan, set `MAX_CONCURRENT_JOBS=1` to ensure video encoding stays well within memory limits.
- **CPU:** Railway automatically provides burstable vCPU. The optimized AI reframer samples at ~6 fps and runs on downscaled 640px frames, keeping CPU usage minimal.

---

## Local Development

You can still develop locally using standard commands:

### Running Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Running Frontend:
```bash
cd frontend
npm install
npm run dev
```

### Running with Docker Compose:
```bash
docker compose up --build
```
