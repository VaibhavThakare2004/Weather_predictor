Deploying Frontend to Vercel (local) and Backend to Render (repo-based build)

This document explains two common, secure ways to deploy the app without requiring you to run `docker login` locally.

Prerequisites
- Vercel account and Vercel CLI installed (`npm i -g vercel`)
- Render account
- A Git provider account (GitHub, GitLab, or Bitbucket) to host the repo that Render will build from
- Your environment variables ready (OPENWEATHER_API_KEY, GOOGLE_API_KEY)

Part A — Frontend: Deploy to Vercel directly from local
1. Install Vercel CLI (if not installed):
   npm i -g vercel

2. From project `frontend` folder, run:
   cd D:\Wheather\frontend
   vercel login
   vercel

   - Follow prompts to link the project. When asked for project name, use something like `weather-frontend`.
   - Set `REACT_APP_API_URL` environment variable in the Vercel dashboard (after backend is live) or during `vercel env add`.

3. On every update, run `vercel --prod` to push a production deployment.

Part B — Backend: Let Render build from your repository (no docker login required)
This is the recommended option if you don't want to push images from your machine. Render will clone your git repo, detect the Dockerfile, build the image, and deploy it.

High-level steps
1. Create a git repository on GitHub/GitLab/Bitbucket and push your project (a single branch is fine).
   - Before pushing: ensure you have an appropriate `.gitignore` that excludes `node_modules/` and other large build artifacts.
   - If your model files are large, either:
     - use Git LFS for the model files, or
     - do NOT push the models and instead host them in cloud storage (S3/GCS) and add logic in `api.py` to download them at startup (set the URL in `MODEL_XGB_URL`).

2. In the Render dashboard, click 'New' → 'Web Service' → 'Connect a repository'
   - Select your Git provider and authorize Render to access the repo.
   - Choose the repo and branch to deploy.
   - Environment: select "Docker" (Render will build from your `Dockerfile`).
   - Dockerfile path: `/` (or `./Dockerfile` if in repo root).
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`.

3. Add the required environment variables in the Render service settings:
   - `OPENWEATHER_API_KEY`
   - `GOOGLE_API_KEY`
   - `MODEL_XGB_URL` (optional, if you host the pretrained model externally)

4. Create the service. Render will build the Docker image and deploy it — you don't need to push or host images yourself, so `docker login` is unnecessary.

Optional: If you still prefer image-based deployment (local build + push), follow the previous `docker build`/`docker push` flow — but it's not required for Render's repo-based builds.

Notes & tips
- Keep `.env` local and never push it. Use Render's env-vars UI.
- If models are large, hosting them in object storage and downloading at startup makes the repo small and keeps deployments fast.
- You can still keep `render.yaml` in the repo to define services as infra-as-code; Render will use it when connected to the repo.

Commands you may use locally to prepare the repo (PowerShell):
```powershell
# initialize and push
git init
git add .
git commit -m "Initial commit"
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

That's it — this flow avoids `docker login` because Render builds and hosts the Docker image for you.