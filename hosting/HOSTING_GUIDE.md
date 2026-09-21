# Hosting the dashboard

`app.py` is a Streamlit dashboard: sliders for every
model parameter, the optimal trajectory plot, the efficient frontier, and a
one-click verification check. This guide covers three ways to put it on the
actual internet, cheapest and simplest first.

## Option 1: Streamlit Community Cloud (recommended, free, no card)

This is the simplest path and the one used elsewhere in this repo (see
`data_engineer/25_cdc_vector_freshness_pipeline/hosting/HOSTING_GUIDE.md` for
the same platform used with a different project).

1. Push this repo to GitHub (a public repo is required for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click "New app," pick this repository, and set:
   - **Main file path**: `app.py`
   - Streamlit Cloud automatically looks for a `requirements.txt` next to
     the main file: `requirements.txt` already has one,
     including a `-e .` line that installs the local `acexec` package itself
     (see `requirements.txt`'s comment). No extra
     configuration needed.
4. Click "Deploy." The first build takes a couple of minutes (installing
   `numpy`/`scipy`/`pandas`/`matplotlib`/`streamlit`); after that, every push
   to the branch you picked redeploys automatically.

You'll get a public URL like `https://<your-app-name>.streamlit.app`.

## Option 2: Docker, on Render / Railway / Fly.io

`Dockerfile` builds a self-contained image: it installs
`requirements.txt` (which installs `acexec` itself via `-e .`), copies the
rest of the project in, and starts Streamlit listening on `$PORT` (the
environment variable every one of these hosts sets automatically).

**Verified locally before writing this guide**: `docker build` completes
cleanly, and `docker run` serves a healthy Streamlit instance — confirmed
by polling `/_stcore/health` until it returned `ok` and checking the
container logs showed the app fully started with no errors, not just
assumed from reading the Dockerfile.

Local test, if you want to repeat it yourself:

```
docker build -t acexec-dashboard .
docker run -p 8501:8501 acexec-dashboard
# open http://localhost:8501
```

Deploying to a host:

- **Render**: New → Web Service → connect this repo → leave the root
  directory at the repo root →
  Render detects the `Dockerfile` automatically. Render sets `$PORT` itself.
- **Railway**: New Project → Deploy from GitHub repo → leave the root
  directory at the repo root → Railway also builds the `Dockerfile` and sets
  `$PORT` automatically.
- **Fly.io**: run `fly launch` from the repo root (it detects
  the `Dockerfile`), then `fly deploy`.

All three have a free tier sufficient for a personal portfolio project like
this one (the app is stateless and computationally light — every calculation
is closed-form algebra or a quick NumPy sweep, no heavy background jobs).

## Option 3: Hugging Face Spaces

Spaces has a native Streamlit SDK, free, no Docker needed:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   choose the **Streamlit** SDK.
2. Push this repo's contents to the Space's git
   repo (Spaces are themselves git repos) — `app.py`, `requirements.txt`,
   and the `acexec/` package folder all need to be present at the Space's
   root (Spaces don't support pointing at a subdirectory the way Streamlit
   Community Cloud does, and this repo already has them at its root).

## Which one to actually pick

For a personal project meant to be linked from a resume or portfolio,
**Option 1 (Streamlit Community Cloud)** is the right default — it needs no
Docker knowledge, deploys straight from this existing repo with no
restructuring, and costs nothing. Reach for Option 2 (Docker) only if you
want a custom domain, more control over resources, or you're already
using Render/Railway/Fly for other projects and want everything in one
place.

See `deploy_checklist.md` before your first deploy.
