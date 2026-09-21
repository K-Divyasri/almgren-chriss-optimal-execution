# Deploy checklist

Run through this before deploying the dashboard anywhere.

## Before your first deploy

- [ ] `pip install -e ".[dashboard]"` succeeds locally
- [ ] `python -m pytest -q` passes (34 tests as of this writing)
- [ ] `streamlit run app.py` works locally and every tab/slider responds
      without an error in the terminal
- [ ] `git status` from the repo root — make sure only the intended files
      are tracked (no stray `__pycache__/`, `.pytest_cache/`, or
      `*.egg-info/` directories; `.gitignore` already
      excludes these)
- [ ] If deploying to a **public** GitHub repo (required for the free tier
      of Streamlit Community Cloud), double check there's nothing sensitive
      anywhere in this project — there shouldn't be, since this project
      needs no API keys, credentials, or private data of any kind

## Streamlit Community Cloud specifically

- [ ] The repo is pushed to GitHub
- [ ] Main file path is set to
      `app.py` (the app is at the repo root, so no subfolder prefix is
      needed)
- [ ] First deploy's build log shows `acexec` installing successfully (look
      for a line building the "editable" wheel for `acexec`, confirming the
      `-e .` line in `requirements.txt` was picked up)

## Docker specifically

- [ ] `docker build -t acexec-dashboard .` completes with
      no errors
- [ ] `docker run -p 8501:8501 acexec-dashboard` serves the app, and
      `curl http://localhost:8501/_stcore/health` returns `ok`
- [ ] Whichever host you deploy to (Render/Railway/Fly) is configured to
      set the `$PORT` environment variable — all three do this
      automatically, but double-check the deploy logs show the app actually
      binding to that port rather than falling back to the default `8501`
      inside a container that's routing traffic to a different port

## After deploying

- [ ] Open the live URL and exercise every control at least once (shares,
      intervals, sigma, eta, gamma, price, and the lambda slider) — a
      dashboard that imports cleanly can still have a control that throws
      once a specific combination of inputs is selected
- [ ] Click "Run verification" and confirm it reports a match, not a
      mismatch, on the live deployment (this exercises the `scipy`
      dependency actually being present in the deployed environment, which
      is a common thing to miss in a `requirements.txt` if it's ever
      hand-edited later)
