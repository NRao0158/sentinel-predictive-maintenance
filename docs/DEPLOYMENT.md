# Deployment

## Local

Follow README setup and run `python -m streamlit run app.py`. Default URL: http://localhost:8501. No raw data needed. Models were trained with Python 3.12 and the pinned requirements.

## GitHub

Create a public repository named `sentinel-predictive-maintenance`. Push the contents of this folder as the repository root. Include `artifacts/model.joblib`, `artifacts/metrics.json`, `artifacts/demo_history.csv` and all source/docs. Do not force-add ignored raw data or secrets. Use GitHub CLI if installed:

```powershell
gh auth login
gh repo create sentinel-predictive-maintenance --public --source . --remote origin --push
```

This command assumes a local commit already exists. If using the web upload interface, upload project files preserving their folder structure.

## Streamlit Community Cloud (free)

1. Sign in at https://share.streamlit.io/ and connect the GitHub account.
2. Select Create app and the repository, `main` branch, and `app.py` entrypoint.
3. In advanced settings choose Python 3.12, then deploy.
4. Check the prediction studio, model evidence, batch sample and a simulated alert.
5. Add the resulting URL to GitHub README and resume once the public app is verified.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy . Free platform documentation: https://docs.streamlit.io/deploy . Account authorization must be completed by the account owner. No deployment URL should be claimed before verification.

## Docker alternative

```bash
docker build -t sentinel .
docker run --rm -p 8501:8501 sentinel
```

Docker configuration is provided; a Docker engine is required. Never deserialize model files submitted by visitors. Public hosting stores no persistent alerts; each visitor has an independent session.
