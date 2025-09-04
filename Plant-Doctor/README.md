
# Plant Doctor — Streamlit

A production-ready Streamlit web app for plant disease prediction using a Keras model.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place your model at `model/plantdisease.keras` and make sure `labels.json` lists the class names in the same order as your model outputs.

## Structure
- `app.py` — Streamlit UI (pages: Detect, Project Details, Author)
- `main.py` — Core predictor logic (UI-agnostic)
- `labels.json` — Class labels in model output order
- `model/plantdisease.keras` — Keras model file
- `.streamlit/config.toml` — Theme
- `assets/` — App assets (logo, icons)

## Notes
- Images are resized to 224×224 and normalized to [0, 1].
- The predictor is cached with `st.cache_resource` for performance.
- Logging follows a simple structured format.
