# YOLOv8 Vehicle Detection — Streamlit Deployment

Deploys the model trained in `Yolo_v8_segmentation_on_traffic_vehicles.ipynb`
as a Streamlit web app, hosted via GitHub → Streamlit Community Cloud.

> **Note on the notebook name:** the notebook trains a standard YOLOv8
> **object-detection** model (`yolov8n.pt`, single class `Vehicle`,
> bounding boxes) rather than a segmentation model (no `-seg` weights or
> mask outputs are used). `app.py` below is built for that
> detection model. If you actually want pixel-level masks, retrain using
> `YOLO('yolov8n-seg.pt')` on a dataset with segmentation labels, and the
> app will need a small change to plot masks (`results[0].plot()` already
> supports this automatically once the model itself is a `-seg` model —
> no code change needed, just swap the weights file).

## Folder contents

```
Deployment/
├── app.py                  # Streamlit app (image + video inference)
├── requirements.txt        # Python dependencies
├── packages.txt            # System (apt) dependencies for Streamlit Cloud
├── .gitignore
├── .streamlit/
│   └── config.toml         # App theme / upload size limit
├── weights/
│   ├── best.pt              # <-- YOU ADD THIS (your trained model)
│   └── README.md            # instructions for adding weights
└── README.md                # this file
```

## 1. Get your trained weights

In Colab, after `model.train(...)` finishes, download the weights file:

```python
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')
```

Place the downloaded `best.pt` into `Deployment/weights/best.pt`.

## 2. Test locally (recommended before deploying)

```bash
cd Deployment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## 3. Push to GitHub

```bash
cd Deployment
git init
git add .
git commit -m "Initial commit: YOLOv8 vehicle detection Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If `best.pt` is over 100 MB, see the Git LFS instructions in
`weights/README.md` — a fine-tuned `yolov8n` is normally only a few MB,
so this usually isn't necessary.

## 4. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to:
   ```
   Deployment/app.py
   ```
   (or `app.py` if you pushed the contents of `Deployment/` as the repo root).
4. Click **Deploy**. Streamlit Cloud will automatically install
   `requirements.txt` and `packages.txt`.
5. Wait for the build to finish — first build can take a few minutes
   because of `torch`/`ultralytics`.

Your app will be live at a URL like:
`https://<your-app-name>.streamlit.app`

## 5. Loading weights from a URL instead (optional)

If you'd rather not commit the `.pt` file to GitHub at all, host it
somewhere (GitHub Release asset, Hugging Face Hub, S3, Google Drive)
and download it on first run. Replace the model-loading section of
`app.py` with:

```python
import urllib.request

MODEL_URL = "https://<your-hosted-weights-url>/best.pt"
MODEL_PATH = Path("weights/best.pt")

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cv2` | Make sure `opencv-python-headless` is in `requirements.txt` (not plain `opencv-python`, which needs GUI libs that aren't on Streamlit Cloud). |
| `ImportError: libGL.so.1` | Ensure `packages.txt` (with `libgl1`) is present at the repo root next to `app.py`. |
| App builds but model fails to load | Confirm `weights/best.pt` was actually committed — check GitHub file size isn't 0 KB or missing (common if `.gitignore` accidentally excludes it). |
| Video processing is slow | Streamlit Cloud's free tier is CPU-only; large videos will be slow. Consider limiting to shorter clips or lowering `imgsz`/frame rate for the demo. |
