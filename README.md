---
title: CrackVision
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_file: app.py
---

# CrackVision

CrackVision is a road-damage detection project built around a fine-tuned YOLOv8-L model and crack-aware inference. The training run you shared is already complete and produced `best.pt` with `mAP@0.5 = 0.5082` after 20 epochs.

## What is included

- A deployable FastAPI inference service in `app.py` for Hugging Face Spaces or any Docker host.
- A shared inference layer in `crackvision_service.py` with Fast mode and Accurate mode.
- A Vercel-ready frontend at the repository root.
- A custom Google Stitch prompt in `STITCH_PROMPT.md` for generating a non-generic UI.

## Recommended deployment shape

1. Deploy the backend to Hugging Face Spaces using the Dockerfile in the repo root.
2. Deploy the frontend to Vercel from the repository root.
3. Set `CRACKVISION_API_URL` in Vercel to the public Hugging Face Space URL.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860` for the API or use the frontend below.

## Frontend run

```bash
npm install
npm run dev
```

Set `CRACKVISION_API_URL` in the root `.env.local` before starting the frontend.

## Model file

The service expects `RoadDamage/yolov8l_road_damage/weights/best.pt` by default.

If you publish the repo to GitHub or Hugging Face, keep the checkpoint in Git LFS or upload it as a separate artifact. If you prefer a different location, set `MODEL_PATH`.

## What is still left for a real submission

- Upload `best.pt` to the deployment target if it is not already bundled.
- Set the public backend URL in Vercel.
- If you want a polished presentation, drop the Google Stitch UI into the root frontend and connect it to the API route.

## Notes

- `accurate` mode uses standard inference plus SAHI slicing and WBF.
- `fast` mode uses standard inference with horizontal-flip TTA only.
- The default confidence threshold is loaded from `RoadDamage/opt_conf.json` when present.