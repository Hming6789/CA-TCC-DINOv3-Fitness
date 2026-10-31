# VHM Motion: Web Application & Render Deployment

VHM Motion is a responsive, English-language video analysis interface. A same-origin FastAPI server serves the interface and a CPU-only ONNX inference API. It accepts one exercise clip, samples 18 RGB frames, extracts DINOv3 CLS embeddings, and applies the trained temporal adapter and classification head.

## Current application state

No exercise-trained classifier checkpoint is included. The application starts in **Preview mode**: upload, playback, removal, and help are available; prediction remains disabled until a valid model package is connected. No sample confidence scores are presented as real predictions. Exercise classification is separate from technique assessment or repetition counting.

## Run locally

From the repository root, using the `catcc-dinov3` Conda environment:

```powershell
conda activate catcc-dinov3
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-web.txt
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The browser plays the selected file locally before analysis. In model mode, only the Analyze action uploads it. Uploaded files are temporary and removed after processing. Limits: 32 MB, 30 seconds, 4K, at least 18 readable frames; one active upload/inference job per server process. MP4 is recommended. MOV/AVI/MKV playback depends on browser codecs; the server performs its own validation.

## Deploy the preview on Render

1. Push this application's files to your GitHub repository, including `webapp/`, `requirements-web.txt`, and `render.yaml`.
2. In Render, choose **New → Blueprint**, connect the repository, select the branch containing these files, and apply the Blueprint.
3. Open the URL provided by Render when deployment is complete. `/healthz` is the health check and `/api/status` shows whether the model is connected.

The Blueprint selects the free plan for the preview and uses these commands:

```text
Build: pip install -r requirements-web.txt && python -m webapp.prepare_model
Start: uvicorn webapp.app:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 5
```

If using **New → Web Service** manually, select the Python runtime and copy those commands. Keep the root directory at the repository root. Set `PYTHON_VERSION=3.11.11`, `MODEL_DIR=model_artifacts`, and `INFERENCE_THREADS=2`.

Render free services spin down after inactivity and use an ephemeral filesystem. A model bundle must therefore be fetched during every build; a manual runtime upload is not persistent. The free preview does not establish that a trained model fits the plan. Measure peak RAM and latency with the exported model, then select an appropriate instance if required. See [Render FastAPI deployment](https://render.com/docs/deploy-fastapi), [Blueprint specification](https://render.com/docs/blueprint-spec), and [free-service limits](https://render.com/docs/free).

## Connect a trained classifier

Train/fine-tune using the existing [DINOv3 guide](DINOV3_FINETUNING.md). Export a checkpoint whose classification head was trained in `supervised`, `ft_1p`, `train_linear`, or `dinov3_finetune` mode. Self-supervised representation checkpoints cannot provide trained exercise logits.

In the training environment, which must contain the cached DINOv3 encoder and its processor:

```powershell
python -m pip install onnx==1.18.0 onnxruntime==1.22.1
python -m webapp.export_model --checkpoint "experiments_logs/path/to/best.pt" --output model_artifacts --version 1.0.0
```

Replace the example path with the checkpoint path printed by training. The exporter requires an empty output directory and validates PyTorch/ONNX numerical agreement. It produces:

```text
model_artifacts/
  encoder.onnx    # normalized RGB frame [1,3,224,224] -> CLS [1,384]
  head.onnx       # embeddings [1,18,384] -> logits [1,3]
  model.json      # classes, preprocessing, checksums, version, provenance
model_artifacts.zip
```

Restart the local application to load `model_artifacts/`. Test on held-out exercise videos before presenting predictions as useful recognition results. The exporter checks numerical equivalence; it does not measure classification accuracy.

For Render, host `model_artifacts.zip` at an HTTPS download URL you control. Respect the upstream model's distribution terms when hosting weights. In the Render service's **Environment** settings, add:

| Variable | Value |
| --- | --- |
| `MODEL_BUNDLE_URL` | Direct HTTPS download URL for the ZIP |
| `MODEL_BUNDLE_SHA256` | SHA-256 printed by the exporter |
| `MODEL_DIR` | `model_artifacts` |
| `INFERENCE_THREADS` | `2` |

Redeploy so the build fetches the checksum-pinned bundle. Check `/api/status` for `ready: true` and confirm the interface says **Model connected**. A missing bundle keeps the preview available. A failed configured download fails the build; an invalid runtime graph keeps prediction disabled and records the error in server logs. Keep credentials in Render environment settings, never in GitHub.

## Verification

```powershell
python -m pip install -r requirements-web-dev.txt
python -m unittest webapp.test_webapp -v
```

The sampler parity check also needs the existing training environment's PyTorch installation. The suite checks preview mode, API inference with disposable toy ONNX graphs, training/serving sampling parity, temporary-file cleanup, invalid uploads, checksums, concurrent requests, bounded chunked uploads, and ZIP path validation. Toy graphs are created only in temporary directories and are never included in the deployment.

Production serving does not import PyTorch or Transformers. It runs the frame encoder sequentially on CPU to reduce peak memory; actual latency depends on the deployment instance. One worker is intentional: additional workers duplicate the model in RAM and change the one-job limit. Cancelling a browser request stops waiting for the response; an already running inference may finish before its temporary file is removed.
