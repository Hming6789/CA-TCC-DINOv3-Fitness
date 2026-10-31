"""Serve the VHM interface and bounded, temporary video inference requests."""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .inference import InvalidVideo, ModelUnavailable, MotionClassifier

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 32 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


class BoundedVideoUpload:
    """Bound the entire multipart body before FastAPI parses it; one job at a time."""
    def __init__(self, app):
        self.app = app
        self.slot = asyncio.Semaphore(1)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/analyze" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        if self.slot.locked():
            return await JSONResponse({"detail": "An analysis is already running. Try again shortly."}, 429, headers={"Retry-After": "5"})(scope, receive, send)
        async with self.slot:
            # Multipart headers add a small overhead to the permitted video size.
            limit = MAX_UPLOAD + 256 * 1024
            headers = dict(scope.get("headers", []))
            try:
                length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                return await JSONResponse({"detail": "Invalid request length."}, 400)(scope, receive, send)
            if length > limit:
                return await JSONResponse({"detail": "The video must be 32 MB or smaller."}, 413)(scope, receive, send)
            with tempfile.SpooledTemporaryFile(max_size=1024*1024) as body:
                size = 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > limit:
                        return await JSONResponse({"detail": "The video must be 32 MB or smaller."}, 413)(scope, receive, send)
                    body.write(chunk)
                    if not message.get("more_body", False):
                        break
                body.seek(0)
                async def replay():
                    chunk = body.read(1024*1024)
                    return {"type": "http.request", "body": chunk, "more_body": body.tell() < size}
                await self.app(scope, replay, send)


def create_app(classifier=None):
    engine = classifier or MotionClassifier(os.getenv("MODEL_DIR", "model_artifacts"), int(os.getenv("INFERENCE_THREADS", "2")))

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(engine.load)
        yield

    app = FastAPI(title="VHM Fitness Motion", version="1.0.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(BoundedVideoUpload)
    app.state.classifier = engine
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/status")
    def status():
        return {**engine.status(), "max_upload_mb": 32, "max_duration_seconds": 30}

    @app.get("/readyz")
    def readiness():
        return JSONResponse(engine.status(), status_code=200 if engine.ready else 503)

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)):
        temporary = None
        try:
            if not engine.ready:
                raise HTTPException(503, engine.status()["message"] or "Connect a trained model first.")
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(415, "Choose an MP4, WebM, MOV, AVI or MKV video.")
            total = 0
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temporary = Path(handle.name)
                while chunk := await file.read(1024*1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "The video must be 32 MB or smaller.")
                    handle.write(chunk)
            if not total:
                raise HTTPException(400, "The uploaded video is empty.")
            return await asyncio.to_thread(engine.predict, temporary)
        except (InvalidVideo, ModelUnavailable) as exc:
            raise HTTPException(422 if isinstance(exc, InvalidVideo) else 503, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Video analysis failed")
            raise HTTPException(500, "Analysis could not finish. Try a shorter MP4 clip.") from exc
        finally:
            await file.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return app


app = create_app()
