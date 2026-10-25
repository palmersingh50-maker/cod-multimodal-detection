import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, dataUrl, getStatus, predict, type Mode, type ModeResult, type PredictResponse, type Status } from "./api";
import { composeOverlay } from "./Compositor";

type Phase = "idle" | "loading" | "success" | "error";
const MODE_LABEL: Record<Mode, string> = { rgb: "RGB only", multimodal: "RGB + estimated depth", both: "Compare both" };

function download(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

function Panel({ title, src, caption }: { title: string; src?: string; caption?: string }) {
  return (
    <figure className="panel">
      <figcaption>{title}</figcaption>
      <div className="frame">{src ? <img src={src} alt={title} /> : <span className="muted">…</span>}</div>
      {caption && <p className="caption">{caption}</p>}
    </figure>
  );
}

function ResultBlock({ label, res, original, opacity, stem }: { label: string; res: ModeResult; original: string; opacity: number; stem: string }) {
  const [overlay, setOverlay] = useState<string>();
  const maskUrl = useMemo(() => dataUrl(res.mask_png), [res]);
  useEffect(() => {
    let live = true;
    composeOverlay(original, maskUrl, opacity).then((u) => live && setOverlay(u)).catch(() => live && setOverlay(dataUrl(res.overlay_png)));
    return () => {
      live = false;
    };
  }, [original, maskUrl, opacity, res]);
  const t = res.timings_ms;
  return (
    <section className="result" aria-label={label}>
      <header>
        <h2>{label}</h2>
        <span className="chips">
          <span className="chip">{t.total_ms.toFixed(0)} ms</span>
          {t.depth_ms !== undefined && <span className="chip">depth {t.depth_ms.toFixed(0)} ms</span>}
          <span className="chip">{(res.foreground_fraction * 100).toFixed(1)}% foreground</span>
        </span>
      </header>
      <div className="grid">
        <Panel title="Original" src={original} />
        <Panel title="Overlay" src={overlay} />
        <Panel title="Mask" src={maskUrl} />
      </div>
      <div className="actions">
        <button onClick={() => download(maskUrl, `${stem}_${label.replace(/\W+/g, "_")}_mask.png`)}>Download mask</button>
        <button onClick={() => overlay && download(overlay, `${stem}_${label.replace(/\W+/g, "_")}_overlay.png`)} disabled={!overlay}>
          Download overlay
        </button>
      </div>
    </section>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [statusErr, setStatusErr] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>();
  const [mode, setMode] = useState<Mode>("rgb");
  const [threshold, setThreshold] = useState(0.5);
  const [opacity, setOpacity] = useState(0.55);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string>();
  const [data, setData] = useState<PredictResponse>();
  const [drag, setDrag] = useState(false);
  const abort = useRef<AbortController>();
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const ac = new AbortController();
    getStatus(ac.signal)
      .then((s) => {
        setStatus(s);
        setMode(s.modes.includes("both") ? "both" : s.modes.includes("multimodal") ? "multimodal" : "rgb");
      })
      .catch((e) => e.name !== "AbortError" && setStatusErr(true));
    return () => ac.abort();
  }, []);

  useEffect(() => {
    if (!file) return setPreview(undefined);
    const u = URL.createObjectURL(file);
    setPreview(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const choose = useCallback((f?: File | null) => {
    if (!f) return;
    setFile(f);
    setData(undefined);
    setError(undefined);
    setPhase("idle");
  }, []);

  const run = async () => {
    if (!file) return;
    abort.current?.abort();
    const ac = (abort.current = new AbortController());
    setPhase("loading");
    setError(undefined);
    try {
      setData(await predict(file, mode, threshold, ac.signal));
      setPhase("success");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError(e instanceof ApiError ? e.message : "Unexpected error.");
      setPhase("error");
    }
  };

  const stem = file?.name.replace(/\.[^.]+$/, "") ?? "image";
  const ready = status?.ready;

  return (
    <main>
      <header className="top">
        <h1>Camouflage Detector</h1>
        <p className="muted">Pixel-level segmentation of objects that blend into their surroundings.</p>
        <p className={"status " + (ready ? "ok" : "bad")} role="status">
          {statusErr ? "Backend unreachable" : !status ? "Connecting…" : ready ? `Model ready · ${status.device}` : "No model loaded on server"}
        </p>
      </header>

      <section
        className={"drop" + (drag ? " over" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          choose(e.dataTransfer.files?.[0]);
        }}
      >
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" hidden onChange={(e) => choose(e.target.files?.[0])} />
        {preview ? <img className="preview" src={preview} alt="Selected input" /> : <p>Drag &amp; drop an image here</p>}
        <button className="secondary" onClick={() => input.current?.click()}>
          {file ? "Choose another image" : "Select image"}
        </button>
        {file && <span className="muted">{file.name}</span>}
      </section>

      <section className="controls">
        <label>
          Model
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} disabled={!status}>
            {(status?.modes ?? ["rgb"]).map((m) => (
              <option key={m} value={m}>
                {MODE_LABEL[m]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Threshold {threshold.toFixed(2)}
          <input type="range" min={0.05} max={0.95} step={0.05} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
        </label>
        <label>
          Overlay opacity {Math.round(opacity * 100)}%
          <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
        <button className="primary" onClick={run} disabled={!file || !ready || phase === "loading"}>
          {phase === "loading" ? "Detecting…" : "Detect"}
        </button>
      </section>

      {phase === "loading" && <div className="banner info" role="status"><span className="spinner" /> Running inference…</div>}
      {phase === "error" && <div className="banner error" role="alert">{error}</div>}

      {phase === "success" && data && preview && (
        <>
          <p className="muted summary">
            {data.width}×{data.height}px · server time {data.total_ms.toFixed(0)} ms · threshold {data.threshold.toFixed(2)}
          </p>
          {data.results.rgb && <ResultBlock label="RGB only" res={data.results.rgb} original={preview} opacity={opacity} stem={stem} />}
          {data.results.multimodal && (
            <ResultBlock label="RGB + estimated depth" res={data.results.multimodal} original={preview} opacity={opacity} stem={stem} />
          )}
        </>
      )}
    </main>
  );
}
