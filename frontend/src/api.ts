export type Mode = "rgb" | "multimodal" | "both";

export interface Timings {
  preprocess_ms: number;
  depth_ms?: number;
  model_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

export interface ModeResult {
  mask_png: string; // base64
  overlay_png: string; // base64
  probability_png: string; // base64, 8-bit foreground probability
  foreground_fraction: number;
  used_depth: boolean;
  timings_ms: Timings;
}

export interface PredictResponse {
  width: number;
  height: number;
  threshold: number;
  total_ms: number;
  results: Partial<Record<"rgb" | "multimodal", ModeResult>>;
}

export interface Status {
  ready: boolean;
  device: string | null;
  modes: Mode[];
  max_upload_mb: number;
}

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function getStatus(signal?: AbortSignal): Promise<Status> {
  const r = await fetch("/api/v1/status", { signal });
  if (!r.ok) throw new ApiError("status_failed", "Backend is not reachable.");
  return r.json();
}

export async function predict(file: File, mode: Mode, threshold: number, signal?: AbortSignal): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  form.append("threshold", String(threshold));
  let r: Response;
  try {
    r = await fetch("/api/v1/predict", { method: "POST", body: form, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("network", "Could not reach the server.");
  }
  if (!r.ok) {
    let code = "http_" + r.status;
    let message = `Request failed (${r.status}).`;
    try {
      const j = await r.json();
      code = j.error?.code ?? code;
      message = j.error?.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(code, message);
  }
  return r.json();
}

export const dataUrl = (b64: string) => `data:image/png;base64,${b64}`;
