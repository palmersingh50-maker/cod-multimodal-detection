// Client-side rendering of mask overlays so opacity can be changed without a round trip.
function load(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("image decode failed"));
    im.src = src;
  });
}

export async function composeOverlay(originalUrl: string, maskUrl: string, opacity: number): Promise<string> {
  const [orig, mask] = await Promise.all([load(originalUrl), load(maskUrl)]);
  const c = document.createElement("canvas");
  c.width = orig.naturalWidth;
  c.height = orig.naturalHeight;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(orig, 0, 0);
  // Tint: draw mask into a scratch canvas, recolour foreground, composite with alpha.
  const t = document.createElement("canvas");
  t.width = c.width;
  t.height = c.height;
  const tctx = t.getContext("2d")!;
  tctx.drawImage(mask, 0, 0, c.width, c.height);
  const d = tctx.getImageData(0, 0, t.width, t.height);
  for (let i = 0; i < d.data.length; i += 4) {
    const on = d.data[i] > 127;
    d.data[i] = 255;
    d.data[i + 1] = 64;
    d.data[i + 2] = 64;
    d.data[i + 3] = on ? 255 : 0;
  }
  tctx.putImageData(d, 0, 0);
  ctx.globalAlpha = opacity;
  ctx.drawImage(t, 0, 0);
  return c.toDataURL("image/png");
}
