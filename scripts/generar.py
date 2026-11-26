"""Fabrica de shorts: guion JSON -> video vertical MP4 con voz clonada, avatar y subtitulos.

Uso: python scripts/generar.py guiones/2026-11-26-01.json salidas/
"""
import json, os, re, subprocess, sys, wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
RAIZ = Path(__file__).resolve().parent.parent
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def cargar_marca():
    p = RAIZ / "marca.json"
    base = {"fondo_1": "#0b1026", "fondo_2": "#3b0d4f", "fondo_3": "#0e3a5c", "color_barra": "#dc2626", "color_texto": "#ffffff",
            "color_resaltado": "#facc15", "nombre_canal": "Mi canal de noticias"}
    if p.exists():
        base.update(json.loads(p.read_text(encoding="utf-8")))
    return base


# ---------- 1. Voz ----------
def dividir_frases(texto, max_chars=220):
    frases = re.split(r"(?<=[.!?¿¡;:])\s+", texto.strip())
    trozos, actual = [], ""
    for f in frases:
        if len(actual) + len(f) < max_chars:
            actual = (actual + " " + f).strip()
        else:
            if actual:
                trozos.append(actual)
            actual = f
    if actual:
        trozos.append(actual)
    return trozos


AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".flac", ".webm", ".mp4", ".3gp", ".amr"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def preparar_referencia(tmp):
    """Toma el primer audio de voz/ (cualquier formato del celular) y lo deja en WAV mono 24 kHz, max 30 s."""
    audios = sorted(p for p in (RAIZ / "voz").glob("*") if p.suffix.lower() in AUDIO_EXT)
    if not audios:
        return None
    dst = tmp / "referencia.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audios[0]), "-t", "30",
                    "-ac", "1", "-ar", "24000",
                    "-af", "silenceremove=start_periods=1:start_threshold=-45dB,loudnorm=I=-16",
                    "-c:a", "pcm_s16le", str(dst)], check=True)
    print("Voz de referencia:", audios[0].name)
    return dst


def generar_voz(texto, salida_wav):
    import torch, torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    modelo = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    ref = preparar_referencia(Path(salida_wav).parent)
    extra = {"audio_prompt_path": str(ref)} if ref else {}
    if not ref:
        print("AVISO: no hay audio en voz/, se usa la voz por defecto")
    pausa = torch.zeros(1, int(modelo.sr * 0.25))
    partes = []
    for trozo in dividir_frases(texto):
        wav = modelo.generate(trozo, language_id="es", exaggeration=0.5, cfg_weight=0.5, **extra)
        partes += [wav.cpu(), pausa]
    crudo = Path(salida_wav).with_suffix(".crudo.wav")
    ta.save(str(crudo), torch.cat(partes, dim=1), modelo.sr)
    # normaliza a PCM 16 bits (lo que entiende el modulo wave) y sube un poco el volumen
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(crudo), "-ac", "1",
                    "-af", "loudnorm=I=-16:TP=-1.5", "-ar", "24000", "-c:a", "pcm_s16le",
                    str(salida_wav)], check=True)


# ---------- 2. Tiempos de cada palabra ----------
def tiempos_palabras(wav_path, texto, duracion):
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(str(wav_path), language="es", word_timestamps=True,
                               initial_prompt=texto[:200])
        pal = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words if w.word.strip()]
        if pal:
            return pal
    except Exception as e:  # sin whisper: reparto uniforme
        print("Whisper no disponible, reparto uniforme:", e)
    palabras = texto.split()
    paso = duracion / max(len(palabras), 1)
    return [(p, i * paso, (i + 1) * paso) for i, p in enumerate(palabras)]


# ---------- 3. Imagenes de apoyo ----------
def imagenes_apoyo(claves, carpeta):
    rutas = []
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return rutas
    import requests
    for i, q in enumerate(claves[:4]):
        try:
            r = requests.get("https://api.pexels.com/v1/search",
                             params={"query": q, "orientation": "portrait", "per_page": 1},
                             headers={"Authorization": key}, timeout=20).json()
            url = r["photos"][0]["src"]["large2x"]
            dst = carpeta / f"img{i}.jpg"
            dst.write_bytes(requests.get(url, timeout=30).content)
            rutas.append(dst)
        except Exception as e:
            print("Sin imagen para", q, e)
    return rutas


def ajustar(img, w, h):
    img = img.convert("RGB")
    r = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * r) + 1, int(img.height * r) + 1))
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


# ---------- 4. Avatar ----------
def avatar_por_defecto(boca_abierta):
    a = Image.new("RGBA", (520, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(a)
    d.ellipse((20, 20, 500, 500), fill="#f2c9a0", outline="#1f2937", width=10)
    d.ellipse((150, 180, 200, 240), fill="#1f2937")
    d.ellipse((320, 180, 370, 240), fill="#1f2937")
    if boca_abierta:
        d.ellipse((200, 320, 320, 420), fill="#7f1d1d", outline="#1f2937", width=8)
    else:
        d.arc((190, 300, 330, 390), 20, 160, fill="#1f2937", width=10)
    return a


def recortar_circulo(img):
    """Si la imagen no tiene fondo transparente, la recorta en circulo con borde."""
    lado = min(img.size)
    img = img.crop(((img.width - lado) // 2, (img.height - lado) // 2,
                    (img.width + lado) // 2, (img.height + lado) // 2)).resize((520, 520))
    masc = Image.new("L", (520, 520), 0)
    ImageDraw.Draw(masc).ellipse((8, 8, 512, 512), fill=255)
    out = Image.new("RGBA", (520, 520), (0, 0, 0, 0))
    ImageDraw.Draw(out).ellipse((0, 0, 519, 519), fill="#ffffff")
    out.paste(img.convert("RGB"), (0, 0), masc)
    return out


def abrir_avatar(p):
    img = Image.open(p)
    tiene_alfa = img.mode in ("RGBA", "LA") and img.getextrema()[-1][0] < 250
    img = img.convert("RGBA") if tiene_alfa else recortar_circulo(img)
    img.thumbnail((620, 620))
    return img


def cargar_avatar():
    """assets/: un archivo con 'cerrada' y otro con 'abierta' en el nombre; si no, las 2 primeras imagenes
    en orden alfabetico (1a = boca cerrada, 2a = boca abierta). Con una sola imagen, se usa para ambas."""
    imgs = sorted(p for p in (RAIZ / "assets").glob("*") if p.suffix.lower() in IMG_EXT)
    cerr = next((p for p in imgs if "cerrad" in p.name.lower()), imgs[0] if imgs else None)
    abie = next((p for p in imgs if "abiert" in p.name.lower()),
                next((p for p in imgs if p != cerr), cerr))
    if not cerr:
        return {"cerrada": avatar_por_defecto(False), "abierta": avatar_por_defecto(True)}
    print("Avatar:", cerr.name, "/", abie.name)
    return {"cerrada": abrir_avatar(cerr), "abierta": abrir_avatar(abie)}


# ---------- 5. Texto ----------
def envolver(draw, texto, fuente, ancho):
    lineas, linea = [], ""
    for p in texto.split():
        prueba = (linea + " " + p).strip()
        if draw.textlength(prueba, font=fuente) <= ancho:
            linea = prueba
        else:
            lineas.append(linea)
            linea = p
    if linea:
        lineas.append(linea)
    return lineas


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def fondo_animado(marca):
    """Degradado de 3 colores (alto 2H) con brillo radial y lineas diagonales sutiles."""
    c1, c2, c3 = (hex_rgb(marca[k]) for k in ("fondo_1", "fondo_2", "fondo_3"))
    y = np.linspace(0, 1, 2 * H)[:, None, None]
    tri = np.abs(((y * 2) % 2) - 1)  # 0..1..0 para que el ciclo no tenga cortes
    grad = np.where(tri < 0.5, c1 + (c2 - c1) * (tri / 0.5), c2 + (c3 - c2) * ((tri - 0.5) / 0.5))
    img = np.broadcast_to(grad, (2 * H, W, 3)).copy()
    yy, xx = np.mgrid[0:2 * H, 0:W]
    lineas = ((xx + yy) // 18) % 9 == 0
    img[lineas] *= 1.12
    cx = (xx - W / 2) / W
    vin = 1 - 0.45 * np.clip(cx ** 2 * 2.2, 0, 1)
    img *= vin[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")


def armar_texto(guion):
    """El guion trae gancho, desarrollo y cierre; se unen en un solo texto para la voz."""
    if "texto" not in guion:
        guion["texto"] = " ".join(guion[k].strip() for k in ("gancho", "desarrollo", "cierre"))
    n = len(guion["texto"].split())
    if not 70 <= n <= 115:
        print(f"AVISO: el guion tiene {n} palabras; para 30-45 s lo ideal es 75-110")
    return guion


def leer_volumen(wav_path):
    with wave.open(str(wav_path)) as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        datos = np.frombuffer(w.readframes(n), dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
        if w.getnchannels() > 1:
            datos = datos.reshape(-1, w.getnchannels()).mean(axis=1)
    por_frame = sr // FPS
    total = len(datos) // por_frame
    rms = np.array([np.sqrt(np.mean(datos[i * por_frame:(i + 1) * por_frame] ** 2)) for i in range(total)])
    return rms / (rms.max() or 1), len(datos) / sr


def render(guion, wav_path, salida_mp4, tmp):
    marca = cargar_marca()
    vol, dur = leer_volumen(wav_path)
    palabras = tiempos_palabras(wav_path, guion["texto"], dur)
    if guion.get("gancho"):
        n_g = len(guion["gancho"].split())
        guion["_fin_gancho"] = palabras[min(n_g, len(palabras)) - 1][2]
    if dur > 46:
        print(f"AVISO: el video dura {dur:.0f} s (meta 30-45 s)")
    fotos = [ajustar(Image.open(p), W, H) for p in imagenes_apoyo(guion.get("palabras_clave", []), tmp)]
    avatar = cargar_avatar()
    f_tit = ImageFont.truetype(FUENTE, 62)
    f_sub = ImageFont.truetype(FUENTE, 84)
    f_can = ImageFont.truetype(FUENTE, 40)

    fondo_alto = fondo_animado(marca)
    oscuro = Image.new("RGBA", (W, H), (0, 0, 0, 120))
    total = int(dur * FPS)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav_path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(salida_mp4)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # titular (fijo)
    capa_tit = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa_tit)
    lineas = envolver(d, guion["titular"].upper(), f_tit, W - 120)[:3]
    alto = 60 + 78 * len(lineas)
    d.rectangle((0, 110, W, 110 + alto), fill=marca["color_barra"])
    for i, l in enumerate(lineas):
        d.text((60, 140 + 78 * i), l, font=f_tit, fill=marca["color_texto"])
    d.text((60, H - 90), marca["nombre_canal"], font=f_can, fill=marca["color_texto"])

    grupos = [palabras[i:i + 3] for i in range(0, len(palabras), 3)]
    for fr in range(total):
        t = fr / FPS
        if fotos:
            idx = min(int(t / (dur / len(fotos))), len(fotos) - 1)
            img = fotos[idx].copy()
            # zoom lento tipo Ken Burns
            z = 1 + 0.06 * ((t % (dur / len(fotos))) / (dur / len(fotos)))
            cw, ch = int(W / z), int(H / z)
            img = img.crop(((W - cw) // 2, (H - ch) // 2, (W + cw) // 2, (H + ch) // 2)).resize((W, H))
            marco = Image.alpha_composite(img.convert("RGBA"), oscuro)
        else:
            y0 = int((H * (1 + np.sin(2 * np.pi * t / 20))) / 2)  # vaiven lento
            marco = fondo_alto.crop((0, y0, W, y0 + H))
        if t < guion.get("_fin_gancho", 0) and int(t * 2) % 2 == 0:
            d = ImageDraw.Draw(marco)
            d.ellipse((60, 60, 92, 92), fill="#ef4444")
            d.text((106, 52), "AHORA", font=f_can, fill="#ffffff")
        marco = Image.alpha_composite(marco, capa_tit)

        # avatar que habla
        v = vol[min(fr, len(vol) - 1)] if len(vol) else 0
        av = avatar["abierta" if v > 0.18 else "cerrada"]
        salto = int(v * 14)
        marco.alpha_composite(av, ((W - av.width) // 2, H - av.height - 560 - salto))

        # subtitulos (grupo de 3 palabras, resaltando la actual)
        grupo = next((g for g in grupos if g[0][1] <= t <= g[-1][2] + 0.15), None)
        if grupo:
            d = ImageDraw.Draw(marco)
            texto = " ".join(p[0] for p in grupo).upper()
            fs = f_sub
            while d.textlength(texto + " ", font=fs) > W - 100 and fs.size > 40:
                fs = ImageFont.truetype(FUENTE, fs.size - 6)
            x = (W - d.textlength(texto, font=fs)) / 2
            y = H - 470
            for p, ini, fin in grupo:
                pal = p.upper() + " "
                color = marca["color_resaltado"] if ini <= t <= fin + 0.05 else marca["color_texto"]
                d.text((x, y), pal, font=fs, fill=color, stroke_width=6, stroke_fill="#000000")
                x += d.textlength(pal, font=fs)

        ff.stdin.write(marco.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    guion_path, carpeta_salida = Path(sys.argv[1]), Path(sys.argv[2])
    guion = armar_texto(json.loads(guion_path.read_text(encoding="utf-8")))
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    tmp = carpeta_salida / "tmp"
    tmp.mkdir(exist_ok=True)
    wav = tmp / "voz.wav"
    if os.environ.get("AUDIO_DE_PRUEBA"):
        wav = Path(os.environ["AUDIO_DE_PRUEBA"])
    else:
        generar_voz(guion["texto"], wav)
    mp4 = carpeta_salida / f"{guion['id']}.mp4"
    render(guion, wav, mp4, tmp)
    print("LISTO", mp4)


if __name__ == "__main__":
    main()
