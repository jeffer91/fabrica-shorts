"""Fabrica de shorts 11 Records: guion JSON -> short vertical de futbol con voz clonada,
imagenes de la noticia cada 5 s, avatar pequeno, subtitulos y musica sin copyright.

Uso: python scripts/generar.py guiones/2026-11-26-01.json salidas/
"""
import json, math, os, random, re, subprocess, sys, time, wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
SEG_IMAGEN = 5.0                       # una imagen nueva cada 5 s
DUR_MIN, DUR_MAX = 30.0, 45.0
RAIZ = Path(__file__).resolve().parent.parent
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FUENTE_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
UA = {"User-Agent": "FabricaShorts11Records/1.0 (github.com/jeffer91/fabrica-shorts)"}

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".flac", ".webm", ".mp4", ".3gp", ".amr"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def cargar_marca():
    base = {"nombre_canal": "11 Records", "lema": "Fútbol mundial",
            "color_primario": "#065f46", "color_oscuro": "#03140f",
            "color_resaltado": "#a3e635", "color_texto": "#ffffff"}
    p = RAIZ / "marca.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        base["nombre_canal"] = d.get("nombre_canal", base["nombre_canal"])
        for k in ("lema",):   # colores fijos del estilo futbolero
            if k in d:
                base[k] = d[k]
    return base


def F(tam, negrita=True):
    return ImageFont.truetype(FUENTE if negrita else FUENTE_REG, tam)


# ---------- 1. Voz ----------
def dividir_frases(texto):
    """Una frase por llamada al modelo: la entonacion sale mas natural que con bloques largos."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?;:])\s+", texto.strip()) if f.strip()]
    out = []
    for f in frases:                       # frases muy largas se parten en la coma mas central
        while len(f) > 200 and "," in f:
            comas = [m.start() for m in re.finditer(",", f)]
            c = min(comas, key=lambda i: abs(i - len(f) / 2))
            out.append(f[:c + 1].strip()); f = f[c + 1:].strip()
        out.append(f)
    return out


def numeros_a_palabras(texto):
    """'2009' -> 'dos mil nueve', '3-0' -> 'tres a cero': el modelo lee mejor palabras que cifras."""
    try:
        from num2words import num2words
    except Exception:
        return texto
    texto = re.sub(r"\b(\d+)\s*-\s*(\d+)\b", lambda m: f"{m.group(1)} a {m.group(2)}", texto)
    texto = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + " por ciento", texto)
    texto = re.sub(r"\b(\d+)º", lambda m: num2words(int(m.group(1)), lang="es", to="ordinal"), texto)
    return re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="es"), texto)


def preparar_referencia(tmp):
    """Limpia tu grabacion: quita ruido y silencios y deja solo voz. El modelo usa ~10 s de referencia."""
    audios = sorted(p for p in (RAIZ / "voz").glob("*") if p.suffix.lower() in AUDIO_EXT)
    if not audios:
        return None
    dst = tmp / "referencia.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+discardcorrupt", "-i", str(audios[0]),
                    "-ac", "1", "-ar", "24000",
                    "-af", "highpass=f=70,afftdn=nf=-25,"
                           "silenceremove=start_periods=1:start_threshold=-40dB:"
                           "stop_periods=-1:stop_duration=0.6:stop_threshold=-40dB,"
                           "loudnorm=I=-16:TP=-1.5",
                    "-t", "20", "-c:a", "pcm_s16le", str(dst)], check=True)
    print("Voz de referencia:", audios[0].name, f"({duracion_wav(dst):.1f} s limpios)")
    return dst


def duracion_wav(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def cargar_local_flexible(carpeta, t3_archivo):
    """Como ChatterboxMultilingualTTS.from_local, pero tolera buffers que el modelo calcula solo
    (p. ej. tokenizer._mel_filters) y que no vienen en el archivo s3gen_v3."""
    import torch
    import chatterbox.mtl_tts as mt
    from safetensors.torch import load_file
    cpu = torch.device("cpu")
    ve = mt.VoiceEncoder()
    ve.load_state_dict(torch.load(carpeta / "ve.pt", map_location=cpu, weights_only=True))
    ve.eval()
    t3 = mt.T3(mt.T3Config.multilingual())
    st = load_file(carpeta / t3_archivo)
    if "model" in st.keys():
        st = st["model"][0]
    t3.load_state_dict(st)
    t3.eval()
    s3 = mt.S3Gen()
    faltan, sobran = s3.load_state_dict(torch.load(carpeta / "s3gen.pt", map_location=cpu, weights_only=True), strict=False)
    graves = [k for k in faltan if not k.startswith("tokenizer.")]
    if graves or sobran:
        raise RuntimeError(f"s3gen incompatible: faltan {graves[:3]}, sobran {list(sobran)[:3]}")
    s3.eval()
    tok = mt.MTLTokenizer(str(carpeta / "grapheme_mtl_merged_expanded_v1.json"))
    conds = mt.Conditionals.load(carpeta / "conds.pt", map_location=cpu) if (carpeta / "conds.pt").exists() else None
    return mt.ChatterboxMultilingualTTS(t3, s3, ve, tok, "cpu", conds=conds)


def cargar_modelo_voz():
    """Intenta, en orden: modelo afinado para espanol latinoamericano (v3) -> multilingue v3 -> v2."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import hf_hub_download, snapshot_download
    try:
        base = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=[
            "ve.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]))
        carpeta = Path("modelo_latam"); carpeta.mkdir(exist_ok=True)
        for f in base.iterdir():
            (carpeta / f.name).unlink(missing_ok=True); (carpeta / f.name).symlink_to(f.resolve())
        t3 = hf_hub_download("ResembleAI/Chatterbox-Multilingual-es-mx-latam", "t3_es_mx_latam.safetensors")
        s3 = hf_hub_download("ResembleAI/Chatterbox-Multilingual-es-mx-latam", "s3gen_v3.pt")
        for nombre, origen in (("t3_es_mx_latam.safetensors", t3), ("s3gen.pt", s3)):
            (carpeta / nombre).unlink(missing_ok=True); (carpeta / nombre).symlink_to(Path(origen).resolve())
        m = cargar_local_flexible(carpeta, "t3_es_mx_latam.safetensors")
        print("Modelo de voz: Chatterbox V3 espanol latinoamericano")
        return m
    except Exception as e:
        print("No se pudo cargar el modelo latam:", repr(e)[:200])
    for version in ("v3", None):
        try:
            m = ChatterboxMultilingualTTS.from_pretrained(device="cpu", **({"t3_model": version} if version else {}))
            print("Modelo de voz: multilingue", version or "v2")
            return m
        except TypeError:
            continue
        except Exception as e:
            print("Fallo modelo", version, repr(e)[:200])
    raise SystemExit("No se pudo cargar ningun modelo de voz")


def recortar_silencio(wav, sr, umbral=0.012, margen=0.04):
    """Quita el silencio que el modelo deja al inicio y al final de cada frase."""
    import torch
    x = wav.squeeze(0)
    env = torch.nn.functional.avg_pool1d(x.abs()[None, None], int(sr * 0.02), stride=int(sr * 0.01))[0, 0]
    activos = (env > umbral * x.abs().max()).nonzero()
    if len(activos) == 0:
        return wav
    ini = max(0, int(activos[0].item() * sr * 0.01 - margen * sr))
    fin = min(x.numel(), int(activos[-1].item() * sr * 0.01 + 0.02 * sr + margen * sr))
    return wav[:, ini:fin]


def pausa_despues(frase):
    f = frase.rstrip()
    if f.endswith(("?", "!")):
        return 0.38
    if f.endswith((".", ":")):
        return 0.32
    if f.endswith(";"):
        return 0.26
    return 0.18          # corte en una coma


def generar_voz(texto, salida_wav, tomas=2):
    """Genera frase por frase con parametros mas expresivos. De cada frase hace 2 tomas y se queda con
    la que Whisper entiende mejor (evita palabras comidas, repeticiones o sonidos raros)."""
    import difflib, torch, torchaudio as ta
    modelo = cargar_modelo_voz()
    ref = preparar_referencia(Path(salida_wav).parent)
    if ref:
        modelo.prepare_conditionals(str(ref), exaggeration=0.6)
    else:
        print("AVISO: no hay audio en voz/, se usa la voz por defecto")
    juez = None
    try:
        from faster_whisper import WhisperModel
        juez = WhisperModel("small", device="cpu", compute_type="int8")
    except Exception as e:
        print("Sin juez de tomas:", e)
    tmp = Path(salida_wav).parent
    partes = []
    for i, frase in enumerate(dividir_frases(texto)):
        hablada = numeros_a_palabras(frase)
        mejor, mejor_p = None, -1
        for k in range((tomas + 1) if juez else 1):
            if k == tomas and mejor_p >= 0.8:      # 3a toma solo si las 2 primeras salieron mal
                break
            torch.manual_seed(1000 * i + k)
            wav = modelo.generate(hablada, language_id="es", exaggeration=0.6, cfg_weight=0.4,
                                  temperature=0.8).cpu()
            wav = recortar_silencio(wav, modelo.sr)
            if not juez:
                mejor = wav
                break
            f = tmp / f"toma_{i}_{k}.wav"
            ta.save(str(f), wav, modelo.sr)
            segs, _ = juez.transcribe(str(f), language="es")
            oido = numeros_a_palabras(" ".join(s.text for s in segs))
            p = difflib.SequenceMatcher(None, normalizar_frase(hablada).split(), normalizar_frase(oido).split()).ratio()
            if p > mejor_p:
                mejor, mejor_p = wav, p
            if p > 0.95:
                break
        print(f"Frase {i + 1}: parecido {mejor_p:.2f}")
        partes += [mejor, torch.zeros(1, int(modelo.sr * pausa_despues(frase)))]
    crudo = Path(salida_wav).with_suffix(".crudo.wav")
    ta.save(str(crudo), torch.cat(partes, dim=1), modelo.sr)
    normalizar_y_ajustar(crudo, salida_wav)


def normalizar_y_ajustar(entrada, salida):
    """PCM 16 bits, volumen estandar y, si se pasa de 45 s, acelera un poco (max 15%) para encajar."""
    tmp = Path(salida).with_suffix(".norm.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(entrada), "-ac", "1",
                    "-af", "highpass=f=70,equalizer=f=180:t=q:w=1:g=2,equalizer=f=6500:t=q:w=1.5:g=-2,loudnorm=I=-16:TP=-1.5",
                    "-ar", "24000", "-c:a", "pcm_s16le", str(tmp)], check=True)
    d = duracion_wav(tmp)
    factor = 1.0
    if d > DUR_MAX - 0.5:
        factor = min(1.15, d / (DUR_MAX - 0.8))
        print(f"Voz de {d:.1f} s: se acelera x{factor:.2f}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-af", f"atempo={factor}",
                    "-c:a", "pcm_s16le", str(salida)], check=True)
    d2 = duracion_wav(salida)
    if d2 < DUR_MIN:
        print(f"AVISO: la voz dura {d2:.1f} s (menos de 30 s). El guion necesita mas palabras.")
    if d2 > DUR_MAX:
        print(f"AVISO: la voz dura {d2:.1f} s (mas de 45 s). El guion necesita menos palabras.")


# ---------- 2. Tiempos de cada palabra ----------
def normalizar(p):
    import unicodedata
    p = unicodedata.normalize("NFKD", p.lower())
    return "".join(c for c in p if c.isalnum())


def tiempos_palabras(wav_path, texto, duracion):
    """Los subtitulos salen SIEMPRE del guion (palabras exactas). Whisper solo se usa para saber
    en que momento se dice cada palabra; las que no reconoce se reparten entre sus vecinas."""
    guion = texto.split()
    tiempos = [None] * len(guion)
    try:
        import difflib
        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(str(wav_path), language="es", word_timestamps=True, initial_prompt=texto[:200])
        oidas = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words if w.word.strip()]
        sm = difflib.SequenceMatcher(None, [normalizar(p) for p in guion], [normalizar(w[0]) for w in oidas], autojunk=False)
        for bloque in sm.get_matching_blocks():
            for k in range(bloque.size):
                _, ini, fin = oidas[bloque.b + k]
                tiempos[bloque.a + k] = (ini, fin)
        print(f"Subtitulos: {sum(t is not None for t in tiempos)}/{len(guion)} palabras sincronizadas con la voz")
    except Exception as e:
        print("Whisper no disponible, reparto uniforme:", e)
    # rellenar huecos repartiendo el tiempo segun el largo de cada palabra
    i = 0
    while i < len(guion):
        if tiempos[i] is not None:
            i += 1
            continue
        j = i
        while j < len(guion) and tiempos[j] is None:
            j += 1
        t0 = tiempos[i - 1][1] if i > 0 else 0.0
        t1 = tiempos[j][0] if j < len(guion) else duracion
        pesos = [len(guion[k]) + 2 for k in range(i, j)]
        total, acc = sum(pesos), t0
        for k, pz in zip(range(i, j), pesos):
            dur_k = (t1 - t0) * pz / total
            tiempos[k] = (acc, acc + dur_k)
            acc += dur_k
        i = j
    return [(p, ini, fin) for p, (ini, fin) in zip(guion, tiempos)]


# ---------- 3. Imagenes de la noticia (Wikimedia Commons, licencias libres, sin llaves) ----------
def limpiar_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


PROHIBIDAS = ("logo", "card", "poster", "stamp", "collage", "signature", "autograph", "map ", "flag",
              "coat of arms", "crest", "banner", "drawing", "cartoon", "screenshot", "infographic", "chart")


def api_json(url, params):
    import requests
    for intento in range(4):
        try:
            resp = requests.get(url, params=params, headers=UA, timeout=25)
            if resp.ok and resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
        except Exception:
            pass
        time.sleep(2 + 3 * intento)
    return {}


def credito(meta):
    autor = limpiar_html(meta.get("Artist", {}).get("value", ""))[:40] or "Wikimedia Commons"
    lic = meta.get("LicenseShortName", {}).get("value", "")
    return f"Foto: {autor} · {lic} · Wikimedia Commons".replace(" ·  ·", " ·")


def buscar_commons(termino, debe, usadas):
    """Busca fotos en Commons. Solo acepta archivos cuyo NOMBRE contiene todas las palabras de 'debe'
    (p. ej. 'Messi'), para no mostrar a otra persona."""
    r = api_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{termino} filetype:bitmap", "gsrlimit": 25, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1400})
    paginas = sorted(r.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    palabras = [w for w in normalizar_frase(termino).split() if len(w) > 2]
    candidatas = []
    for p in paginas:
        titulo = p.get("title", "")
        tn = normalizar_frase(titulo)
        if debe and not all(normalizar_frase(d) in tn for d in debe):
            continue
        if any(x in titulo.lower() for x in PROHIBIDAS):
            continue
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") != "image/jpeg" or ii.get("width", 0) < 700:
            continue
        url = ii.get("thumburl") or ii.get("url")
        if not url or url in usadas:
            continue
        # puntaje: cuantas palabras de la busqueda (club, año, torneo) aparecen en el nombre del archivo;
        # se castigan fotos con varias personas ("&", "vs", "and")
        puntos = sum(w in tn for w in palabras) * 2 - p.get("index", 0) * 0.05
        if re.search(r"&| vs\.? | and | y ", titulo.lower()):
            puntos -= 3
        candidatas.append((puntos, url, credito(ii.get("extmetadata", {})), titulo))
    if candidatas:
        _, url, cred, titulo = max(candidatas)
        return url, cred, titulo
    return None


def foto_principal_wikipedia(nombre, usadas):
    """Foto principal del articulo de Wikipedia (siempre es la persona/club correcto)."""
    for lang in ("es", "en"):
        r = api_json(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "format": "json", "titles": nombre, "redirects": 1, "prop": "pageprops"})
        for p in r.get("query", {}).get("pages", {}).values():
            archivo = p.get("pageprops", {}).get("page_image_free")
            if not archivo:
                continue
            c = api_json("https://commons.wikimedia.org/w/api.php", {
                "action": "query", "format": "json", "titles": "File:" + archivo, "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata", "iiurlwidth": 1400})
            for q in c.get("query", {}).get("pages", {}).values():
                ii = (q.get("imageinfo") or [{}])[0]
                url = ii.get("thumburl") or ii.get("url")
                if url and url not in usadas and ii.get("mime") in ("image/jpeg", "image/png"):
                    return url, credito(ii.get("extmetadata", {})), "Wikipedia: " + nombre
    return None


def normalizar_frase(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum() or c == " ")


def leer_pedido(item):
    """Cada imagen del guion puede ser texto o {"buscar": ..., "debe": [...], "wiki": ...}."""
    if isinstance(item, str):
        return {"buscar": item, "debe": [], "wiki": None}
    debe = item.get("debe") or []
    return {"buscar": item.get("buscar", ""), "debe": [debe] if isinstance(debe, str) else debe,
            "wiki": item.get("wiki")}


def bajar(url, dst):
    import requests
    for intento in range(4):
        try:
            resp = requests.get(url, headers=UA, timeout=40)
            if resp.ok and resp.headers.get("content-type", "").startswith("image/"):
                dst.write_bytes(resp.content)
                return Image.open(dst).convert("RGB")
        except Exception:
            pass
        time.sleep(2 + 3 * intento)
    return None


def imagenes_noticia(guion, n, carpeta):
    """Devuelve n tuplas (Image, credito, descripcion). Nunca rellena con fotos al azar de otra persona:
    si una busqueda no encuentra algo seguro, usa la foto de Wikipedia o repite una ya validada."""
    pedidos = [leer_pedido(x) for x in (guion.get("imagenes") or guion.get("palabras_clave", [])) if x]
    res, usadas = [], set()
    for ped in pedidos:
        if len(res) >= n:
            break
        hit = buscar_commons(ped["buscar"], ped["debe"], usadas) if ped["buscar"] else None
        if not hit and ped["wiki"]:
            hit = foto_principal_wikipedia(ped["wiki"], usadas)
        if not hit:
            print("Sin imagen segura para:", ped["buscar"])
            continue
        url, cred, titulo = hit
        img = bajar(url, carpeta / f"img{len(res)}.jpg")
        if img is None:
            continue
        res.append((img, cred, f'{ped["buscar"]} -> {titulo}'))
        usadas.add(url)
        print(f"Imagen {len(res)}: {ped['buscar']} -> {titulo}")
        time.sleep(1)
    if not res:
        hit = buscar_commons("football stadium crowd", [], usadas)
        img = bajar(hit[0], carpeta / "img0.jpg") if hit else None
        res = [(img or Image.new("RGB", (1200, 900), "#065f46"), hit[1] if hit else "", "generica")]
    base = list(res)
    while len(res) < n:                 # si faltan, se repiten las ya validadas
        res.append(base[len(res) % len(base)])
    return res[:n]


def hoja_contactos(imgs, salida):
    """Miniaturas numeradas de las imagenes usadas, para revisarlas antes de publicar."""
    th, cols = 300, 3
    filas = math.ceil(len(imgs) / cols)
    hoja = Image.new("RGB", (cols * th, filas * (th + 60)), "#111111")
    d = ImageDraw.Draw(hoja)
    for i, (img, _, desc) in enumerate(imgs):
        x, y = (i % cols) * th, (i // cols) * (th + 60)
        hoja.paste(cubrir(img, th - 10, th - 10), (x + 5, y + 5))
        lineas = envolver(d, f"{i + 1}. {desc}", F(14, False), th - 16)[:3]
        for k, l in enumerate(lineas):
            d.text((x + 8, y + th - 2 + k * 18), l, font=F(14, False), fill="#ffffff" if k == 0 else "#bbbbbb")
    hoja.save(salida, quality=80)


def cubrir(img, w, h):
    r = max(w / img.width, h / img.height)
    img = img.resize((math.ceil(img.width * r), math.ceil(img.height * r)), Image.LANCZOS)
    x, y = (img.width - w) // 2, max(0, int((img.height - h) * 0.3))  # sesgo hacia arriba (caras)
    return img.crop((x, y, x + w, y + h))


# ---------- 4. Musica sin copyright (generada aqui mismo) ----------
def musica_fondo(dur, salida):
    """Usa una pista de musica/ si existe; si no, compone una base deportiva original con numpy."""
    pistas = sorted(p for p in (RAIZ / "musica").glob("*") if p.suffix.lower() in AUDIO_EXT)
    if pistas:
        elegida = random.choice(pistas)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(elegida),
                        "-t", f"{dur + 1:.2f}", "-ac", "2", "-ar", "44100", str(salida)], check=True)
        print("Musica:", elegida.name)
        return
    sr, bpm = 44100, 118
    beat = 60 / bpm
    n = int((dur + 1.5) * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    acordes = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]   # Am F C G
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    total_beats = int((dur + 1.5) / beat)
    rng = np.random.default_rng(11)
    for b in range(total_beats):
        i0 = int(b * beat * sr)
        # bombo en cada tiempo
        L = int(0.25 * sr)
        tt = np.arange(min(L, n - i0)) / sr
        out[i0:i0 + len(tt)] += 0.55 * np.sin(2 * np.pi * (50 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
        # palmada en 2 y 4
        if b % 2 == 1:
            L = int(0.18 * sr)
            tt = np.arange(min(L, n - i0)) / sr
            out[i0:i0 + len(tt)] += 0.18 * rng.standard_normal(len(tt)) * np.exp(-tt * 22)
        # charles en corcheas
        for k in (0, 0.5):
            j = int((b + k) * beat * sr)
            L = int(0.05 * sr)
            if j + L < n:
                out[j:j + L] += 0.06 * rng.standard_normal(L) * np.exp(-np.arange(L) / sr * 80)
    compas = 4 * beat
    for c in range(int((dur + 1.5) / compas) + 1):
        a = acordes[c % 4]
        i0, i1 = int(c * compas * sr), min(n, int((c + 1) * compas * sr))
        tt = t[i0:i1] - t[i0]
        env = np.minimum(1, tt * 4) * np.minimum(1, (compas - tt) * 4)
        pad = sum(np.sin(2 * np.pi * hz(m) * tt) + 0.3 * np.sin(2 * np.pi * hz(m + 12) * tt) for m in a)
        bajo = np.sign(np.sin(2 * np.pi * hz(a[0] - 24) * tt)) * 0.5 + np.sin(2 * np.pi * hz(a[0] - 24) * tt)
        out[i0:i1] += env * (0.035 * pad + 0.09 * bajo * (np.sin(np.pi * 2 * tt / beat) > -0.2))
    fade = int(1.2 * sr)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    out = (out / (np.abs(out).max() or 1) * 0.8 * 32767).astype(np.int16)
    with wave.open(str(salida), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(out.tobytes())
    print("Musica: base original generada (sin copyright)")


# ---------- 5. Avatar pequeno ----------
def avatar_por_defecto(boca_abierta):
    a = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(a)
    d.ellipse((60, 40, 452, 470), fill="#f2c9a0")
    d.ellipse((170, 190, 220, 250), fill="#1f2937"); d.ellipse((292, 190, 342, 250), fill="#1f2937")
    if boca_abierta:
        d.ellipse((205, 320, 307, 410), fill="#7f1d1d")
    else:
        d.arc((195, 300, 317, 380), 20, 160, fill="#1f2937", width=10)
    return a


def avatar_circular(img, tam, marca):
    """Recorta el avatar en circulo con fondo y anillo del color del canal."""
    img = img.convert("RGBA")
    lado = min(img.size)
    img = img.crop(((img.width - lado) // 2, (img.height - lado) // 2,
                    (img.width + lado) // 2, (img.height + lado) // 2)).resize((tam, tam), Image.LANCZOS)
    base = Image.new("RGBA", (tam, tam), marca["color_primario"])
    base.alpha_composite(img)
    masc = Image.new("L", (tam, tam), 0)
    ImageDraw.Draw(masc).ellipse((0, 0, tam - 1, tam - 1), fill=255)
    borde = 10
    out = Image.new("RGBA", (tam + 2 * borde, tam + 2 * borde), (0, 0, 0, 0))
    ImageDraw.Draw(out).ellipse((0, 0, out.width - 1, out.height - 1), fill=marca["color_resaltado"])
    out.paste(base, (borde, borde), masc)
    return out


def cargar_avatar(marca, tam=210):
    imgs = sorted(p for p in (RAIZ / "assets").glob("*") if p.suffix.lower() in IMG_EXT)
    cerr = next((p for p in imgs if "cerrad" in p.name.lower()), imgs[0] if imgs else None)
    abie = next((p for p in imgs if "abiert" in p.name.lower()), next((p for p in imgs if p != cerr), cerr))
    if not cerr:
        a, b = avatar_por_defecto(False), avatar_por_defecto(True)
    else:
        print("Avatar:", cerr.name, "/", abie.name)
        a, b = Image.open(cerr), Image.open(abie)
    return {"cerrada": avatar_circular(a, tam, marca), "abierta": avatar_circular(b, tam, marca)}


# ---------- 6. Utilidades de dibujo ----------
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def envolver(draw, texto, fuente, ancho):
    lineas, linea = [], ""
    for p in texto.split():
        prueba = (linea + " " + p).strip()
        if draw.textlength(prueba, font=fuente) <= ancho:
            linea = prueba
        else:
            if linea:
                lineas.append(linea)
            linea = p
    if linea:
        lineas.append(linea)
    return lineas


def esquinas_redondas(img, r):
    masc = Image.new("L", img.size, 0)
    ImageDraw.Draw(masc).rounded_rectangle((0, 0, img.width - 1, img.height - 1), r, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), masc)
    return out


def sombra(tam, r, radio=30, alfa=160):
    s = Image.new("RGBA", (tam[0] + 4 * radio, tam[1] + 4 * radio), (0, 0, 0, 0))
    ImageDraw.Draw(s).rounded_rectangle((2 * radio, 2 * radio, 2 * radio + tam[0], 2 * radio + tam[1]), r, fill=(0, 0, 0, alfa))
    return s.filter(ImageFilter.GaussianBlur(radio))


def leer_volumen(wav_path):
    with wave.open(str(wav_path)) as w:
        sr, n = w.getframerate(), w.getnframes()
        datos = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            datos = datos.reshape(-1, w.getnchannels()).mean(axis=1)
    por = sr // FPS
    total = len(datos) // por
    rms = np.array([np.sqrt(np.mean(datos[i * por:(i + 1) * por] ** 2)) for i in range(total)])
    return rms / (rms.max() or 1), len(datos) / sr


# ---------- 7. Render ----------
# Disposicion (1080x1920). Todo lo importante queda entre y=80 y y=1450 y lejos del borde
# derecho, porque abajo y a la derecha YouTube/TikTok/Instagram ponen sus botones y textos.
TOP_Y = 90                      # cabecera: marca + titular
IMG_X, IMG_Y, IMG_W, IMG_H = 60, 430, 960, 720     # imagen principal (se ajusta bajo el titular)
AV_TAM = 210                    # avatar pequeno, montado en la esquina inferior izquierda de la imagen


def capa_fija(guion, marca):
    """Cabecera (marca + titular) y placa del avatar: no cambian durante el video."""
    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    ac, prim = marca["color_resaltado"], marca["color_primario"]
    # pastilla de marca con balon
    nombre = marca["nombre_canal"].upper()
    f_m = F(40)
    ancho = int(d.textlength(nombre, font=f_m)) + 110
    d.rounded_rectangle((60, TOP_Y, 60 + ancho, TOP_Y + 70), 35, fill=prim)
    cx, cy, r = 60 + 38, TOP_Y + 35, 22
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#ffffff")
    pent = [(cx + 9 * math.cos(math.radians(a - 90)), cy + 9 * math.sin(math.radians(a - 90))) for a in range(0, 360, 72)]
    d.polygon(pent, fill="#111111")
    for a in range(0, 360, 72):
        x0, y0 = cx + 9 * math.cos(math.radians(a - 90)), cy + 9 * math.sin(math.radians(a - 90))
        d.line((x0, y0, cx + 21 * math.cos(math.radians(a - 90)), cy + 21 * math.sin(math.radians(a - 90))), fill="#111111", width=3)
    d.text((60 + 75, TOP_Y + 13), nombre, font=f_m, fill="#ffffff")
    # titular
    f_t = F(64)
    lineas = envolver(d, guion["titular"].upper(), f_t, W - 170)[:3]
    while len(envolver(d, guion["titular"].upper(), f_t, W - 170)) > 3 and f_t.size > 44:
        f_t = F(f_t.size - 4)
        lineas = envolver(d, guion["titular"].upper(), f_t, W - 170)[:3]
    alto_l = f_t.size + 14
    y0 = TOP_Y + 100
    caja_h = 40 + alto_l * len(lineas)
    d.rounded_rectangle((60, y0, W - 60, y0 + caja_h), 24, fill=(0, 0, 0, 170))
    d.rounded_rectangle((60, y0, 74, y0 + caja_h), 7, fill=ac)
    for i, l in enumerate(lineas):
        d.text((100, y0 + 20 + i * alto_l), l, font=f_t, fill="#ffffff")
    return capa, y0 + caja_h


def render(guion, wav_path, salida_mp4, tmp):
    marca = cargar_marca()
    ac = marca["color_resaltado"]
    vol, dur = leer_volumen(wav_path)
    palabras = tiempos_palabras(wav_path, guion["texto"], dur)
    fin_gancho = 0
    if guion.get("gancho"):
        fin_gancho = palabras[min(len(guion["gancho"].split()), len(palabras)) - 1][2]

    n_img = max(1, math.ceil(dur / SEG_IMAGEN))
    imgs = imagenes_noticia(guion, n_img, tmp)
    hoja_contactos(imgs, Path(salida_mp4).with_name(Path(salida_mp4).stem + "-imagenes.jpg"))
    tarjetas, fondos, creditos = [], [], []
    for img, cred, _ in imgs:
        g = cubrir(img, int(IMG_W * 1.1), int(IMG_H * 1.1))       # margen para el zoom
        tarjetas.append(g)
        f = cubrir(img, W // 4, H // 4).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)
        f = Image.blend(f, Image.new("RGB", (W, H), marca["color_oscuro"]), 0.62)
        fondos.append(f.convert("RGBA"))
        creditos.append(cred)

    fija, fin_cabecera = capa_fija(guion, marca)
    global IMG_Y
    IMG_Y = max(IMG_Y, fin_cabecera + 40)
    sombra_img = sombra((IMG_W, IMG_H), 34)
    av_x, av_y = 28, IMG_Y + IMG_H - AV_TAM + 10      # sobre la esquina inferior izquierda de la imagen
    sub_y = IMG_Y + IMG_H + 95
    avatar = cargar_avatar(marca)
    f_sub, f_cred = F(80), F(22, False)
    grupos = [palabras[i:i + 3] for i in range(0, len(palabras), 3)]

    musica = tmp / "musica.wav"
    musica_fondo(dur, musica)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav_path), "-i", str(musica),
           "-filter_complex",
           # musica baja (-20 dB aprox) y se agacha aun mas cuando hablas (sidechain), voz intacta
           "[1:a]aresample=44100,aformat=channel_layouts=stereo,asplit=2[voz][llave];"
           "[2:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.13[mus];"
           "[mus][llave]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=350[musd];"
           "[voz][musd]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(salida_mp4)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total = int(dur * FPS)
    TRANS = 0.45
    for fr in range(total):
        t = fr / FPS
        idx = min(int(t / SEG_IMAGEN), n_img - 1)
        local = t - idx * SEG_IMAGEN
        marco = fondos[idx].copy()

        # tarjeta de imagen con zoom lento (Ken Burns) y entrada suave
        z = 1.0 + 0.08 * (local / SEG_IMAGEN)
        g = tarjetas[idx]
        cw, ch = int(g.width / 1.1 / z * 1.0), int(g.height / 1.1 / z * 1.0)
        ox, oy = (g.width - cw) // 2, (g.height - ch) // 2
        tarjeta = g.crop((ox, oy, ox + cw, oy + ch)).resize((IMG_W, IMG_H), Image.BILINEAR)
        tarjeta = esquinas_redondas(tarjeta, 34)
        entrada = min(1.0, local / TRANS) if idx > 0 else 1.0
        dy = int((1 - entrada) * 60)
        marco.alpha_composite(sombra_img, (IMG_X - 60, IMG_Y - 60 + 18 + dy))
        if entrada < 1:
            tarjeta.putalpha(tarjeta.getchannel("A").point(lambda a: int(a * entrada)))
        marco.alpha_composite(tarjeta, (IMG_X, IMG_Y + dy))
        d = ImageDraw.Draw(marco)
        d.rounded_rectangle((IMG_X, IMG_Y + dy, IMG_X + IMG_W, IMG_Y + IMG_H + dy), 34, outline=ac, width=5)
        if creditos[idx]:
            cr = creditos[idx][:70]
            d.text((IMG_X + IMG_W - 18 - d.textlength(cr, font=f_cred), IMG_Y + 16 + dy), cr, font=f_cred,
                   fill=(255, 255, 255, 210), stroke_width=2, stroke_fill="#000000")
        # puntos de progreso de imagenes
        for k in range(n_img):
            x = W // 2 + 60 - (n_img * 26) // 2 + k * 26
            d.ellipse((x, IMG_Y + IMG_H + 22, x + 14, IMG_Y + IMG_H + 36), fill=ac if k == idx else (255, 255, 255, 90))

        marco.alpha_composite(fija)

        # barra de progreso superior
        d.rectangle((0, 0, int(W * t / dur), 10), fill=ac)
        # etiqueta EN VIVO durante el gancho
        if t < fin_gancho:
            f_v = F(28)
            x0 = W - 60 - 70 - int(d.textlength("ÚLTIMA HORA", font=f_v))
            d.rounded_rectangle((x0, TOP_Y + 8, W - 60, TOP_Y + 62), 27, fill="#dc2626")
            if int(t * 2) % 2 == 0:
                d.ellipse((x0 + 20, TOP_Y + 25, x0 + 40, TOP_Y + 45), fill="#ffffff")
            d.text((x0 + 52, TOP_Y + 18), "ÚLTIMA HORA", font=f_v, fill="#ffffff")

        # subtitulos: 3 palabras, la actual resaltada
        grupo = next((gr for gr in grupos if gr[0][1] - 0.05 <= t <= gr[-1][2] + 0.15), None)
        if grupo:
            texto = " ".join(p[0] for p in grupo).upper()
            fs = f_sub
            while d.textlength(texto + " ", font=fs) > W - 100 and fs.size > 44:
                fs = F(fs.size - 6)
            x = (W - d.textlength(texto, font=fs)) / 2
            for p, ini, fin in grupo:
                pal = p.upper() + " "
                color = ac if ini - 0.03 <= t <= fin + 0.05 else "#ffffff"
                d.text((x, sub_y), pal, font=fs, fill=color, stroke_width=7, stroke_fill="#000000")
                x += d.textlength(pal, font=fs)

        # avatar pequeno que habla
        v = vol[min(fr, len(vol) - 1)] if len(vol) else 0
        av = avatar["abierta" if v > 0.18 else "cerrada"]
        marco.alpha_composite(av, (av_x, av_y - int(v * 10)))

        ff.stdin.write(marco.convert("RGB").tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg fallo al armar el video")


def armar_texto(guion):
    if "texto" not in guion:
        guion["texto"] = " ".join(guion[k].strip() for k in ("gancho", "desarrollo", "cierre"))
    n = len(guion["texto"].split())
    if not 75 <= n <= 115:
        print(f"AVISO: el guion tiene {n} palabras; para 30-45 s lo ideal es 80-110")
    return guion


def main():
    guion_path, carpeta_salida = Path(sys.argv[1]), Path(sys.argv[2])
    guion = armar_texto(json.loads(guion_path.read_text(encoding="utf-8")))
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    tmp = carpeta_salida / "tmp"
    tmp.mkdir(exist_ok=True)
    wav = tmp / "voz.wav"
    if os.environ.get("AUDIO_DE_PRUEBA"):
        normalizar_y_ajustar(os.environ["AUDIO_DE_PRUEBA"], wav)
    else:
        generar_voz(guion["texto"], wav)
    mp4 = carpeta_salida / f"{guion['id']}.mp4"
    render(guion, wav, mp4, tmp)
    print("LISTO", mp4)


if __name__ == "__main__":
    main()
