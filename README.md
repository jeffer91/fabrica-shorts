# Fábrica de shorts

Convierte un guion en un short vertical (1080x1920) con tu voz clonada, tu avatar y subtítulos.

## Cómo funciona
1. Claude escribe un guion en `guiones/AAAA-MM-DD-NN.json` y lo sube al repositorio.
2. GitHub Actions (`.github/workflows/generar-short.yml`) se activa solo:
   - Chatterbox Multilingual genera la voz en español clonando `voz/referencia.wav`.
   - faster-whisper calcula el tiempo de cada palabra para los subtítulos.
   - Pexels pone imágenes de apoyo (opcional, secreto `PEXELS_API_KEY`).
   - Se anima el avatar (`assets/avatar_cerrada.png` y `assets/avatar_abierta.png`).
3. El video se sube a Supabase Storage (bucket público `shorts`) y su enlace queda en `salidas/<id>.json`.
4. Claude toma ese enlace y lo programa en Metricool.

## Formato del guion (75-110 palabras = 30-45 s)
```json
{
  "id": "2026-11-26-01",
  "titular": "Titular corto en pantalla",
  "gancho": "1 frase, 0-3 s: dato, pregunta o anecdota que detiene el scroll.",
  "desarrollo": "Los hechos verificados, 20-35 s.",
  "cierre": "Conclusion + llamado a seguir, 5 s.",
  "imagenes": ["6 a 9 busquedas en orden de la narracion, una por cada 5 s: nombres de jugadores, clubes, estadios, trofeos"],
  "fuente": "https://enlace-de-la-noticia"
}
```

## Archivos que faltan poner
- `voz/referencia.wav`: 10 a 30 s de tu voz, limpia y sin música.
- `assets/avatar_cerrada.png` y `assets/avatar_abierta.png`: tu caricatura con fondo transparente.
- `marca.json`: nombre del canal y colores.

## Secretos del repositorio (Settings → Secrets → Actions)
- `SUPABASE_URL` y `SUPABASE_KEY`: para alojar los videos.
- `PEXELS_API_KEY`: opcional, gratis en pexels.com/api.
