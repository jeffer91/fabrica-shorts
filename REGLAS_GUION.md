# Reglas para cada guion de 11 Records (fútbol mundial)

Estas reglas las sigue Claude cada vez que escribe un guion para la fábrica.

## Duración y estructura (30 a 45 s)
- 80 a 110 palabras en total.
- **gancho** (1 frase, máx. 18 palabras): dato impactante, pregunta, anécdota, contradicción o "esto te afecta". Nunca saludar.
- **desarrollo** (55 a 80 palabras): qué pasó, quién, cuándo y por qué importa. Frases cortas.
- **cierre** (12 a 20 palabras): conclusión + llamado a comentar o seguir a 11 Records.
- Los subtítulos salen **exactamente** del texto del guion: revisar ortografía y tildes, porque se ven en pantalla.
- Escribir los números como se dicen si son raros ("ocho Balones de Oro"); los años pueden ir en cifras.

## Que suene natural (no robótico)
- Escribir como se habla, no como se lee un periódico: "Ojo con esto", "Mira", "¿Y sabes qué?", "Imagínate".
- Frases de 6 a 15 palabras. Mezclar una pregunta y una exclamación en cada guion: dan subidas y bajadas de tono.
- Usar comas donde uno respira; el modelo hace la pausa ahí.
- Evitar siglas raras y paréntesis; escribir nombres extranjeros como se pronuncian si son difíciles.

## Verificación de datos
- Confirmar cada dato en al menos 2 fuentes confiables (medios deportivos, sitio oficial del club, liga o FIFA/UEFA).
- Si es rumor o fichaje no oficial, decirlo ("según reportes de…").
- Nada de cifras, citas o resultados inventados. Si falta un dato, quitarlo.

## Imágenes (una cada 5 s → 7 a 9 por video)
Cada imagen es un objeto, en el orden en que se narra:
```json
{"buscar": "Lionel Messi Inter Miami", "debe": ["Messi"], "wiki": "Lionel Messi"}
```
- **buscar**: términos en inglés: persona + club/selección + año/torneo si importa.
- **debe**: palabras que TIENEN que estar en el nombre del archivo de la foto. Usar el apellido de la persona
  que se nombra en ese momento. Si el apellido es ambiguo, poner nombre y apellido
  (Cristiano Ronaldo → ["Cristiano", "Ronaldo"], no solo "Ronaldo").
- Si el nombre puede ser de varias personas (ej. "Aleksandar Pavlović" también es basquetbolista), poner en
  "debe" el apellido + el club (["Pavlovic", "Bayern"]) y en "wiki" el título con aclaración, o no poner "wiki".
- **wiki**: título exacto del artículo de Wikipedia de esa persona/club; se usa si la búsqueda no encuentra una foto segura.
- No pedir trofeos genéricos ("Ballon d'Or trophy") sin "debe": salen fotos de otros jugadores.
- Para objetos/estadios sin persona: `{"buscar": "Santiago Bernabéu stadium", "debe": ["Bernabeu"]}`.
- La fábrica nunca rellena con fotos al azar: si una búsqueda falla, repite una imagen ya validada.

## Revisión antes de publicar (obligatoria)
- La fábrica deja en `salidas/<id>-imagenes.jpg` una hoja con todas las imágenes numeradas.
- Claude la revisa: si alguna muestra a otra persona o no corresponde a la noticia, corrige ese pedido
  en el guion y vuelve a generar el video. Solo se programa en Metricool cuando todas están bien.
