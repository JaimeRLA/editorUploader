# editoruploader

Gestiona **varios canales de YouTube Shorts, cada uno de una temática**. Para cada canal analiza
qué vídeos largos están siendo virales en su temática, elige los mejores momentos *para ese canal*,
los monta como pequeñas historias verticales a pantalla completa (gancho, cortes, zooms,
transiciones, subtítulos y música) y los sube a su canal. Todo gratis y en tu PC.

```bash
python -m editoruploader channel add            # crea un canal (pregunta temática, idioma y fuentes)
python -m editoruploader channel auth ciencia   # lo conecta a su canal de YouTube
python -m editoruploader run --all              # procesa y sube en todos los canales
```

```
Por cada canal (channels/<id>/channel.toml)
  capa viral     →  búsquedas de la temática (las del canal + las que propone el LLM cada día),
        │           ordenadas por visitas del último mes; nota = visitas/día, visitas frente a la
        │           mediana de su canal, participación y encaje con la temática (LLM)
        ▼           → channels/<id>/viral.txt
  faster-whisper (GPU)  →  transcripción palabra a palabra      ┐ compartido entre canales
        ▼                                                         ┘ (work/<vídeo>/)
  LLM local (Ollama)    →  momentos que encajan con la temática y GUION de cada Short:
        │                  gancho (la frase más fuerte primero) + tramos sin relleno
        ▼
  montaje               →  pantalla completa siempre (sigue a quien habla o lo importante de la
        │                  imagen, corta con los cambios de plano), cortes de salto sin silencios,
        │                  zoom de golpe en los cortes y acercamiento lento, destello + «whoosh»
        │                  entre partes, subtítulos palabra a palabra con palabras clave, barra de
        ▼                  progreso y música según el ambiente (baja sola con la voz)
  YouTube Data API v3   →  su canal, con créditos y un máximo diario
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m editoruploader setup   # descarga ffmpeg, deno, modelo de caras, fuentes y música
```

`setup` no necesita sudo y al final comprueba la GPU y Ollama.

### Ollama (el LLM que elige los momentos)

Instala [Ollama](https://ollama.com/download) y descarga un modelo:

```bash
ollama pull qwen3:8b       # funciona con cualquier Ollama reciente
ollama pull qwen3.5:9b     # mejor; requiere Ollama actualizado (luego OLLAMA_MODEL=qwen3.5:9b en .env)
```

### YouTube (una vez para todos los canales)

1. En <https://console.cloud.google.com> crea un proyecto y habilita **YouTube Data API v3**
   (**APIs y servicios → Biblioteca**).
2. **Google Auth Platform** → configura la pantalla de consentimiento con tipo de usuario **Externo**.
   En **Público → Usuarios de prueba** añade el Gmail de **cada** cuenta con la que vayas a conectar
   canales (las cuentas de marca van con el Gmail que las administra).
3. **Clientes → Crear cliente** → **Aplicación de escritorio** → descarga el JSON y guárdalo como
   `credentials/client_secret.json`. Este único cliente sirve para todos los canales.

Con la app en modo **Prueba**, Google caduca cada sesión a los 7 días. Cuando pase, `run` lo avisa
y deja los Shorts pendientes; se arregla con `python -m editoruploader channel auth <canal>`.

## Canales

### Crear un canal

```bash
python -m editoruploader channel add
```

Pregunta un identificador corto (`ciencia`, `motor-es`…), el nombre, la temática en una frase, el
idioma y, opcionalmente, las fuentes. Con la temática, el LLM propone el resto: qué momentos buscar
y cuáles evitar, búsquedas para encontrar vídeos, ambientes de música, tipografía, color, categoría
y etiquetas. Todo queda en `channels/<id>/channel.toml`, comentado, para que lo ajustes.

Al final ofrece conectarlo a YouTube. Al iniciar sesión, si tu Gmail administra varios canales
(cuentas de marca), **elige el canal correcto**: queda registrado en `channels/<id>/youtube.json` y
el programa se niega a subir si la sesión resulta ser de otro canal.

Se puede crear también sin preguntas:

```bash
python -m editoruploader channel add motor --name "Motor Puro" --language es \
  --niche "coches, motos y competición" --url "https://www.youtube.com/@canalfuente" --no-auth
```

### Personalizar un canal

Edita `channels/<id>/channel.toml`. Los cambios se aplican en la siguiente ejecución y los Shorts
aún no subidos se vuelven a renderizar con la edición nueva.

| Sección | Qué controla |
|---|---|
| `niche`, `language` | temática (el LLM solo elige momentos que encajen) e idioma de títulos y descripciones |
| `[sources]` | `mode` (`viral` o `latest`), `period` del análisis viral (`week`, `month`, `year`), `urls`, `searches`, `videos_per_run`, duración de las fuentes, `languages` habladas admitidas, `exclude` y `only_creative_commons` |
| `[clips]` | Shorts por vídeo, duración (60-90 s por defecto, hasta 180), nota mínima, `focus` y `avoid` para el LLM |
| `[edit]` | montaje (`story`, `jump_cuts`, `zoom_effects`, `transitions`, `progress_bar`), subtítulos (fuente, tamaño, colores, color de palabras clave, mayúsculas, palabras por línea, altura), texto gancho, música (volumen y ambientes) y encuadre (`auto` o `center`; siempre a pantalla completa) |
| `[upload]` | activar la subida, privacidad, categoría, etiquetas, máximo diario y plantilla de descripción |

Tipografías incluidas: Poppins ExtraBold, Anton, Bebas Neue, Archivo Black, Luckiest Guy, Bangers y
Permanent Marker (puedes añadir cualquier `.ttf` a `tools/fonts/`).

Música propia de un canal: pon tus temas en `channels/<id>/music/<ambiente>/` (ambientes:
`energetic`, `funny`, `chill`, `emotional`, `tense`, `inspiring`). Si esa carpeta tiene música, se usa
en vez de la global (`music/`, con 42 temas de Kevin MacLeod, CC BY 4.0, que `setup` descarga).

### Ejecutar

```bash
python -m editoruploader run ciencia            # vídeos nuevos de las fuentes de «ciencia»
python -m editoruploader run --all              # todos los canales, uno tras otro
python -m editoruploader run ciencia "https://www.youtube.com/watch?v=XXXXXXXXXXX"   # un vídeo concreto
python -m editoruploader run ciencia --no-upload    # crear los Shorts sin subirlos
python -m editoruploader upload --all           # subir lo pendiente
python -m editoruploader status                 # resumen de todos los canales
python -m editoruploader status ciencia         # detalle de un canal, Short a Short
python -m editoruploader channel list
```

Cada `run`:
1. Sube primero los Shorts pendientes de otros días (hasta `max_per_day`).
2. Con `mode = "viral"`, analiza lo que más funciona en la temática y coge los `videos_per_run`
   mejores que ningún canal tuyo haya usado (sin repetir tema con los últimos); el ranking queda en
   `channels/<canal>/viral.txt`. Con `latest`, coge lo más reciente de las fuentes. Comprueba
   idioma (y licencia, si se pide) antes de descargar.
3. Elige los momentos, escribe el guion de cada Short y lo monta en `output/<canal>/<vídeo>/`,
   junto a `candidatos.txt` (todas las propuestas del LLM con su nota, tramos y motivo).
4. Los sube por orden de nota, sin pasar del máximo diario. El resto espera a la siguiente ejecución.

Si se acumulan más de 3 días de Shorts sin subir, no procesa vídeos nuevos hasta que bajen.

`run --all` no hace preguntas ni abre el navegador, así que se puede programar. Por ejemplo, en el
Programador de tareas de Windows, una tarea diaria que ejecute:

```
wsl -d Ubuntu -- bash -lc "cd ~/editoruploader && .venv/bin/python -m editoruploader run --all >> run.log 2>&1"
```

## Derechos de autor y Content ID

- Con `only_creative_commons = true` solo usa vídeos que YouTube permite reutilizar citando al autor
  (la cita se añade sola a la descripción). Es lo que evita reclamaciones.
- Con `false` usa cualquier vídeo, también los virales de otros creadores. Content ID puede
  reclamarlos (los ingresos van al dueño), bloquearlos o dar avisos de copyright; con 3 avisos
  YouTube cierra el canal. La lista `exclude` evita al menos las cadenas y estudios que suelen
  bloquear sus vídeos en todo el mundo.
- El programa no hace nada para ocultar el contenido a Content ID.
- La música es CC BY y su crédito se añade solo.

## Límites de YouTube

- La API permite **100 subidas al día por proyecto de Google Cloud, sumando todos los canales**.
  Reparte `max_per_day` entre ellos.
- Cada canal tiene además su propio límite diario, más bajo si no está verificado. Verifícalos en
  <https://www.youtube.com/verify>.
- Los Shorts se suben en **privado** por defecto (`privacy` en `[upload]`), para revisarlos antes de
  publicarlos.
