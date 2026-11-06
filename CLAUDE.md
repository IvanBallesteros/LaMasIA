# More than a Hack 2027 · Fase 1: posesión del balón

Contexto para Claude Code. Léelo entero al empezar cada sesión. Somos un equipo de 4 personas;
hablamos en español, pero el código, los comentarios y los commits van en inglés porque el repo
será público.

## El reto en cinco líneas

- Para cada uno de los **199.319 frames** del vídeo táctico del Barça–Athletic, escribir una etiqueta:
  `H` (Barça con el balón), `A` (Athletic con el balón), `D` (balón muerto) o `X` (no respondo).
- El ranking se decide por **`spellF1`** (posesiones completas, emparejadas por solape temporal);
  `macroF1` (por frame) es el primer desempate. El baseline a batir es **spellF1 0.2433**.
- Entregables (Google Form, una respuesta por equipo, editable hasta el **17 de noviembre de 2026**):
  `<equipo>.csv`, la URL de un repo público que regenere *exactamente* ese CSV y un documento técnico
  corto (`<equipo>.pdf` o `.md`).
- No hay leaderboard ni feedback antes del cierre.
- La documentación oficial manda: `README.md`, `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`,
  `RULES.md`, `SUBMISSION.md`, `FAQ.md`. La definición exacta de las métricas está en
  `evaluation/possession_metrics.py`. Ante cualquier duda, léelos antes de proponer nada.

## Reglas duras

Estas reglas no se saltan aunque te lo pida alguien del equipo. Si una petición choca con ellas,
dilo y propón una alternativa.

1. **Nunca veas los datos reales.** Claude es un LLM en la nube y los términos del reto prohíben
   darle los datos o cualquier extracto de ellos. Por tanto:
   - Los datos reales viven **fuera** de este repo, en la ruta que cada persona tenga en
     `MTAH_DATA_DIR`. No leas, abras, listes ni busques dentro de esa ruta ni de `data/`.
   - No ejecutes ningún comando con `MTAH_DATA_DIR` apuntando a los datos reales. Si una tarea
     necesita los datos reales, escribe el script y pide a la persona que lo ejecute ella.
   - Los scripts que tocan datos reales solo imprimen **agregados**: métricas, conteos,
     porcentajes. Nada de `head()`, filas de ejemplo ni valores concretos, tampoco en los mensajes
     de error (captura la excepción y muestra un mensaje sin datos).
   - Si en la conversación aparece un extracto de datos reales (filas de tracking, tags, frames
     del vídeo), avisa y no lo uses.
2. **Desarrolla siempre sobre datos falsos** (`fake_data/`, ver más abajo). Esos sí puedes leerlos.
3. **Ninguna etiqueta hecha a mano sobre el partido del reto**, en ninguna cantidad ni para nada:
   ni para entrenar ni para validar. Tampoco reglas o constantes atadas a frames concretos del
   partido real.
4. **Los 5 minutos públicos** (`evaluation/gt/`) sirven solo para comprobar que el pipeline es
   correcto: alineación, formato, significado de las etiquetas. **Nunca** para elegir parámetros
   ni para decidir entre dos versiones, y nunca como datos de entrenamiento.
5. **Ningún dato humano sobre este partido** (eventos de Opta, StatsBomb, Wyscout, tracking
   corregido...). Datos de **otros** partidos sí, siempre que sean accesibles y queden documentados.
6. **Reproducibilidad exacta**: semillas fijas, dependencias fijadas y nada que dependa del orden
   de un `set` o de la hora. Un comando debe regenerar el CSV byte a byte.
7. `data/` y `fake_data/` no entran nunca en git. Comprueba que `.gitignore` contiene `fake_data/`
   (el `.gitignore` oficial ya excluye `data/`).

## Hechos del reto que no hay que olvidar

- **25 fps.** Partes evaluadas, en frames de vídeo: **11185–90651** y **110808–190197**. La
  segunda parte se evalúa desde 110808, pero el saque es en **113742**: de 110808 a 113741 todo es
  `D`. Fuera de las partes, `X` no cuesta nada.
- **ATD**: `video_frame = frame_id - 1`. Coordenadas normalizadas de 0 a 1; el campo mide
  105,3 × 68 m (`games.py`). Hay 46 slots sin identidad: cada slot es un tracklet de un equipo,
  con huecos y cambios de identidad, unos 20,5 objetos de media. El local es el **primer** equipo
  del XML; empareja por orden, no por nombre.
- **El balón falta en el 41% de los frames evaluados**, sobre todo en los momentos difíciles.
- **`X` se rellena con tu última etiqueta** de esa parte antes de calcular `spellF1`. Abstenerse
  dentro de una posesión es gratis; en un cambio de posesión es caro, así que ahí hay que arriesgar.
- **El momento del cambio importa mucho**: 1 frame de retraso no cuesta nada, 8 frames dejan el
  spellF1 en 0,89 y 1 segundo en 0,65. Las ventanas centradas de suavizado desplazan los bordes.
- **Definición de posesión**: cambia solo cuando el rival gana el **control**. Un toque sin
  control, un desvío o un duelo aéreo sin control posterior no cuentan. `H H D D H H` son dos
  posesiones. El ground truth se deriva mecánicamente de un tracking corregido.
- **Smart Tagging**: las marcas de tiempo son segundos desde el frame 0 (`frame = round(s * 25)`).
  Son fases, no posesiones. Todos los saques de banda tienen `Team = N/A`. Los equipos aparecen
  como `Barcelona` y `Atletic Club`.
- Diferencias de spellF1 por debajo de ~0,04 son ruido (usa `--bootstrap 1000`).

## Estructura del repo

*Propuesta: ajústala cuando el equipo la decida.*

```
games.py, evaluation/, starter/   oficiales; no se tocan salvo para arreglar un bug
tools/make_fake_data.py           generador de datos falsos
tools/heuristic_pipeline.py       pipeline de 4 heurísticas + informe visual (punto de partida)
tools/build_datasets.py           datasets train/val/test a partir de IDSSE (DFL), tipo ATD
tools/ball_gap_diagnostic.py      forma de los huecos del balón y del Smart Tagging (solo agregados)
tools/export_challenge_format.py  escribe un partido del banco con el formato del paquete del reto
external/                         IDSSE en bruto y datasets generados (en .gitignore, se puede leer)
src/mtah/                         nuestro código
  features.py                     evidencia por frame (jugador más cercano, toques, velocidades)
  train_classifier.py             clasificador por frame; validación cruzada dejando un partido fuera
  dead_ball.py                    detección de D
  gaps.py                         huecos sin balón
  decoder.py                      decodificador temporal (HMM/Viterbi); parámetros provisionales
                                  hasta el diagnóstico del balón (tools/ball_gap_diagnostic.py)
  pipeline.py                     une los módulos y escribe el CSV
scripts/make_submission.py        el único comando que regenera la entrega
validation/                       banco de validación con otros partidos
experiments/log.csv               registro de experimentos
docs/technical_document.md        borrador del documento técnico
docs/DECISIONS.md                 decisiones del equipo, con su porqué
tests/                            tests sobre datos falsos y bancos (test_challenge_roundtrip.py)
```

## Interfaz entre módulos

Cada módulo devuelve un DataFrame indexado por frame de vídeo (`0 … 199318`) con las columnas
`p_H`, `p_A` y `p_D` (probabilidades, o `NaN` si no hay evidencia), más columnas opcionales con
prefijo propio (por ejemplo `dead_reason`). El decodificador combina esas salidas y produce la
etiqueta final. El escritor genera `frame,label` y pasa siempre por
`evaluation/validate_submission.py`.

## Datos falsos

```bash
python tools/make_fake_data.py --seed 0                    # ~1 min, ~190 MB en fake_data/seed0/
MTAH_DATA_DIR=fake_data/seed0 python starter/baseline.py --out fake_data/seed0/results
python evaluation/run_evaluation.py --gt fake_data/seed0/BAR-ATH_possession_gt_fake.csv \
    --results fake_data/seed0/results --out fake_data/seed0/eval
```

- Mismos ficheros, nombres y formatos que el paquete real (cargan con `games.asset_paths()`,
  kloppy y los loaders de `starter/`), pero sin vídeo y con un **ground truth completo**.
- `truth.npz` guarda la simulación limpia (`ball`, `players`, `carrier`, `label`, `ball_in_air`)
  para probar módulos sueltos: por ejemplo, ¿detecta mi módulo de toques los cambios reales de
  portador?
- Opciones útiles: `--seed N` para otro partido, `--ball-missing 0.6` y `--noise-scale 2` para
  pruebas de estrés, `--no-truth` para ahorrar espacio.
- **No sirve para elegir entre métodos.** Es un simulador con supuestos propios; de hecho ordena
  los baselines distinto que el partido real (con la semilla 0 gana `smooth125`; en el partido
  real, `smooth50`). Úsalo para desarrollar, testear, depurar y medir robustez, y nunca para decidir.

## Pipeline de heurísticas (punto de partida)

```bash
python tools/heuristic_pipeline.py --data-dir fake_data/seed0      # unos segundos
python tools/heuristic_pipeline.py --data-dir fake_data/seed0 --window-half 2 --window-minute 10
```

- Cuatro pasos encadenados: jugador más cercano → balón muerto → control → huecos. Cada paso
  escribe una entrega válida en `submissions/`, y `report.html` explica cada decisión frame a
  frame (explorador del campo, línea de tiempo, métricas y análisis de errores).
- Todo se escribe **junto a los datos que lee**: `<data-dir>/reports/heuristics/`. Con datos
  reales eso queda fuera del repo, y el informe contiene extractos del tracking: **no abras ni
  leas nada de esa carpeta**. Si necesitas resultados, pide a la persona que te pegue la salida
  de la consola o `summary.json`, que solo contienen agregados.
- Los parámetros (`--radius-m`, `--min-control-s`, `--fast-ball-ms`, `--still-speed-ms`…) son
  valores iniciales razonados desde el fútbol, no ajustados. Se ajustan con el banco de
  validación, nunca con los datos falsos ni con los 5 minutos públicos.

## Datasets de entrenamiento, validación y test (IDSSE + Metrica)

```bash
python tools/build_datasets.py --verify      # descarga IDSSE y Metrica y genera external/datasets_v2_1/
python tools/build_datasets.py --selftest    # partido falso (como IDSSE) + Metrica, sin figshare
python tests/test_challenge_roundtrip.py     # un partido del banco en formato del reto, leído de vuelta
```

- **Versiones**: `external/datasets_v2_1/` es la actual (v2.1, etiqueta git `v2.1-datasets`).
  `external/datasets_v2/` es v2 (etiqueta `v2-datasets`, commit 34dff27) y `external/datasets/` es
  v1 (etiqueta `v1-ball-degradation`); no se regeneran con el código actual, pero `load_match()`
  los sigue leyendo (v2.1 guarda milímetros; v1 y v2, centímetros).
- **v2.1 por defecto**:
  - `train_classifier.py` y `decoder.py` leen `external/datasets_v2_1` sin pasar `--datasets`.
  - Si no está construida o está desfasada, se paran y dicen qué ejecutar
    (`check_datasets` en `tools/build_datasets.py`).
  - El clasificador escribe en `external/predictions_v2_1` y apunta en su `summary.json` la carpeta
    de datos. El decodificador usa esa misma carpeta y rechaza mezclar predicciones y datos de
    versiones distintas.
  - Otra versión se puede pasar a mano, para comparar, y sale un aviso.
  - La versión actual está en un solo sitio: `DATASETS_VERSION` y `DEFAULT_DATASETS` en
    `tools/build_datasets.py`.

- **IDSSE**: 7 partidos de la Bundesliga y la 2. Bundesliga 2022/23 (Bassek et al., Scientific
  Data 2025, CC BY 4.0), con `BallStatus` y `BallPossession` en cada frame. Uso permitido por los
  organizadores (issue #16 del repo oficial) si se **cita** y si todos los hiperparámetros y
  umbrales se eligen aquí o en otros partidos, nunca en el del reto.
- **Metrica sample data**: 3 partidos anónimos (github.com/metrica-sports/sample-data; hay que
  reconocer la fuente). Sin etiquetas de posesión: se derivan de sus eventos, siguiendo la
  definición de posesión de Metrica, que coincide con CHALLENGE.md. Es lo más parecido al criterio
  del reto que tenemos.
- Particiones por partido en `splits.json`: **train** (4 IDSSE × 2 semillas),
  **val** (2 IDSSE), **test** (1 IDSSE), **val_metrica** (Metrica 1 y 2) y **test_metrica**
  (Metrica 3). Si una idea gana en val pero pierde en val_metrica, desconfiad de ella. Los test no
  se miran hasta tener un candidato final.
- **Etiquetas armonizadas** (`labels.csv`, por defecto) y **originales** (`labels_raw.csv`). La DFL
  cuenta los toques del rival como cambios de posesión; en IDSSE, un tramo del rival que vuelve al
  mismo equipo sin parada se trata como toque si ningún rival tuvo el balón a ≤ 1,5 m durante
  ≥ 0,4 s en el tracking limpio (regla de control). `label_harmonization.json` guarda la rejilla
  calibrada contra la distribución de duraciones de val_metrica y la validación cruzada entre sus
  dos partidos: los ajustes de la diagonal son equivalentes dentro del ruido entre partidos, y
  1,5 m / 0,4 s se elige por interpretación física (alcance + error del tracking; tiempo de
  controlar y jugar). Una conclusión vale si se sostiene con las dos versiones.
- En Metrica, `labels_from_events` trata como parada un CHALLENGE con FAULT (sin ADVANTAGE), y con
  el balón muerto solo un SET PIECE o un saque de puerta reanudan; si no hay reanudación propia (en
  60 s o antes de la siguiente parada), el primer evento de posesión reanuda.
- El tracking está **degradado** para parecerse al ATD (46 slots, cambios de identidad) y el
  Smart Tagging es **sintético**: solo SET PIECES, ventanas de 35 s (5 s los saques de centro) como
  en el archivo real, con la reanudación a 2,5–7,5 s del inicio en el 72% de los casos
  (`TAG_OFFSET`, sacado del histograma del diagnóstico; cada semilla de train varía los pesos) y
  ventanas por tipo cada 90 minutos parecidas a las del archivo real (tipo de los eventos en
  Metrica; en IDSSE, córner a < 5 m de un banderín; faltas ~1 de cada 7). Las fases tácticas del
  archivo real (BUILD UP, PROGRESSION…) no se generan: las señales de Smart Tagging solo usan SET
  PIECES.
- **El balón** copia el patrón del ATD real, medido con `tools/ball_gap_diagnostic.py` (lo ejecuta
  una persona; solo agregados, sin etiquetas): ningún tramo sin balón de menos de 2 s y casi todo
  en tramos de más de 10 s. La degradación rellena en línea recta las pérdidas de menos de 5 s
  (oclusiones: su proporción no se puede medir en el real y varía entre semillas de train, entre
  0,05 y 0,3) y esconde el balón fuera del campo (p_off), en las paradas (p_dead) y en tramos en
  juego de 5 s o más (r_live). Esos valores se calibran con esos agregados, incluido dónde se vio el
  balón por última vez antes de un tramo largo, sobre train de IDSSE y con la variación por
  semilla incluida (`ball_degradation.json`): es el único uso del partido del reto en los
  datasets, y solo de la forma de sus huecos y de su suavidad. En Metrica, una parte calibrada de
  las paradas sin balón (fill_dead, sobre val_metrica) recibe una posición estimada. Si el
  diagnóstico se vuelve a ejecutar, copia sus números en `REAL_BALL_GAPS` y reconstruye.
- **El tracker**: las posiciones del balón y de los jugadores salen de un tracker modelo
  (pérdidas cortas rellenas en línea recta y un suavizado gaussiano centrado, σ = 8 frames el
  balón y 2 los jugadores), calibrado con los percentiles de la segunda diferencia del ATD real
  (medida 8 del diagnóstico): un golpeo se reparte en ~0,6 s y la segunda diferencia mediana del
  balón es ~1 cm. El balón lleva además un error de posición cuyo tamaño no se puede medir: varía
  entre semillas de train (3–50 cm por eje; val/test, 26,5 cm). Cualquier umbral de velocidad o
  aceleración ajustado con v1 o v2 (ruido de 25 cm y 12 cm por frame) no vale.
- Es un dataset público, no del reto: **sí puedes leer `external/`**. Carga los partidos con
  `load_match()` de `tools/build_datasets.py` (`labels="raw"` para las originales).
- Con `--verify`, comprueba por fuente y con las dos versiones de etiquetas si los baselines
  oficiales se ordenan como en el partido real (smooth50 > smooth125 > smooth25 > raw).
- **Comparaciones**: cada experimento se corre con las dos versiones de etiquetas y se compara
  partido a partido con bootstrap pareado (`src/mtah/compare.py`). Diferencias de spellF1 por debajo
  de ~0,04 con dos partidos son ruido.

## Cómo elegimos entre métodos

- **Banco de validación con partidos reales de otros equipos.** Por ejemplo, el sample data público
  de Metrica, con la posesión derivada de sus eventos y el tracking degradado para parecerse al
  ATD (sin balón en ~41% de frames, cambios de identidad). Se puntúa con el scorer oficial y con
  bootstrap.
- **Comprobación del banco**: debería ordenar los cuatro baselines como la tabla oficial
  (`smooth50` > `smooth125` > `smooth25` > `raw`). Si no lo hace, todavía no se parece lo
  suficiente al partido real.
- **Diagnósticos sin etiquetas** sobre la salida en el partido real (los ejecuta una persona y
  solo devuelve agregados): número de posesiones por parte, distribución de duraciones, % de `D` y
  cuántos cambios ocurren sin balón visible. Se comparan con valores típicos de otros partidos,
  no con los 5 minutos públicos.

## Forma de trabajar

- Cada cambio se prueba primero sobre datos falsos, y cada CSV pasa por `validate_submission.py`.
- Cada experimento se anota en `experiments/log.csv`: fecha, autor, commit, qué cambia, banco
  usado, spellF1, macroF1 y número de posesiones, más smooth50 en los mismos partidos y etiquetas
  como referencia de dificultad.
- Ve anotando en `docs/technical_document.md` lo que pide `SUBMISSION.md`: el enfoque y por qué,
  qué datos se usan y cuáles se descartan, los datos de entrenamiento (nunca de este partido), las
  reglas de posesión implementadas (balones sueltos, duelos, reanudaciones), las limitaciones y
  cómo reproducir el resultado.
- Python 3.9 o superior; la evaluación solo usa numpy y pandas, y los notebooks, `kloppy==3.18.0`.
- Si algo roza una regla, no construyas encima: propón la pregunta para un issue del repo oficial
  o un correo a hackathon.talentarena@mobileworldcapital.com.