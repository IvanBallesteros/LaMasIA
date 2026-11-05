# Decisiones

Una entrada por decisión: qué se decidió, por qué, con qué datos y qué queda abierto. Las más
recientes, arriba.

## 2026-11-05 · Datasets v2.1: tracker suavizado, posiciones en mm y ventanas por tipo (paquete A)

**Qué.** `tools/build_datasets.py` genera `external/datasets_v2_1/`. v2 (`external/datasets_v2/`,
commit 34dff27) y v1 se quedan como estaban. `load_match()` no cambia y lee las tres versiones.

- **Posiciones en milímetros** (decisión del equipo): `tracking.npz` guarda `ball_mm` y
  `slots_mm` en int32, en lugar de `ball_cm` y `slots_cm` en int16. `truth.npz` sigue en cm.
  - Por qué: en el ATD real, la segunda diferencia mediana es de 9 mm en el balón y 1,4 mm en los
    jugadores. En cm, el redondeo dejaba un suelo de ~8 mm.
- **Tracker modelo**:
  - Las pérdidas de menos de 5 s se rellenan en línea recta, sin el error que crecía con la
    distancia.
  - Después, un suavizado gaussiano centrado, de σ = 8 frames en el balón y 2 en los jugadores.
    Es centrado porque suponemos que el ATD es un producto offline.
  - Se calibra con los percentiles de la segunda diferencia del ATD real. La cola, que son los
    golpeos, fija la anchura: un golpeo se reparte en ~0,6 s.
  - Sin ruido independiente por frame y sin detecciones falsas del balón.
- **Error de posición del balón, no calibrado**:
  - Tiene dos números: su RMS por eje y la desviación de su segunda diferencia, que se calibra
    (0,006 m).
  - El RMS no se puede calibrar: uno de 3 cm que cambia rápido y uno de 50 cm que cambia despacio
    dan los mismos seis percentiles.
  - Probamos a acotarlo con la distancia del balón lento (< 5 m/s) al jugador más cercano (medida
    9 del diagnóstico). Dentro de un mismo tracker la separa: de 3 a 48 cm, el p1 pasa de 0,09 a
    0,19 m. Pero la convención de cada tracker la mueve mucho más: p50 de 2,6 m en el limpio de
    IDSSE y de 0,1 m en el de Metrica, que pega el balón al pie. Así que no sirve en el real.
  - Train sortea el RMS uniforme entre 0,03 y 0,5 m; val y test usan el centro, 0,265 m.
  - El error se aplica en los frames detectados y en línea recta en los rellenos, como lo haría un
    tracker.
- **Oclusiones, no calibradas**: son pérdidas de menos de 5 s que siempre se rellenan. Su
  proporción no se mide en el real; train la sortea entre 0,05 y 0,3, y val y test usan 0,175.
- **Tramos sin balón con el balón en juego**: duran 5 s o más, con una log-normal por encima.
- **Calibración con la variación por semilla**: cada sorteo de la calibración varía los parámetros
  como una semilla de train, para que la media de train reproduzca el real.
  - La pérdida suma el error en el % sin balón.
  - El exceso de SET PIECES precedidos sobre el azar se informa, pero ya no se ajusta: es ruidoso.
- **Ventanas por tipo**:
  - Tipo de reanudación de los eventos en Metrica (294 de 303 reanudaciones tienen evento a ≤ 2 s).
  - En IDSSE, por posición: córner a < 5 m de un banderín.
  - Faltas: 1 de cada 7. Córners: todos.
- **Desplazamiento de la reanudación**: cada semilla de train sortea los pesos de una Dirichlet
  centrada en 0,11 / 0,72 / 0,17, con 16 observaciones.
- **Diagnóstico**: medidas 8 (segunda diferencia del balón y de los jugadores) y 9 (jugador más
  cercano a un balón lento).
- **`requirements-mtah.txt`**: incluye `kloppy==3.18.0`, que necesita el test de ida y vuelta.

**Por qué.**
- v2 añadía al balón 25 cm de ruido independiente por frame (segunda diferencia mediana de
  0,55 m, ~60 veces la real) y 12 cm a los jugadores (~250 veces).
- Generaba la mitad de los córners del archivo real y el doble de faltas.

**Con qué datos del partido del reto.** Solo agregados sin etiquetas de
`tools/ball_gap_diagnostic.py`, ejecutado por una persona fuera de Claude Code: los de v2, más los
percentiles de la segunda diferencia y las ventanas por tipo.

**Medidas descartadas antes de construir.**
- **Movimiento rectilíneo a velocidad constante** como detector de interpolación: no separa. En
  los bancos solo funcionaba por nuestro ruido de 25 cm. En un tracker suavizado, el 80–99 % de
  un balón en movimiento ya parece recto, y en el real sale el 99,9 %.
- **Balón inmóvil con los jugadores moviéndose**: no se puede validar, porque nuestros bancos no
  congelan el balón.

**Resultado.** El diagnóstico agrega todas las particiones y semillas. smooth50 se calcula en
IDSSE train+val y en val_metrica, con etiquetas armonizadas.

| | IDSSE v1 | IDSSE v2 | IDSSE v2.1 | Metrica v1 | Metrica v2 | Metrica v2.1 | Real |
|---|---|---|---|---|---|---|---|
| sin balón | 38,7 % | 45,8 % | 43,8 % | 47,3 % | 45,3 % | 37,0 % | 41,1 % |
| tramos 5–10 / 10–30 / > 30 s | 2,3 / 18,0 / 18,4 | 8,3 / 16,7 / 20,8 | 7,7 / 16,2 / 19,8 | 4,2 / 19,9 / 23,1 | 9,7 / 17,1 / 18,4 | 8,5 / 15,5 / 13,0 | 6,0 / 16,8 / 18,2 |
| tramos ≥ 10 s por parte | 39,0 | 39,4 | 39,3 | 46,0 | 40,3 | 36,8 | 38,5 |
| exceso sobre el azar | 63,8 % | 49,1 % | 45,4 % | 80,7 % | 46,7 % | 31,2 % | 56,4 % |
| fin a ≤ 3 s del inicio / de una ventana | 30,6 / 38,9 % | 8,4 / 46,2 % | 7,5 / 45,8 % | 32,0 / 37,1 % | 10,5 / 43,9 % | 7,8 / 46,8 % | 10,5 / 51,3 % |
| visto: fuera / banda / fondo / dentro | 14/21/16/49 | 19/10/8/63 | 16/11/7/66 | 10/13/19/58 | 19/6/5/70 | 17/5/1/77 | 18/9/9/64 |
| balón, 2.ª dif. p50 / p90 / p99 / p99,9 (m) | 0,59 / 1,25 / 1,92 / 21,5 | 0,55 / 1,22 / 1,92 / 22,5 | 0,0092 / 0,024 / 0,051 / 0,101 | 0,61 / 1,27 / 1,92 / 22,0 | 0,55 / 1,22 / 1,92 / 21,3 | 0,0094 / 0,026 / 0,050 / 0,083 | 0,0091 / 0,029 / 0,051 / 0,096 |
| jugadores, 2.ª dif. p50 / p90 / p99 (m) | 0,35 / 0,63 / 0,91 | 0,35 / 0,63 / 0,91 | 0,0014 / 0,0041 / 0,0092 | 0,35 / 0,63 / 0,91 | 0,35 / 0,63 / 0,91 | 0,0014 / 0,0040 / 0,0078 | 0,0014 / 0,0043 / 0,0096 |
| balón en tramos rectos | 15,9 % | 16,1 % | 99,9 % | 14,6 % | 16,5 % | 100 % | 99,9 % |
| ventanas cada 90 min: banda / córner / puerta / falta | 23,3 / 4,3 / 7,4 / 9,6 | 23,0 / 4,4 / 7,3 / 8,9 | 21,3 / 8,9 / 8,5 / 5,2 | 18,2 / 2,5 / 10,7 / 7,2 | 21,0 / 2,2 / 9,7 / 9,1 | 18,5 / 8,2 / 10,4 / 5,3 | 22,5 / 10,4 / 8,7 / 4,3 |
| smooth50 spellF1 / macroF1 | 0,268 / 0,632 | 0,219 / 0,636 | 0,240 / 0,626 | 0,328 / 0,759 | 0,246 / 0,674 | 0,262 / 0,609 | 0,243 / 0,67 |

Los saques de centro no cambian: 2,3 / 1,8 en IDSSE, 3,8 / 1,9 en Metrica y 3,5 / 1,7 en el real.

Orden de los baselines (comprobación, no objetivo):
- Con etiquetas armonizadas, v2.1 ordena smooth50 > smooth125 > smooth25 > raw en IDSSE y en
  Metrica, como el partido real. En v2, IDSSE ponía smooth125 por delante.
- Con etiquetas originales, IDSSE ordena smooth50 > smooth25 > smooth125.

**Abierto.**
- **Train de IDSSE queda un poco fuera del criterio de parada**: 43,8 % sin balón y 7,4 % en tramos
  de 5–10 s, frente a 40–43 % y 5–7 %. La calibración esperaba 42,5 % y 6,9 %. La diferencia es
  muestreo: las 8 semillas de train sortean su variación distinta de los 12 sorteos de la
  calibración. Por partición:
  - val: 44,3 % y 8,4 %;
  - test: 42,1 % y 8,5 %.
- **`live_sigma` = 0,25 está en el borde inferior de la rejilla**: los tramos en juego duran casi
  siempre 5 + ~4 s.
- **Metrica: `fill_dead` = 1**, el máximo; se rellenan todas sus paradas sin balón.
  - Queda con menos balón ausente que el real: 38,8 % en val_metrica y 33,2 % en test_metrica.
  - Casi nunca se le pierde el balón junto a la línea de fondo (1 % frente a 9 %).
  - Su exceso sobre el azar es del 31 %.
  - Su smooth50 tiene un macroF1 de 0,61, frente a 0,67 en el real.
- **El suavizado gaussiano centrado y la forma del error del balón son suposiciones.** El tamaño
  del error es lo que más pesa en las señales de cercanía, y no lo conocemos.

**Comprobación.** `python tests/test_challenge_roundtrip.py` pasa sobre v2.1. La exportación
escribe ahora 6 decimales y el error máximo al leerla de vuelta es de 0,05 mm, con una tolerancia
de 0,1 mm.

## 2026-11-04 · Datasets v2: Smart Tagging de 35 s y balón recalibrado (paquete A)

**Qué.** `tools/build_datasets.py` genera `external/datasets_v2/`. v1 (`external/datasets/`,
etiqueta `v1-ball-degradation`) se queda como estaba. `load_match()` y los ficheros no cambian;
`degradation.json` y `ball_degradation.json` ganan campos nuevos.

- **Smart Tagging sintético**: ventanas SET PIECES de 35 s (los saques de centro, 5 s), como en
  el archivo real. La reanudación cae a −2,5–2,5 s del inicio de la ventana con peso 0,11, a
  2,5–7,5 s con 0,72 y a 7,5–12,5 s con 0,17 (uniforme dentro de cada tramo).
- **Balón**: fuera del campo se esconde con probabilidad `p_off` (antes, siempre), y la
  calibración añade dónde se vio el balón por última vez antes de un tramo largo.
- **Metrica**: un 70 % (`fill_dead`) de las paradas en las que su tracking no tiene balón recibe
  una posición estimada: la última vista, llevada en línea recta hasta la reanudación.
- **Calibrado**: `p_off` 0,3, `p_dead` 0,6, `r_live` 6 por minuto vivo, mediana 4 s, sigma 0,55;
  3 degradaciones por partido y ajuste. Ningún parámetro en el borde de la rejilla (el script
  avisa si pasa).

**Por qué.** v1 tenía ventanas de 5 s (las reales duran 35 s), escondía siempre el balón fuera
del campo y Metrica salía demasiado fácil (47 % sin balón, exceso sobre el azar del 81 %,
smooth50 con macroF1 0,76 frente a 0,67 en el partido real).

**Con qué datos del partido del reto.** Solo agregados sin etiquetas de
`tools/ball_gap_diagnostic.py`, ejecutado por una persona fuera de Claude Code (`REAL_BALL_GAPS`):
tramos sin balón por duración, tramos largos por parte, SET PIECES precedidos de un tramo largo
frente al azar, dónde se vio el balón por última vez y el histograma de (fin del tramo − inicio
de la ventana). Nunca su posesión ni sus etiquetas.

**El desplazamiento de la reanudación.** Histograma del diagnóstico en el partido real, menos lo
esperado con las ventanas desplazadas al azar: todo el exceso (~24 de 45 pares) cae entre −2,5 y
12,5 s, con el máximo en 5–7,5 s; más allá de 12,5 s hay menos que al azar. El balón reaparece
en la reanudación, así que el fin del tramo la marca con ~1 s de error. Tramos de 5 s porque los
de 2,5 s tienen entre 1 y 12 casos.

**Resultado** (diagnóstico agregando todas las semillas; smooth50 en IDSSE train+val y
val_metrica, etiquetas armonizadas):

| | IDSSE v1 | IDSSE v2 | Metrica v1 | Metrica v2 | Real |
|---|---|---|---|---|---|
| sin balón | 38,7 % | 45,8 % | 47,3 % | 45,3 % | 41,1 % |
| tramos 5–10 / 10–30 / > 30 s | 2,3 / 18,0 / 18,4 | 8,3 / 16,7 / 20,8 | 4,2 / 19,9 / 23,1 | 9,7 / 17,1 / 18,4 | 6,0 / 16,8 / 18,2 |
| tramos ≥ 10 s por parte | 39,0 | 39,4 | 46,0 | 40,3 | 38,5 |
| SET PIECES precedidos (azar) | 69,8 % (16,5) | 85,8 % (72,0) | 84,8 % (21,3) | 87,2 % (76,1) | 84,2 % (63,8) |
| exceso sobre el azar | 63,8 % | 49,1 % | 80,7 % | 46,7 % | 56,4 % |
| fin a ≤ 3 s del inicio de ventana | 30,6 % | 8,4 % | 32,0 % | 10,5 % | 10,5 % |
| fin a ≤ 3 s de una ventana | 38,9 % | 46,2 % | 37,1 % | 43,9 % | 51,3 % |
| visto: fuera / banda / fondo / dentro | 14/21/16/49 | 19/10/8/63 | 10/13/19/58 | 19/6/5/70 | 18/9/9/64 |
| smooth50 spellF1 | 0,268 | 0,219 | 0,328 | 0,246 | 0,243 |
| smooth50 macroF1 | 0,632 | 0,636 | 0,759 | 0,674 | 0,67 |

**Abierto.**
- `r_live` y la mediana de los tramos en juego no están identificados: los tramos de menos de
  5 s se interpolan, así que «muchos tramos muy cortos» y «menos tramos algo más largos» dan
  pérdidas parecidas (0,16–0,23 a lo largo de la diagonal, con ruido de Monte Carlo de ~0,04).
  El mínimo elegido produce demasiados tramos de 5–10 s (8 % frente al 6 % real).
- Con etiquetas armonizadas, IDSSE v2 ordena smooth125 (0,231) por encima de smooth50 (0,220);
  el partido real, al revés. Metrica v2 mantiene el orden real. Es una comprobación, no un
  objetivo, y la diferencia está dentro del ruido entre partidos; pero conviene revisarlo si se
  reparametrizan los tramos en juego.
- El exceso sobre el azar se mide con ~50 ventanas por partido: va del ~20 % al ~70 % entre
  semillas, y el 56 % del real tiene la misma incertidumbre.

**Arreglos por el camino.**
- Las semillas de la degradación salían de los 4 primeros bytes del id del partido: todos los de
  IDSSE («J03W») y todos los de Metrica («METR») compartían sorteos y, en train, la misma variación
  de parámetros por semilla. v2 usa un CRC32 del id completo; v1 conserva el fallo.
- La caché de smooth50 de `train_classifier.py` no distinguía la carpeta del dataset: con v2 y la
  carpeta de salida por defecto habría devuelto los números de v1.

**Comprobación.** `python tests/test_challenge_roundtrip.py`: un partido del banco escrito en el
formato del reto se lee con el starter oficial (kloppy y `load_smart_tagging`) y con
`heuristic_pipeline.load_tracking` exactamente como lo guarda el npz (frames, coordenadas, balón
ausente, slots, ventanas).
