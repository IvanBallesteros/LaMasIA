# Decisiones

Una entrada por decisión: qué se decidió, por qué, con qué datos y qué queda abierto. Las más
recientes, arriba.

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
