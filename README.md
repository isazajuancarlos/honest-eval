# honest-eval

Dos primitivas, en Python puro y sin dependencias, para **no mentirte** al
evaluar un experimento sobre datos temporales:

- **`temporal_split`** — separa train/test dejando el tramo más reciente como
  test, con embargo, para no filtrar el futuro (*leakage*).
- **`select_best_variant`** — decide si una variante le gana **de verdad** a un
  baseline, o si fue suerte: comparación apareada, tamaño de muestra efectivo de
  Kish y cota inferior de confianza.

Nacieron dentro de un bot de trading, para decidir con evidencia qué estrategia
desplegar. Pero el rigor no tiene nada de específico al trading: sirve para
cualquier A/B sobre muestras pareadas y cualquier validación de un modelo
temporal.

## Instalación

```bash
pip install honest-eval
```

Sin dependencias. Requiere Python ≥ 3.9.

## 1. `temporal_split` — el test es el futuro, no una muestra al azar

En una serie temporal, `sklearn.train_test_split` mete muestras de mañana en el
train y la métrica sale **inflada**: el modelo "predice" cosas que en producción
aún no habrían pasado. El test honesto es siempre el tramo más reciente.

```python
from honest_eval import temporal_split

train_idx, test_idx = temporal_split(timestamps, test_frac=0.20, embargo=24)
X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
```

Devuelve **índices** (no copia tus datos), así lo aplicas a numpy, pandas o
listas por igual.

**El embargo.** Si tu label mira `h` pasos adelante, una muestra de train a
menos de `h` del corte ya "conoce" parte del resultado del test. `embargo=h`
descarta esas muestras del borde. La métrica baja, pero deja de mentir.

## 2. `select_best_variant` — ¿ganó, o tuvo suerte?

Elegir la variante de mayor media **premia la varianza, no la ventaja**: con
pocas muestras, la más ruidosa suele quedar arriba por azar. Dos correcciones:

- **Aparear.** Mide variante y baseline sobre el *mismo* ensayo y trabaja con
  `δ = variante − baseline`. La varianza común del ensayo se cancela en la resta.
- **Exigir cota inferior > 0.** Gradúa solo si `mean − z·SE > 0`: "incluso siendo
  pesimista dentro del margen de confianza, sigue por encima del baseline".

```python
from honest_eval import select_best_variant

# Por cada variante, sus deltas (variante − baseline) apareados por ensayo:
variants = {
    "chandelier": [0.8, 1.1, -0.2, 0.9, 1.0, 0.7],
    "momentum":   [2.0, -1.5, 3.0, -0.5, 1.2, -1.0],   # media alta, muy dispersa
}

elegido = select_best_variant(variants, z=1.6449, min_effective_n=5)
if elegido is None:
    politica = "baseline"          # nadie superó al baseline con significancia
else:
    politica = elegido.name        # p.ej. "chandelier": mayor LCB, no mayor media
```

`momentum` puede tener media más alta y aun así **no graduar**: su dispersión
hunde la cota inferior. Eso es exactamente lo que quieres que pase.

### Ponderación por recencia

Si el proceso cambia con el tiempo, pesa lo reciente más que lo viejo:

```python
from honest_eval import halflife_weight

weights = [halflife_weight(now - t, halflife=7*86400) for t in exit_ts]
variants = {"chandelier": (deltas, weights)}
```

El **tamaño de muestra efectivo de Kish** `(Σw)² / Σw²` se encarga de que unas
pocas muestras muy pesadas no se hagan pasar por muchas: con pesos desiguales,
`n_eff < n`, y el gate lo tiene en cuenta.

### La estadística cruda

Si solo quieres los números de una comparación apareada:

```python
from honest_eval import paired_lcb

r = paired_lcb(deltas, weights=weights, z=1.6449)
r.mean, r.lcb, r.se, r.n_eff
```

## Por qué existe

Casi todo el "edge" que se mide en un backtest es una de estas dos ilusiones: el
modelo vio el futuro, o la variante ganadora ganó por ruido. Estas dos funciones
son el mínimo para descartar ambas antes de arriesgar nada real. No crean señal
—eso son datos y features— pero dejan de fabricarla donde no la hay.

## Tests

```bash
pip install "honest-eval[test]"
pytest
```

23 tests, con los valores esperados calculados a mano para anclar la matemática,
y verificados por mutación (alterar el signo de la cota o el corte del split
rompe los tests que deben romperse).

## Licencia

Apache-2.0.
