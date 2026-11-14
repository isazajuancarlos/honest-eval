# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Juan Carlos Isaza

"""honest-eval — dos primitivas para no mentirte al evaluar un experimento
sobre datos temporales.

    temporal_split(timestamps, ...)   Separa train/test dejando el tramo MÁS
                                      RECIENTE como test, con embargo, para no
                                      filtrar el futuro (leakage) en una serie
                                      temporal.

    select_best_variant(variants)     ¿Una variante le gana de verdad a un
                                      baseline, o fue suerte? Comparación
                                      APAREADA (la varianza común se cancela),
                                      con tamaño de muestra efectivo (Kish) y una
                                      cota inferior de confianza: gradúa solo si
                                      la ventaja es significativa.

Sin dependencias: Python puro. Nacieron dentro de un bot de trading para decidir
con evidencia qué estrategia desplegar, y se extrajeron porque el rigor no tiene
nada de específico al trading — sirve para cualquier A/B sobre muestras pareadas
o cualquier validación de un modelo temporal.

El sesgo que ataca cada una:

- **temporal_split** contra el leakage del split aleatorio: en una serie
  temporal, `train_test_split` mete velas de mañana en el train y la métrica
  sale inflada. El test honesto es siempre el futuro, nunca una muestra al azar.

- **select_best_variant** contra el "ganó por suerte": elegir la variante de
  mayor media premia la varianza, no la ventaja. Aparear (medir variante y
  baseline sobre el MISMO ensayo) cancela el ruido común, y exigir cota inferior
  > 0 rechaza a los ganadores de alta varianza y poca muestra.
"""
from __future__ import annotations

from collections import namedtuple

__all__ = [
    "temporal_split",
    "paired_lcb",
    "select_best_variant",
    "halflife_weight",
    "PairedResult",
    "VariantChoice",
    "Z_95_ONE_SIDED",
    "Z_90_ONE_SIDED",
    "Z_99_ONE_SIDED",
    "__version__",
]

__version__ = "0.1.1"

# z de una cola para la cota inferior de confianza (LCB = mean − z·SE).
Z_90_ONE_SIDED = 1.2816   # 90%
Z_95_ONE_SIDED = 1.6449   # 95% (default)
Z_99_ONE_SIDED = 2.3263   # 99%


# ---------------------------------------------------------------------------
# 1. Split temporal sin fuga de futuro
# ---------------------------------------------------------------------------
def temporal_split(timestamps, test_frac: float = 0.20, embargo: int = 0):
    """Devuelve ``(train_idx, test_idx)`` — dos listas de índices sobre
    ``timestamps`` — separando como TEST el tramo más reciente.

    En una serie temporal, un split aleatorio filtra el futuro (deja muestras
    posteriores en el train) e infla la métrica. Aquí el test es siempre el
    ``test_frac`` final en el tiempo, y el train es lo anterior.

    ``embargo`` descarta del FINAL del train las últimas ``embargo`` muestras,
    las que se solapan con el horizonte del label del test. Si el label mira
    ``h`` pasos adelante, una muestra de train a menos de ``h`` del corte
    conoce en parte el resultado del test: ponle ``embargo=h`` para cortar ese
    puente. La métrica honesta sale más baja que la del split aleatorio, pero es
    la real out-of-sample.

    No ordena en sitio ni copia tus datos: solo devuelve índices, así aplicas el
    split a X, y, o lo que sea (numpy, pandas, listas) tú mismo::

        train_idx, test_idx = temporal_split(ts, test_frac=0.2, embargo=24)
        X_tr, X_te = X[train_idx], X[test_idx]

    - ``timestamps``: secuencia ordenable (epoch, datetime, int...). No hace
      falta que venga ordenada; se ordena de forma estable internamente.
    - Empates en el timestamp: el orden estable preserva el orden de entrada.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac debe estar en (0, 1)")
    if embargo < 0:
        raise ValueError("embargo no puede ser negativo")

    n = len(timestamps)
    if n == 0:
        return [], []

    # argsort estable: ante empates conserva el orden original.
    order = sorted(range(n), key=lambda i: timestamps[i])
    corte = int(n * (1 - test_frac))
    tr_fin = max(0, corte - embargo)

    train_idx = order[:tr_fin]
    test_idx = order[corte:]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# 2. Gate de significancia apareado
# ---------------------------------------------------------------------------
PairedResult = namedtuple("PairedResult", ["mean", "lcb", "se", "n_eff"])


def halflife_weight(age, halflife) -> float:
    """Peso por recencia: ``0.5 ** (age / halflife)``. Con ``halflife`` <= 0 o
    None devuelve 1.0 (sin ponderar). ``age`` es la antigüedad de la muestra en
    las mismas unidades que ``halflife`` (segundos, días...). Lo viejo pesa menos
    → la decisión se adapta cuando el proceso cambia."""
    if not halflife or halflife <= 0:
        return 1.0
    return 0.5 ** (age / halflife)


def paired_lcb(deltas, weights=None, z: float = Z_95_ONE_SIDED) -> PairedResult:
    """Estadística de una comparación apareada. ``deltas[i]`` = (valor de la
    variante − valor del baseline) medidos sobre el MISMO ensayo ``i``; al
    restar, la varianza común del ensayo se cancela.

    Devuelve ``PairedResult(mean, lcb, se, n_eff)`` con:

    - ``mean``  media (ponderada) de los deltas.
    - ``n_eff`` tamaño de muestra EFECTIVO de Kish ``(Σw)² / Σw²``: con pesos
      iguales es ``len(deltas)``; con pesos desiguales es menor (unas pocas
      muestras dominan → menos información real).
    - ``se``    error estándar ``sqrt(var / n_eff)``, con ``var`` la varianza
      ponderada de los deltas.
    - ``lcb``   cota inferior de confianza ``mean − z·se``. Si ``se`` == 0 (sin
      dispersión) entonces ``lcb`` = ``mean``.

    La regla de decisión vive en :func:`select_best_variant`: graduar solo si
    ``lcb > 0`` significa "incluso siendo pesimista dentro del margen de
    confianza, la variante sigue por encima del baseline".

    Requiere al menos 2 deltas (si no, ValueError: no hay dispersión estimable).
    """
    deltas = list(deltas)
    if len(deltas) < 2:
        raise ValueError("se necesitan al menos 2 deltas apareados")
    if weights is None:
        weights = [1.0] * len(deltas)
    else:
        weights = list(weights)
        if len(weights) != len(deltas):
            raise ValueError("weights y deltas deben tener la misma longitud")

    sw = sum(weights)
    if sw <= 0:
        raise ValueError("la suma de pesos debe ser positiva")

    n_eff = (sw * sw) / sum(w * w for w in weights)
    mean = sum(w * d for w, d in zip(weights, deltas)) / sw
    var = sum(w * (d - mean) ** 2 for w, d in zip(weights, deltas)) / sw
    se = (var / n_eff) ** 0.5
    lcb = mean - z * se if se > 0 else mean
    return PairedResult(mean=mean, lcb=lcb, se=se, n_eff=n_eff)


VariantChoice = namedtuple("VariantChoice", ["name", "lcb", "result"])


def select_best_variant(variants, z: float = Z_95_ONE_SIDED,
                        min_effective_n: float = 2.0):
    """Elige la variante que le gana al baseline con significancia, o None.

    ``variants``: dict ``{nombre: deltas}`` o ``{nombre: (deltas, weights)}``,
    donde cada ``deltas`` son los ``variante − baseline`` apareados de esa
    variante. (El baseline no va como variante: es el 0 de referencia, ya
    restado en cada delta.)

    Para cada variante calcula :func:`paired_lcb` y descarta las que tengan
    ``n_eff < min_effective_n`` o ``lcb <= 0``. Entre las que quedan, devuelve la
    de mayor ``lcb`` como ``VariantChoice(name, lcb, result)``. Si ninguna
    califica, devuelve ``None`` — la señal de "quédate con el baseline".

    Elegir por LCB y no por media es lo que rechaza al ganador de suerte: una
    variante con media alta pero muy dispersa, o con poca muestra efectiva, tiene
    LCB bajo o negativo y no gradúa.
    """
    mejor = None
    for name, spec in variants.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            deltas, weights = spec
        else:
            deltas, weights = spec, None
        if len(list(deltas)) < 2:
            continue
        res = paired_lcb(deltas, weights=weights, z=z)
        if res.n_eff < min_effective_n:
            continue
        if res.lcb <= 0:
            continue
        if mejor is None or res.lcb > mejor.lcb:
            mejor = VariantChoice(name=name, lcb=res.lcb, result=res)
    return mejor
