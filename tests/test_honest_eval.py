# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Juan Carlos Isaza

"""Tests de honest-eval. Sin dependencias. Los valores esperados del gate
están calculados a mano en los comentarios para anclar la matemática — un test
que pasa 'porque sí' vale menos que no tenerlo."""
import math

import pytest

import honest_eval as he


# ===========================================================================
# temporal_split
# ===========================================================================
def test_test_es_el_tramo_mas_reciente():
    ts = [10, 20, 30, 40, 50]           # ya ordenado
    tr, te = he.temporal_split(ts, test_frac=0.20)   # corte = int(5*0.8)=4
    assert tr == [0, 1, 2, 3]
    assert te == [4]                     # el más reciente


def test_desordenado_se_ordena_por_tiempo():
    ts = [50, 10, 40, 20, 30]           # índices por tiempo: 1,3,4,2,0
    tr, te = he.temporal_split(ts, test_frac=0.20)   # corte=4
    assert tr == [1, 3, 4, 2]            # los 4 más antiguos, en orden temporal
    assert te == [0]                     # ts=50, el más reciente (índice original 0)


def test_embargo_recorta_el_final_del_train():
    ts = list(range(10))                 # corte = int(10*0.8) = 8
    tr, te = he.temporal_split(ts, test_frac=0.20, embargo=3)
    # tr_fin = 8 - 3 = 5  → train [0..4], test [8,9]; el hueco 5,6,7 es el embargo
    assert tr == [0, 1, 2, 3, 4]
    assert te == [8, 9]


def test_embargo_mayor_que_corte_deja_train_vacio():
    ts = list(range(10))
    tr, te = he.temporal_split(ts, test_frac=0.20, embargo=100)
    assert tr == []                      # max(0, 8-100)=0
    assert te == [8, 9]


def test_empates_preservan_orden_de_entrada():
    ts = [5, 5, 5, 5]                    # todos iguales → orden estable = 0,1,2,3
    tr, te = he.temporal_split(ts, test_frac=0.25)   # corte=3
    assert tr == [0, 1, 2]
    assert te == [3]


def test_vacio():
    assert he.temporal_split([], test_frac=0.2) == ([], [])


def test_test_frac_invalido():
    with pytest.raises(ValueError):
        he.temporal_split([1, 2, 3], test_frac=0.0)
    with pytest.raises(ValueError):
        he.temporal_split([1, 2, 3], test_frac=1.0)


def test_embargo_negativo():
    with pytest.raises(ValueError):
        he.temporal_split([1, 2, 3], embargo=-1)


# ===========================================================================
# paired_lcb — la matemática, con números a mano
# ===========================================================================
def test_lcb_sin_pesos():
    # deltas=[1,2,3]: mean=2; var=((1)+(0)+(1))/3=0.6667; n_eff=3;
    # se=sqrt(0.6667/3)=0.4714; lcb=2 - 1.6449*0.4714 = 1.2246
    r = he.paired_lcb([1.0, 2.0, 3.0])
    assert math.isclose(r.mean, 2.0)
    assert math.isclose(r.n_eff, 3.0)
    assert math.isclose(r.se, math.sqrt(2 / 3 / 3), rel_tol=1e-9)
    assert math.isclose(r.lcb, 2.0 - he.Z_95_ONE_SIDED * r.se, rel_tol=1e-12)
    assert 1.2245 < r.lcb < 1.2247


def test_kish_n_eff_pesos_iguales_es_n():
    r = he.paired_lcb([1.0, 1.0, 1.0, 1.0], weights=[2.0, 2.0, 2.0, 2.0])
    assert math.isclose(r.n_eff, 4.0)    # pesos iguales → n_eff = n


def test_kish_n_eff_pesos_desiguales_menor_que_n():
    # weights=[1,1,2]: Σw=4, Σw²=6, n_eff=16/6=2.6667 < 3
    r = he.paired_lcb([1.0, 1.0, 1.0], weights=[1.0, 1.0, 2.0])
    assert math.isclose(r.n_eff, 16 / 6, rel_tol=1e-9)


def test_se_cero_lcb_es_la_media():
    # sin dispersión: todos los deltas iguales → var=0 → se=0 → lcb=mean
    r = he.paired_lcb([0.5, 0.5, 0.5])
    assert r.se == 0.0
    assert r.lcb == 0.5


def test_pesos_ponderan_la_media():
    # deltas=[0,10], weights=[3,1]: mean = (3*0+1*10)/4 = 2.5
    r = he.paired_lcb([0.0, 10.0], weights=[3.0, 1.0])
    assert math.isclose(r.mean, 2.5)


def test_menos_de_dos_deltas_falla():
    with pytest.raises(ValueError):
        he.paired_lcb([1.0])


def test_longitudes_desiguales_fallan():
    with pytest.raises(ValueError):
        he.paired_lcb([1.0, 2.0], weights=[1.0])


def test_suma_pesos_no_positiva_falla():
    with pytest.raises(ValueError):
        he.paired_lcb([1.0, 2.0], weights=[0.0, 0.0])


# ===========================================================================
# select_best_variant — la regla de decisión
# ===========================================================================
def test_elige_la_de_mayor_lcb():
    # A: deltas fuertes y consistentes → LCB alto
    # B: media parecida pero dispersa → LCB más bajo
    variants = {
        "A": [1.0, 1.1, 0.9, 1.0, 1.0],
        "B": [3.0, -1.0, 2.0, -0.5, 1.0],
    }
    elegido = he.select_best_variant(variants)
    assert elegido.name == "A"


def test_rechaza_lcb_no_positivo():
    # media positiva pero tan dispersa que LCB <= 0 → nadie gradúa → None
    variants = {"ruidosa": [5.0, -5.0, 5.0, -5.0]}
    assert he.select_best_variant(variants) is None


def test_rechaza_por_muestra_efectiva_insuficiente():
    # LCB positivo, pero exigimos n_eff >= 10 y solo hay 3 → descartada
    variants = {"pocas": [1.0, 1.0, 1.0]}
    assert he.select_best_variant(variants, min_effective_n=10) is None


def test_variante_con_pesos():
    variants = {"A": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])}
    elegido = he.select_best_variant(variants)
    assert elegido.name == "A"
    assert elegido.lcb == 1.0            # sin dispersión → lcb = mean = 1.0


def test_nadie_califica_devuelve_none():
    variants = {"neg": [-1.0, -1.0, -1.0]}
    assert he.select_best_variant(variants) is None


def test_variante_con_un_solo_delta_se_ignora():
    variants = {"corta": [1.0], "buena": [1.0, 1.0, 1.0]}
    elegido = he.select_best_variant(variants)
    assert elegido.name == "buena"


def test_recency_halflife():
    assert he.halflife_weight(0, 100) == 1.0        # recién → peso pleno
    assert math.isclose(he.halflife_weight(100, 100), 0.5)   # una vida media
    assert he.halflife_weight(10, 0) == 1.0         # sin halflife → sin ponderar
    assert he.halflife_weight(10, None) == 1.0
