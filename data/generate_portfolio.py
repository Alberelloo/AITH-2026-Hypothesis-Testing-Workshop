"""
Генератор портфеля из 6 тестов для воркшопа (встреча 6: мета-анализ, forest plot, I²).

Одна строка — один завершённый A/B-тест из портфеля команды за квартал.
Эффект и его стандартная ошибка даны в процентных пунктах (п.п.) конверсии,
поэтому тесты можно честно пулить в один forest plot.

Колонки portfolio_tests.csv:
    test_id      идентификатор теста (HD-101, CHK-203, ...)
    name         короткое название изменения
    metric       метрика (все — конверсия, п.п., чтобы пулинг был осмысленным)
    n            размер каждой группы, пользователей
    base_rate    базовая конверсия в группе A (доля, не проценты)
    effect       оценка эффекта, п.п.
    se           стандартная ошибка эффекта, п.п.
    ci_low       нижняя граница 95% ДИ, п.п.
    ci_high      верхняя граница 95% ДИ, п.п.
    p_value      двусторонний p-value (считается из effect и se)

Состав портфеля (это и есть материал для обсуждения):
    * 3 правдоподобных «слабых» теста с эффектом порядка MDE;
    * 1 нулевой тест (эффект ≈ 0) — не значим, и это нормальный результат;
    * 1 гетерогенный выброс (эффект в 4–6 раз больше остальных) — из-за него I²
      улетает вверх, и пулить портфель «в среднем» нельзя;
    * 1 тест с отрицательным эффектом (похоже на вред) — тоже не значим,
      но требует решения, а не «залистать».

se выводится из размера выборки и базовой конверсии:
    se = 100 * sqrt(2 * p0 * (1 - p0) / n)
так что участники могут пересчитать её сами и проверить руками.

Запуск: python generate_portfolio.py  (перезаписывает portfolio_tests.csv рядом со скриптом)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7301                      # фиксированный seed: jitter воспроизводим
Z95 = 1.959963984540054          # квантиль нормального распределения для 95% ДИ

# ---------------------------------------------------------------------------
# Параметры портфеля: (test_id, name, n на группу, базовая конверсия, истинный эффект п.п.)
# ---------------------------------------------------------------------------
TRUE_TESTS = [
    ("HD-101", "Блок рекомендаций на главной",      52_000, 0.0500,  0.32),
    ("PRM-104", "Промо-баннер в выдаче",            38_000, 0.0480,  0.28),
    ("CHK-203", "Новый чекаут v2",                  78_000, 0.0520,  0.19),
    ("PUSH-307", "Другое время push-рассылки",      64_000, 0.0500,  0.06),
    ("ONB-112", "Мобильный онбординг",              30_000, 0.0470,  1.20),
    ("FILT-408", "Фильтр по цене в поиске",         44_000, 0.0510, -0.22),
]

# Шум оценки эффекта: se выводится из n и base_rate, jitter — только чтобы
# значения не были «слишком круглыми» (не влияет на порядок величин).
JITTER_SD = 0.03                 # 3% от se


def _p_value(effect, se):
    """Двусторонний p-value без scipy: 2 * P(Z > |z|)."""
    z = abs(effect / se)
    return math.erfc(z / math.sqrt(2.0))


def build(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for test_id, name, n, p0, effect in TRUE_TESTS:
        se = 100 * math.sqrt(2 * p0 * (1 - p0) / n)
        est = effect * (1 + rng.normal(0, JITTER_SD))
        half = Z95 * se
        rows.append({
            "test_id": test_id,
            "name": name,
            "metric": "конверсия, п.п.",
            "n": n,
            "base_rate": p0,
            "effect": round(est, 3),
            "se": round(se, 3),
            "ci_low": round(est - half, 3),
            "ci_high": round(est + half, 3),
            "p_value": round(_p_value(est, se), 4),
        })
    return pd.DataFrame(rows)


def heterogeneity(df):
    """Q, df, I² для фиксированной модели (df = 5, поэтому χ²-хвост считается точно)."""
    w = 1 / df["se"].to_numpy() ** 2
    e = df["effect"].to_numpy()
    pooled = (w * e).sum() / w.sum()
    q = (w * (e - pooled) ** 2).sum()
    dfree = len(df) - 1
    i2 = max(0.0, (q - dfree) / q) if q > 0 else 0.0
    # P(χ²_5 > q) — точная формула для нечётного df = 5
    p_q = math.erfc(math.sqrt(q / 2)) + math.sqrt(2 * q / math.pi) * math.exp(-q / 2)
    return pooled, q, dfree, i2, p_q


if __name__ == "__main__":
    out = Path(__file__).with_name("portfolio_tests.csv")
    data = build()
    data.to_csv(out, index=False)
    print(f"{out.name}: {len(data)} строк, {data.shape[1]} колонок")
    with pd.option_context("display.width", 200):
        print(data.to_string(index=False))
    pooled, q, dfree, i2, p_q = heterogeneity(data)
    print(f"\nфиксированная модель: эффект {pooled:+.3f} п.п.; "
          f"Q = {q:.2f}, df = {dfree}, p_Q = {p_q:.2e}, I2 = {100 * i2:.1f}%")
    print(f"значимых тестов (p<0.05): {(data.p_value < 0.05).sum()} из {len(data)}")
