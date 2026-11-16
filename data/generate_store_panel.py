"""
Генератор синтетической панели «магазин × неделя» для воркшопа (встреча 5).

Панель нужна, чтобы показать, где обычный A/B ломается: единица рандомизации —
магазин, наблюдение — магазино-неделя, а внутри магазина недели сильно коррелированы.
На этих данных участники считают ICC и дизайн-эффект, наивный DiD, DiD по парам,
плацебо по датам и мэтчинг контроля.

Колонки store_panel.csv:
    store    номер магазина (1..N_STORES)
    region   регион (6 значений)
    format   формат магазина: a / b / c
    week     понедельник недели, YYYY-MM-DD
    sales    выручка за неделю, денежные единицы
    traffic  число покупателей за неделю
    promo    была ли на неделе маркетинговая акция (0/1)

Служебный файл store_panel_truth.csv (только для преподавателя, участникам не выдавать):
    store, region, format, treated, treat_week, true_traffic_lift, true_check_lift, true_sales_lift

Как устроено смещение (это и есть урок встречи 5):
    * лечение досталось не случайным магазинам, а выбранным по формату и региону
      (крупные магазины формата «a» — их же чаще и выбирают менеджеры);
    * у леченых магазинов зашит более быстрый тренд (TREATMENT["growth_boost"]),
      поэтому параллельные тренды нарушены и наивный DiD завышает эффект;
    * плацебо по датам (то же окно до старта) показывает ненулевой «эффект»;
    * мэтчинг по предпериоду уровень выравнивает, но тренд не лечит.

Истинные эффекты заданы в TRUE_EFFECTS. Не подсматривайте до разбора.

Запуск: python generate_store_panel.py  (перезаписывает оба CSV рядом со скриптом)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Параметры генерации
# ---------------------------------------------------------------------------
SEED = 4411
N_STORES = 300
START_WEEK = "2024-01-01"            # понедельник
N_WEEKS = 78                         # 1,5 года
N_TREATED = 60
TREAT_WEEK_INDEX = 40                # старт пилота: 40-я неделя панели (окт. 2024)
TREAT_WEEKS = 12                     # длительность пилота

REGIONS = {
    "ЦФО": 1.20,
    "СЗФО": 1.10,
    "ПФО": 1.00,
    "УФО": 0.95,
    "СФО": 0.90,
    "ЮФО": 0.85,
}
FORMATS = {"a": 1.15, "b": 1.00, "c": 0.85}   # формат «a» — крупные магазины у дома

# Сезонность и акции
ANNUAL_AMPLITUDE = 0.10              # годичная волна по трафику, ±10%
ANNUAL_PEAK_WEEK = 50                # неделя пика (декабрь)
HOLIDAY_LIFT = {                     # наценка поверх сезонности, по номеру недели в году
    52: 0.25,   # новогодняя неделя
    1: 0.18,    # первая неделя января
    9: 0.06,    # 8 марта
    47: 0.05,   # ноябрьские праздники
}
PROMO_SHARE = 0.15                   # доля магазино-недель с акцией
PROMO_TRAFFIC_LIFT = 0.07            # акция добавляет трафик, но не чек

BASE_SALES_MU, BASE_SALES_SIGMA = 13.90, 0.35   # недельная выручка, медиана ≈ 1,08 млн
BASE_CHECK_MU, BASE_CHECK_SIGMA = 7.00, 0.15    # средний чек, медиана ≈ 1 100
WEEK_NOISE = 0.035                   # недельный шум выручки
CHECK_NOISE = 0.020
GROWTH_WEEKLY = 0.0008               # базовый недельный тренд

# ---------------------------------------------------------------------------
# Истинные эффекты (то, что участники должны восстановить)
# ---------------------------------------------------------------------------
TRUE_EFFECTS = {
    "traffic_lift": 0.015,           # +1,5% покупателей от бьюти-зоны
    "check_lift": 0.020,             # +2,0% среднего чека
    "sales_lift": (1.015 * 1.020) - 1.0,   # ≈ +3,53% выручки (произведение)
    "growth_boost": 0.0016,          # леченые растут быстрее: +0.16 п.п. в неделю (нарушение тренда)
    "selection_by_format": {"a": 0.55, "b": 0.30, "c": 0.15},  # вероятность попасть в лечение
}


def holiday_lift(week):
    """Праздничная надбавка по номеру ISO-недели."""
    return HOLIDAY_LIFT.get(int(week.isocalendar().week), 0.0)


def build():
    rng = np.random.default_rng(SEED)

    stores = np.arange(1, N_STORES + 1)
    region = rng.choice(list(REGIONS), size=N_STORES,
                        p=np.array([0.30, 0.15, 0.25, 0.10, 0.12, 0.08]))
    fmt = rng.choice(list(FORMATS), size=N_STORES, p=[0.45, 0.35, 0.20])

    # Неслучайный отбор: вероятность лечения зависит от формата (крупные — чаще).
    # Взвешенная выборка без возвращения даёт ровно N_TREATED магазинов.
    probs = np.array([TRUE_EFFECTS["selection_by_format"][f] for f in fmt])
    probs = probs / probs.sum()
    treated = np.zeros(N_STORES, dtype=int)
    treated[rng.choice(N_STORES, size=N_TREATED, replace=False, p=probs)] = 1

    level = rng.lognormal(BASE_SALES_MU, BASE_SALES_SIGMA, N_STORES) * \
        np.array([REGIONS[r] for r in region]) * np.array([FORMATS[f] for f in fmt])
    check_base = rng.lognormal(BASE_CHECK_MU, BASE_CHECK_SIGMA, N_STORES) * \
        np.array([REGIONS[r] for r in region])
    traffic_base = level / check_base

    growth = GROWTH_WEEKLY + TRUE_EFFECTS["growth_boost"] * treated
    week_starts = pd.date_range(START_WEEK, periods=N_WEEKS, freq="7D")
    announce = week_starts[TREAT_WEEK_INDEX]

    frames = []
    for w_i, wk in enumerate(week_starts):
        season_frac = (wk.isocalendar().week - ANNUAL_PEAK_WEEK) / 52.0
        season = 1.0 + ANNUAL_AMPLITUDE * np.cos(2 * np.pi * season_frac) + holiday_lift(wk)
        trend = (1.0 + growth) ** w_i
        promo = (rng.random(N_STORES) < PROMO_SHARE).astype(int)

        traffic = traffic_base * season * trend * (1 + PROMO_TRAFFIC_LIFT * promo)
        traffic *= np.exp(rng.normal(0, WEEK_NOISE, N_STORES))
        check = check_base * (1 + 0.25 * (season - 1)) * (1 + growth) ** (w_i * 0.3)
        check *= np.exp(rng.normal(0, CHECK_NOISE, N_STORES))

        in_window = (TREAT_WEEK_INDEX <= w_i < TREAT_WEEK_INDEX + TREAT_WEEKS)
        if in_window:
            on = treated == 1
            traffic[on] *= 1 + TRUE_EFFECTS["traffic_lift"]
            check[on] *= 1 + TRUE_EFFECTS["check_lift"]

        traffic = np.maximum(np.round(traffic), 1).astype(int)
        frames.append(pd.DataFrame({
            "store": stores,
            "region": region,
            "format": fmt,
            "week": wk.strftime("%Y-%m-%d"),
            "sales": np.round(traffic * check, 2),
            "traffic": traffic,
            "promo": promo,
        }))

    panel = pd.concat(frames, ignore_index=True)
    truth = pd.DataFrame({
        "store": stores,
        "region": region,
        "format": fmt,
        "treated": treated,
        "treat_week": np.where(treated == 1, announce.strftime("%Y-%m-%d"), ""),
        "true_traffic_lift": np.where(treated == 1, TRUE_EFFECTS["traffic_lift"], 0.0),
        "true_check_lift": np.where(treated == 1, TRUE_EFFECTS["check_lift"], 0.0),
        "true_sales_lift": np.where(treated == 1, TRUE_EFFECTS["sales_lift"], 0.0),
    })
    return panel, truth


def _did(panel, treated_ids, t0, weeks=12, pre=12):
    """Наивный DiD в логарифмах: (после-до у леченых) − (после-до у контроля)."""
    df = panel.copy()
    df["y"] = np.log(df["sales"])
    is_t = df["store"].isin(treated_ids)
    post = df["week"].isin([*panel["week"].drop_duplicates()][t0:t0 + weeks])
    prew = df["week"].isin([*panel["week"].drop_duplicates()][max(t0 - pre, 0):t0])
    g = df.assign(t=is_t, after=post)
    g = g[post | prew]
    return (g[g.t & g.after].y.mean() - g[g.t & ~g.after].y.mean()) - \
           (g[~g.t & g.after].y.mean() - g[~g.t & ~g.after].y.mean())


def report(panel, truth):
    treated_ids = truth.loc[truth.treated == 1, "store"]
    weeks = [*panel["week"].drop_duplicates()]
    did_true = _did(panel, treated_ids, TREAT_WEEK_INDEX)
    placebo = _did(panel, treated_ids, TREAT_WEEK_INDEX - 20)
    print(f"строк: {len(panel):,}; магазинов: {panel['store'].nunique()}, недель: {len(weeks)}")
    print(f"средняя недельная выручка: {panel['sales'].mean():,.0f}; трафик: {panel['traffic'].mean():,.0f}")
    print(f"леченых: {int(truth.treated.sum())}; окно: {weeks[TREAT_WEEK_INDEX]} + {TREAT_WEEKS} нед.")
    print(f"истинный эффект: {TRUE_EFFECTS['sales_lift']:+.4f}")
    print(f"наивный DiD (окно пилота): {did_true:+.4f}")
    print(f"DiD по плацебо-окну (до старта): {placebo:+.4f}")
    print(f"форматы леченых: {truth[truth.treated == 1]['format'].value_counts().to_dict()}")


if __name__ == "__main__":
    here = Path(__file__).parent
    panel, truth = build()
    panel.to_csv(here / "store_panel.csv", index=False)
    truth.to_csv(here / "store_panel_truth.csv", index=False)
    print(f"store_panel.csv: {len(panel):,} строк, {panel.shape[1]} колонок")
    print(f"store_panel_truth.csv: {len(truth)} строк")
    report(panel, truth)
