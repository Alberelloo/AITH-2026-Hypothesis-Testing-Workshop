"""
Генератор логов бронирования с сегментами для воркшопа (встреча 6).

База — логика практики 2.П1 (sequential): сервис бронирования отелей тестирует
блок «Горящие предложения», 28 дней, сплит 50/50, эффект затухает по когортам
(новизна), выручка с тяжёлым хвостом, часть броней отменяется.

Одна строка — один пользователь, впервые попавший в тест.

Колонки:
    user_id      идентификатор пользователя
    date         день первого захода в тест
    group        A — без блока, B — с блоком
    booked       сделал ли бронирование (0/1)
    revenue      стоимость бронирования, ₽ (0, если брони нет)
    cancelled    отменил ли бронирование (0/1)
    device       web / mobile
    reg_date     дата регистрации пользователя (до старта теста)
    tenure_days  сколько дней прошло от регистрации до входа в тест
    is_new       tenure_days <= NEWCOMER_DAYS (0/1): «новизна»
    orders_90d   заказов за 90 дней до теста (активность)
    segment      сегмент по активности: low / mid / high

Зачем сегменты: на встрече 6 участники проверяют гетерогенность эффекта без
p-hacking. В данных зашита честная гетерогенность по device и is_new, а сегмент
по активности сделан ПУСТЫМ (истинного различия нет) — любой «значимый» разрез
по segment это ложное срабатывание, которое и надо поймать.

Истинные эффекты заданы в TRUE_EFFECTS. Не подсматривайте до разбора.

Запуск: python generate_segments.py  (перезаписывает hotdeals_segmented.csv рядом со скриптом)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Параметры генерации (база — как в курсе, П4)
# ---------------------------------------------------------------------------
SEED = 2027
START_DATE = "2026-09-07"            # понедельник
N_DAYS = 28
USERS_WEEKDAY, USERS_WEEKEND = 5_600, 7_000

TRUE_PARAMS = {
    "conv_weekday": 0.050,           # базовая конверсия в будни
    "conv_weekend": 0.056,           # базовая конверсия в выходные
    "effect_late": 0.002,            # долгосрочный эффект на конверсию
    "effect_early": 0.010,           # эффект на когорте первого дня
    "effect_decay_days": 4.0,        # скорость затухания эффекта по когортам
    "check_median": 12_000,          # медиана стоимости брони в группе A
    "check_sigma": 1.2,
    "check_ratio_b": 0.95,           # брони в B на 5% дешевле (скидки)
    "cancel_a": 0.18,
    "cancel_b": 0.195,
}

# ---------------------------------------------------------------------------
# Истинные эффекты гетерогенности (множители к эффекту дня)
# ---------------------------------------------------------------------------
NEWCOMER_DAYS = 30                   # новичок: зарегистрирован не позже 30 дней до теста
DEVICE_SHARE_MOBILE = 0.55
EFFECT_BY_DEVICE = {"mobile": 2.00, "web": 0.60}     # блок помогает мобильным (честная разница)
EFFECT_BY_TENURE = {"new": 2.40, "returning": 0.68}  # новичкам блок помогает сильнее (честная)
EFFECT_BY_SEGMENT = {"low": 1.00, "mid": 1.00, "high": 1.00}   # пусто: ловушка для p-hacking

BASE_CONV_SHIFT = {"mobile": -0.006, "web": 0.0}     # мобильные бронируют хуже
CHECK_BY_DEVICE = {"mobile": 0.92, "web": 1.00}      # чек на мобильных ниже
CANCEL_BY_TENURE = 0.02                              # новички чаще отменяют


def effect_by_cohort(day, params=TRUE_PARAMS):
    """Истинный эффект на конверсию для пользователей, пришедших в день day (с нуля)."""
    late, early = params["effect_late"], params["effect_early"]
    return late + (early - late) * np.exp(-day / params["effect_decay_days"])


def segment_of(orders):
    """Сегмент по активности: фиксированные пороги, чтобы границы не «плавали»."""
    return np.where(orders <= 1, "low", np.where(orders <= 4, "mid", "high"))


def build(seed=SEED, params=TRUE_PARAMS):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS)
    frames = []
    next_id = 100_000

    for day, date in enumerate(dates):
        weekend = date.dayofweek >= 5
        n = rng.poisson(USERS_WEEKEND if weekend else USERS_WEEKDAY)
        group = np.where(rng.random(n) < 0.5, "B", "A")
        is_b = group == "B"

        device = np.where(rng.random(n) < DEVICE_SHARE_MOBILE, "mobile", "web")
        is_new = rng.random(n) < 0.30
        tenure = np.where(is_new, rng.integers(0, NEWCOMER_DAYS + 1, n),
                          rng.integers(NEWCOMER_DAYS + 1, 731, n))
        reg_date = (date - pd.to_timedelta(tenure, unit="D")).strftime("%Y-%m-%d")
        orders = rng.poisson(np.where(is_new, 0.8, 3.2))

        seg = segment_of(orders)
        m = (np.vectorize(EFFECT_BY_DEVICE.get)(device)
             * np.where(is_new, EFFECT_BY_TENURE["new"], EFFECT_BY_TENURE["returning"])
             * np.vectorize(EFFECT_BY_SEGMENT.get)(seg))
        m = m / m.mean()             # нормируем: средний эффект остаётся как в курсе

        base_conv = np.where(weekend, params["conv_weekend"], params["conv_weekday"])
        base_conv = base_conv + np.vectorize(BASE_CONV_SHIFT.get)(device)
        booked = rng.random(n) < base_conv + is_b * effect_by_cohort(day, params) * m

        median = params["check_median"] * np.where(is_b, params["check_ratio_b"], 1.0) \
            * np.vectorize(CHECK_BY_DEVICE.get)(device)
        check = rng.lognormal(np.log(median), params["check_sigma"])
        cancel_p = np.where(is_b, params["cancel_b"], params["cancel_a"]) + \
            CANCEL_BY_TENURE * is_new

        frames.append(pd.DataFrame({
            "user_id": np.arange(next_id, next_id + n),
            "date": date.strftime("%Y-%m-%d"),
            "group": group,
            "booked": booked.astype(int),
            "revenue": np.round(check * booked).astype(int),
            "cancelled": (booked & (rng.random(n) < cancel_p)).astype(int),
            "device": device,
            "reg_date": reg_date,
            "tenure_days": tenure,
            "is_new": is_new.astype(int),
            "orders_90d": orders,
            "segment": seg,
        }))
        next_id += n

    return pd.concat(frames, ignore_index=True)


def report(df):
    """Разрезы: честная гетерогенность (device, is_new) и пустой сегмент (ловушка p-hacking)."""
    a, b = df[df.group == "A"], df[df.group == "B"]
    print(f"строк: {len(df):,}; дней: {df['date'].nunique()}; групп: {df['group'].value_counts().to_dict()}")
    print(f"конверсия A/B: {a['booked'].mean():.4f} / {b['booked'].mean():.4f}; "
          f"ARPU: {a['revenue'].mean():.0f} / {b['revenue'].mean():.0f}")
    for col in ("device", "is_new", "segment"):
        print(f"\nэффект по {col} (п.п., z-тест):")
        for val in sorted(df[col].unique()):
            sub = df[df[col] == val]
            sa, sb = sub[sub.group == "A"]["booked"], sub[sub.group == "B"]["booked"]
            lift = sb.mean() - sa.mean()
            se = np.sqrt(sa.var(ddof=1) / len(sa) + sb.var(ddof=1) / len(sb))
            z = lift / se
            p = math.erfc(abs(z) / math.sqrt(2))
            flag = "  <-- значимо при 5%" if p < 0.05 else ""
            print(f"  {val!s:8s} n={len(sub):6d}  {100 * lift:+.2f} п.п.  p={p:.3f}{flag}")


if __name__ == "__main__":
    out = Path(__file__).with_name("hotdeals_segmented.csv")
    data = build()
    data.to_csv(out, index=False)
    print(f"{out.name}: {len(data):,} строк, {data.shape[1]} колонок")
    report(data)
