"""
Генератор сквозного датасета воркшопа: чекаут с предпериодом.

Одна строка — один пользователь, дошедший до чекаута за 14 дней теста
(«новая страница чекаута» против текущей, сплит 80/20).

Колонки:
    user_id            идентификатор пользователя
    date               день первого захода на чекаут (14 дней)
    group              A — старый чекаут, B — новый (20% трафика)
    converted          оформил ли заказ (0/1)
    revenue            выручка с пользователя за период теста, ₽ (0, если заказа нет)
    checkout_time_sec  время на странице чекаута, с
    payment_error      была ли ошибка оплаты (0/1)
    orders_prev28      заказов за 28 дней ДО старта теста (исторический предпериод)
    revenue_prev28     выручка за 28 дней ДО старта теста, ₽

Зачем предпериод: на встрече 2 участники считают CUPED. Ковариата
revenue_prev28 коррелирует с revenue на уровне пользователя (PEARSON_TARGET),
поэтому CUPED честно снижает дисперсию оценки эффекта примерно на ρ² (у нас ≈35%).
Предпериод не зависит от группы — иначе CUPED был бы невалиден.

Как устроена корреляция: у пользователя есть латентная «денежность» w,
которая одновременно задаёт средний чек до и после периода, число заказов
в предпериоде и склонность оформить заказ в тесте. Остальное — независимый шум.

В данные намеренно подложены дефекты логирования (для встречи 3, SRM-детектив):
    * N_DUPLICATES строк записаны в лог дважды;
    * N_BOTH_GROUPS пользователей попали в обе группы (по записи в каждой);
    * у «задвоенных» пользователей предпериод совпадает, а метрики теста — нет.

Истинные эффекты заданы в TRUE_PARAMS. Не подсматривайте до разбора.

Запуск: python generate_checkout.py  (перезаписывает checkout_ab.csv рядом со скриптом)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Параметры генерации
# ---------------------------------------------------------------------------
SEED = 2035                          # фиксированный seed: данные воспроизводимы
SALT = "checkout-v2"                 # соль хеша: при смене соли меняется весь сплит
SHARE_B = 0.20                       # 20% трафика на новую страницу
START_DATE = "2026-09-07"            # понедельник
N_DAYS = 14
USERS_PER_DAY = 7_500

PRE_DAYS = 28                        # длина предпериода (4 недели)
PRE_ORDERS_BASE = 3.0                # среднее число заказов за 28 дней у «среднего» пользователя
PRE_ORDERS_LOAD = 0.30               # вклад «денежности» w в число заказов предпериода
W_LOAD = 0.72                        # вклад w в лог среднего чека (до и после теста)
CONV_LOAD = 0.60                     # вклад w в лог-odds конверсии теста
AOV_SIGMA_TOTAL = 0.80               # полный log-SD чека; остаток после w — независимый шум
AOV_SIGMA_RESID = float(np.sqrt(AOV_SIGMA_TOTAL ** 2 - W_LOAD ** 2))   # ≈0.35

# Истинные эффекты теста: (группа A, группа B)
TRUE_PARAMS = {
    "conv":      (0.700, 0.711),                    # конверсия в заказ: +1 п.п.
    "aov_mu":    (7.80, 7.80),                      # log-среднее суммы заказа: не меняется
    "aov_sigma": AOV_SIGMA_TOTAL,
    "time_mu":   (np.log(95), np.log(95 * 0.98)),   # медиана времени на чекауте: −2% в B
    "time_sigma": 0.60,
    "pay_err":   (0.0200, 0.0225),                  # ошибка оплаты (guardrail): растёт в B
}

N_DUPLICATES = 250                   # строки, записанные в лог дважды
N_BOTH_GROUPS = 60                   # пользователи, попавшие в обе группы


def assign_group(user_id, salt=SALT, share_b=SHARE_B):
    """Детерминированное назначение группы хешем от соли и user_id."""
    h = int(hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest(), 16)
    return "B" if h % 10_000 < share_b * 10_000 else "A"


def _sig(x):
    """Логистическая функция без scipy."""
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_logit(z, target, load):
    """Сдвиг интерцепта, при котором средняя вероятность по выборке равна target.

    Нужен, потому что шум вовлечённости w делает E[sigmoid(logit(p) + load*w)] != p
    (неравенство Йенсена). Без калибровки заявленная конверсия 70% давала бы 68.5%.
    """
    lo, hi = -4.0, 4.0
    base = np.log(target / (1 - target))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sig(base + mid + load * z).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate(seed=SEED):
    rng = np.random.default_rng(seed)

    n_days = rng.poisson(USERS_PER_DAY, size=N_DAYS)
    n = int(n_days.sum())
    dates = np.repeat(pd.date_range(START_DATE, periods=N_DAYS).strftime("%Y-%m-%d"), n_days)
    user_ids = rng.choice(np.arange(1_000_000, 10_000_000), size=n, replace=False)
    groups = np.array([assign_group(u) for u in user_ids])
    is_b = (groups == "B").astype(int)

    # --- Латентная «денежность» пользователя: общий фактор для до и после ---
    w = rng.normal(0.0, 1.0, size=n)

    # --- Предпериод: 28 дней ДО старта. От группы не зависит. -------------
    orders_prev = rng.poisson(PRE_ORDERS_BASE * np.exp(PRE_ORDERS_LOAD * w))
    aov_prev = rng.lognormal(TRUE_PARAMS["aov_mu"][0] + W_LOAD * w, AOV_SIGMA_RESID)
    revenue_prev = np.round(orders_prev * aov_prev, 2)

    # --- Период теста -----------------------------------------------------
    def pick(param):
        a, b = TRUE_PARAMS[param]
        return np.where(is_b == 1, b, a)

    p_a, p_b = TRUE_PARAMS["conv"]
    shift_a = _calibrate_logit(w, p_a, CONV_LOAD)
    shift_b = _calibrate_logit(w, p_b, CONV_LOAD)
    p_conv = _sig(np.log(pick("conv") / (1 - pick("conv"))) + CONV_LOAD * w
                  + np.where(is_b == 1, shift_b, shift_a))
    converted = (rng.random(n) < p_conv).astype(int)

    order_value = rng.lognormal(pick("aov_mu") + W_LOAD * w, AOV_SIGMA_RESID)
    revenue = np.round(converted * order_value, 2)

    checkout_time = np.round(rng.lognormal(pick("time_mu"), TRUE_PARAMS["time_sigma"]), 1)
    payment_error = (rng.random(n) < pick("pay_err")).astype(int)

    df = pd.DataFrame({
        "user_id": user_ids,
        "date": dates,
        "group": groups,
        "converted": converted,
        "revenue": revenue,
        "checkout_time_sec": checkout_time,
        "payment_error": payment_error,
        "orders_prev28": orders_prev,
        "revenue_prev28": revenue_prev,
    })

    # Дефект 1: пользователи в обеих группах. Предпериод у них тот же
    # (это один и тот же человек), метрики теста перезаписаны как «вторая сессия».
    both = df.sample(N_BOTH_GROUPS, random_state=seed).copy()
    both["group"] = np.where(both["group"] == "A", "B", "A")
    both["converted"] = (rng.random(N_BOTH_GROUPS) < 0.7).astype(int)
    both["revenue"] = np.round(both["converted"] * rng.lognormal(7.8, 0.8, N_BOTH_GROUPS), 2)

    # Дефект 2: полные дубли строк (событие записалось дважды).
    dups = df.sample(N_DUPLICATES, random_state=seed + 1)

    df = pd.concat([df, both, dups], ignore_index=True)
    return df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)


def report(df):
    """Контрольные величины: размер, эффекты, корреляция предпериода и выигрыш CUPED."""
    clean = df.drop_duplicates(subset=["user_id"], keep="first")
    corr_all = np.corrcoef(clean["revenue"], clean["revenue_prev28"])[0, 1]
    buyers = clean[clean["revenue"] > 0]
    corr_buyers = np.corrcoef(buyers["revenue"], buyers["revenue_prev28"])[0, 1]

    theta = np.cov(clean["revenue"], clean["revenue_prev28"])[0, 1] / np.var(
        clean["revenue_prev28"], ddof=1)
    adj = clean["revenue"] - theta * (clean["revenue_prev28"] - clean["revenue_prev28"].mean())
    a, b = clean[clean.group == "A"], clean[clean.group == "B"]
    var_naive = a["revenue"].var(ddof=1) / len(a) + b["revenue"].var(ddof=1) / len(b)
    a2, b2 = adj[a.index], adj[b.index]
    var_cuped = a2.var(ddof=1) / len(a) + b2.var(ddof=1) / len(b)

    print(f"строк: {len(df):,}; уникальных user_id: {clean['user_id'].nunique():,}")
    print(f"дублей и повторов: {len(df) - len(clean):,}; групп: {df['group'].value_counts().to_dict()}")
    print(f"конверсия A/B: {a['converted'].mean():.4f} / {b['converted'].mean():.4f}")
    print(f"ARPU A/B: {a['revenue'].mean():.1f} / {b['revenue'].mean():.1f}")
    print(f"ошибка оплаты A/B: {a['payment_error'].mean():.4f} / {b['payment_error'].mean():.4f}")
    print(f"предпериод: заказов {clean['orders_prev28'].mean():.2f}, "
          f"выручка {clean['revenue_prev28'].mean():.0f}")
    print(f"corr(revenue, revenue_prev28) = {corr_all:.3f} (среди покупателей {corr_buyers:.3f})")
    print(f"theta CUPED = {theta:.4f}; снижение дисперсии = {1 - var_cuped / var_naive:.3f}")


if __name__ == "__main__":
    out = Path(__file__).with_name("checkout_ab.csv")
    data = generate()
    data.to_csv(out, index=False)
    print(f"{out.name}: {len(data):,} строк, {data.shape[1]} колонок")
    report(data)
