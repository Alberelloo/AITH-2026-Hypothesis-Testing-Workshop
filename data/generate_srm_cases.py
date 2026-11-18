"""
Генератор данных встречи 3: пять выгрузок для SRM-детектива и тест с потерями телеметрии.

1. srm_case_{1..5}.csv — пять тестов платформы, сплит 80/20, 14 дней. В каждом
   есть SRM, у каждого своя причина. Одна строка — один пользователь в логе.

   Колонки:
       user_id     идентификатор пользователя
       date        день первого события теста в логе
       group       A — контроль (80%), B — вариант (20%)
       device      ios / android / web
       source      organic / ads / push / email — источник визита
       is_bot      1, если антифрод платформы пометил пользователя ботом
       lag_sec     секунд от назначения группы до первого события теста в логе
       converted   оформил ли заказ (0/1)

   Платформа считает тест по строкам с is_bot == 0. Причины SRM лежат в ANSWERS —
   не подсматривайте до разбора.

2. telemetry_ab.csv — тест нового виджета выбора слота доставки, сплит 50/50, 7 дней.
   Одна строка — один назначенный пользователь (все назначенные, без потерь).

   Колонки:
       user_id          идентификатор пользователя
       date             день назначения
       group            A — старый виджет, B — новый
       device           ios / android / web
       triggered        серверный флаг: в корзине есть курьерская доставка, то есть
                        пользователь должен увидеть виджет (одинаково в обеих группах)
       exposure_logged  клиентское событие показа виджета (в A — «показали бы»);
                        часть событий теряется, и потеря в группах разная
       converted        заказ по серверным данным (без потерь)

   Истинные параметры — в TELEMETRY_PARAMS.

Запуск: python generate_srm_cases.py  (перезаписывает CSV рядом со скриптом)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 3033
TELEMETRY_SEED = 3056                # отдельный seed для telemetry_ab.csv
START_DATE = "2026-09-07"
N_DAYS = 14
USERS_PER_DAY = 2_200
SHARE_B = 0.20

DEVICES = (["ios", "android", "web"], [0.35, 0.40, 0.25])
SOURCES = (["organic", "ads", "push", "email"], [0.40, 0.30, 0.15, 0.15])
BOT_RATE = 0.03                       # базовая доля ботов, одинаковая в группах

# Какая причина в каком кейсе. Номера кейсов перемешаны относительно порядка в теории.
CASE_CAUSES = {
    1: "redirect",
    2: "start_date",
    3: "bot_filter",
    4: "assignment_bug",
    5: "telemetry_loss",
}

# Ответы для разбора. Участникам не показывать до раскрытия.
ANSWERS = {
    1: {
        "cause": "redirect",
        "title": "Редирект только в ветке B",
        "mechanism": "B открывается через редирект на /checkout-new; у рекламных ссылок "
                     "редирект срезает параметры, и 45% пользователей из ads не связываются с назначением.",
        "signal": "Доля B проседает только в source = ads (~12% против 20%); у B lag_sec на ~1.5 с больше "
                  "и не бывает меньше 1.5 с — лишний переход.",
        "fix": "Убрать редирект (рендерить вариант на том же URL) или логировать назначение до редиректа.",
    },
    2: {
        "cause": "start_date",
        "title": "Неверная дата старта / кэш",
        "mechanism": "Вариант B реально включился только к середине второго дня: CDN отдавал закэшированную "
                     "старую страницу, и пользователи B в первые дни логировались как A.",
        "signal": "Доля B по дням: ~5% в первый день, ~13% во второй, дальше 20%. По устройствам и "
                  "источникам дефицит равномерный.",
        "fix": "Считать тест с фактической даты включения (с третьего дня); при старте сбрасывать кэш "
               "и проверять долю B в первые часы.",
    },
    3: {
        "cause": "bot_filter",
        "title": "Ботфильтр после назначения",
        "mechanism": "Новый чекаут добавил JS-проверку антифрода; она принимает автозаполнение за бота "
                     "и помечает ботами ещё ~6% живых пользователей B.",
        "signal": "Со всеми строками сплит 80/20 честный; доля is_bot = 1 — ~3% в A и ~9% в B; "
                  "лишние «боты» B — быстрые (маленький lag_sec), с обычных устройств и источников.",
        "fix": "Фильтровать ботов по признакам до назначения (или одинаковым правилом в обеих группах) "
               "и проверять SRM до и после фильтра.",
    },
    4: {
        "cause": "assignment_bug",
        "title": "Баг назначения на iOS",
        "mechanism": "iOS SDK при таймауте конфига падает в контроль: 55% пользователей B на iOS "
                     "получают и логируют A.",
        "signal": "Доля B на iOS ~9%, на android и web — 20%. A при этом больше ожидаемого "
                  "(пользователи перетекли, а не потерялись).",
        "fix": "Чинить SDK: при ошибке исключать пользователя из теста, а не класть в контроль. "
               "Сверять группу в логе с пересчитанным хешем.",
    },
    5: {
        "cause": "telemetry_loss",
        "title": "Потеря телеметрии в ветке B",
        "mechanism": "Новый чекаут отправляет события пачкой при следующем запуске; пайплайн отбрасывает "
                     "события, опоздавшие больше чем на час. В B теряется ~6.5% пользователей.",
        "signal": "У A lag_sec — десятки секунд, у B — минуты и обрыв ровно на 3600 с. "
                  "Дефицит B равномерный по дням, устройствам и источникам.",
        "fix": "Отправлять событие назначения/экспозиции сразу с сервера; окно опоздания пайплайна — "
               "больше максимального лага; мониторить долю пользователей с событием по группам.",
    },
}

TELEMETRY_PARAMS = {
    "n_days": 7,
    "users_per_day": 8_000,
    "share_b": 0.50,
    "trigger_rate": {"ios": 0.26, "android": 0.24, "web": 0.25},
    "conv_untriggered": 0.30,          # эффекта нет: виджет эти пользователи не видят
    "conv_triggered": (0.40, 0.43),    # A, B: +3 п.п. среди затронутых
    "loss_base": 0.02,                 # потеря клиентского события показа везде
    # В B на android новый виджет падает до отправки события; падение чаще у тех,
    # кто потом бросил заказ.
    "loss_b_android": {"converted": 0.10, "not_converted": 0.35},
}


def assign_group(user_id, salt, share_b=SHARE_B):
    """Детерминированное назначение: md5(соль:user_id) → бакет 0..9999."""
    h = int(hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest(), 16)
    return "B" if h % 10_000 < share_b * 10_000 else "A"


def _base_users(rng, salt, users_per_day=USERS_PER_DAY, n_days=N_DAYS, share_b=SHARE_B):
    """Все назначенные пользователи до любых поломок."""
    per_day = rng.poisson(users_per_day, size=n_days)
    n = int(per_day.sum())
    day_index = np.repeat(np.arange(n_days), per_day)
    user_ids = rng.choice(np.arange(10_000_000, 99_999_999), size=n, replace=False)
    group = np.array([assign_group(u, salt, share_b) for u in user_ids])
    device = rng.choice(DEVICES[0], size=n, p=DEVICES[1])
    source = rng.choice(SOURCES[0], size=n, p=SOURCES[1])
    return pd.DataFrame({"user_id": user_ids, "day": day_index, "group": group,
                         "device": device, "source": source})


def make_srm_case(case, seed=SEED):
    cause = CASE_CAUSES[case]
    rng = np.random.default_rng(seed + case)
    df = _base_users(rng, salt=f"platform-test-{case}")
    n = len(df)
    is_b = df["group"].eq("B").to_numpy()

    # Боты: чаще из рекламы и с веба, доля одинакова в группах.
    p_bot = BOT_RATE * np.where(df["source"].eq("ads"), 2.0, 0.6) * np.where(df["device"].eq("web"), 1.6, 0.8)
    p_bot = p_bot * BOT_RATE / p_bot.mean()
    df["is_bot"] = (rng.random(n) < p_bot).astype(int)

    lag = rng.lognormal(np.log(25), 1.0, size=n)
    lag[df["is_bot"].eq(1)] = rng.lognormal(np.log(3), 0.5, size=int(df["is_bot"].sum()))
    conv = 0.30 + 0.10 * df["device"].eq("ios") - 0.03 * df["source"].eq("ads")
    df["converted"] = ((rng.random(n) < conv) & df["is_bot"].eq(0)).astype(int)

    keep = np.ones(n, dtype=bool)
    if cause == "redirect":
        lag = lag + np.where(is_b, 1.5, 0.0)
        lost = is_b & df["source"].eq("ads").to_numpy() & (rng.random(n) < 0.45)
        keep &= ~lost
    elif cause == "start_date":
        # доля пользователей B, которым реально отдали B (остальные увидели кэш и записаны в A)
        served_b = np.select([df["day"].eq(0), df["day"].eq(1)], [0.25, 0.65], 1.0)
        to_a = is_b & (rng.random(n) > served_b)
        df.loc[to_a, "group"] = "A"
    elif cause == "bot_filter":
        flagged = is_b & df["is_bot"].eq(0).to_numpy() & (rng.random(n) < 0.065)
        df.loc[flagged, "is_bot"] = 1
        df.loc[flagged, "converted"] = 0
        lag[flagged] = rng.lognormal(np.log(6), 0.5, size=int(flagged.sum()))
    elif cause == "assignment_bug":
        to_a = is_b & df["device"].eq("ios").to_numpy() & (rng.random(n) < 0.55)
        df.loc[to_a, "group"] = "A"
    elif cause == "telemetry_loss":
        flush = rng.exponential(22 * 60, size=n)
        lag = lag + np.where(is_b, flush, 0.0)
        keep &= ~(is_b & (lag > 3600))
    else:
        raise ValueError(cause)

    df["lag_sec"] = np.round(lag, 1)
    df["date"] = (pd.Timestamp(START_DATE) + pd.to_timedelta(df["day"], unit="D")).dt.strftime("%Y-%m-%d")
    df = df[keep]
    cols = ["user_id", "date", "group", "device", "source", "is_bot", "lag_sec", "converted"]
    return df[cols].sample(frac=1, random_state=seed + case).reset_index(drop=True)


def make_telemetry(seed=TELEMETRY_SEED):
    p = TELEMETRY_PARAMS
    rng = np.random.default_rng(seed + 100)
    df = _base_users(rng, salt="delivery-slot-widget", users_per_day=p["users_per_day"],
                     n_days=p["n_days"], share_b=p["share_b"])
    n = len(df)
    is_b = df["group"].eq("B").to_numpy()
    trig_p = df["device"].map(p["trigger_rate"]).to_numpy()
    triggered = rng.random(n) < trig_p
    conv_a, conv_b = p["conv_triggered"]
    p_conv = np.where(triggered, np.where(is_b, conv_b, conv_a), p["conv_untriggered"])
    converted = rng.random(n) < p_conv

    loss = np.full(n, p["loss_base"])
    android_b = is_b & df["device"].eq("android").to_numpy()
    loss[android_b] = np.where(converted[android_b], p["loss_b_android"]["converted"],
                               p["loss_b_android"]["not_converted"])
    logged = triggered & (rng.random(n) >= loss)

    df["date"] = (pd.Timestamp(START_DATE) + pd.to_timedelta(df["day"], unit="D")).dt.strftime("%Y-%m-%d")
    df["triggered"] = triggered.astype(int)
    df["exposure_logged"] = logged.astype(int)
    df["converted"] = converted.astype(int)
    cols = ["user_id", "date", "group", "device", "triggered", "exposure_logged", "converted"]
    return df[cols].sample(frac=1, random_state=seed + 100).reset_index(drop=True)


def srm_pvalue(groups, share_b=SHARE_B):
    from scipy.stats import chisquare
    counts = groups.value_counts().reindex(["A", "B"], fill_value=0).to_numpy()
    return chisquare(counts, counts.sum() * np.array([1 - share_b, share_b])).pvalue


if __name__ == "__main__":
    here = Path(__file__).parent
    for case in CASE_CAUSES:
        data = make_srm_case(case)
        data.to_csv(here / f"srm_case_{case}.csv", index=False)
        human = data[data["is_bot"] == 0]
        p_val = srm_pvalue(human["group"])
        print(f"srm_case_{case}.csv: {len(data):,} строк; доля B (без ботов) = "
              f"{human['group'].eq('B').mean():.4f}; SRM p = {p_val:.1e}")
        assert p_val < 1e-3, f"кейс {case}: SRM должен быть заметен"

    tel = make_telemetry()
    tel.to_csv(here / "telemetry_ab.csv", index=False)
    trig = tel[tel["triggered"] == 1]
    loss = 1 - trig.groupby("group")["exposure_logged"].mean()
    print(f"telemetry_ab.csv: {len(tel):,} строк; доля triggered = {tel['triggered'].mean():.3f}; "
          f"потеря событий A/B = {loss['A']:.3f}/{loss['B']:.3f}")
