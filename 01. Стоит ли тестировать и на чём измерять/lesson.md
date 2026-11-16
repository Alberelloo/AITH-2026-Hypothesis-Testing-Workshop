# Встреча 1. Стоит ли тестировать: ценность теста в деньгах

Один сквозной вопрос на всю встречу: **стоит ли запускать A/B-тест и сколько денег он принесёт**. Кейс — интернет-магазин.

| Параметр | Значение |
| :--- | :--- |
| Конверсия в заказ на чекауте | 5% |
| Пользователей на чекауте | 50 000 / нед |
| Выручка | 3 000 000 ₽ / нед |
| Горизонт | 12 месяцев = 52 недели |
| Затронутая выручка за горизонт | 156 млн ₽ |

Допущение на всю встречу: **относительный рост конверсии = относительный рост выручки**. Для чекаута это разумно — корзина уже сформирована, мы просто доводим пользователя до заказа.

---

## Шаг 1. Завязка: тест — это затраты плюс опция

**Интуиция.** Корректно проведённый тест сам по себе ничего не стоит: ценность имеет только решение, которое меняет выручку. Тест — это одновременно два актива:

1. **Фиксированные затраты** — деньги, которые мы платим независимо от результата (разработка варианта, инфраструктура, ожидание).
2. **Опция** — право внедрить вариант только тогда, когда данные говорят «окупится», и отказаться, когда нет.

Опция ценна асимметрией: **downside ограничен затратами на тест, upside — весь годовой эффект от хорошего варианта**.

**Отсюда три действия** для каждой гипотезы:

| Действие | Что платим | Что получаем |
| :--- | :--- | :--- |
| **Тестировать** | тест сейчас | внедряем выборочно, только хорошее |
| **Внедрять без теста** | внедрение + поддержка сразу | весь эффект (хороший или плохой) на весь горизонт |
| **Не внедрять** | ничего | ничего (0) |

Вердикт — это выбор максимума из трёх чисел; посчитаем его в шаге 5.

```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# --- Единая палитра встречи ---
H1_COLOR = "#4C72B0"   # синий — H1
H2_COLOR = "#55A868"   # зелёный — H2
H3_COLOR = "#C44E52"   # красный — H3
INK      = "#1b1b1b"
GRID     = "#dcdcdc"
ACCENT   = "#8172B3"

plt.rcParams.update({
    "figure.dpi": 100,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

def trapz(y, x):
    """Интегрирование методом трапеций; совместимо с numpy 1.x и 2.x."""
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def fmt_rub(x):
    """Краткая запись денег: '408 тыс ₽', '4.95 млн ₽'."""
    a = abs(x)
    sign = "+" if x > 0 else ("-" if x < 0 else " ")
    if a >= 1_000_000:
        return f"{sign}{a/1_000_000:,.2f} млн ₽"
    if a >= 1_000:
        return f"{sign}{a/1_000:,.0f} тыс ₽"
    return f"{sign}{a:,.0f} ₽"
```

```python
# --- Сквозной кейс: интернет-магазин ---
BASELINE_CR    = 0.05        # конверсия в заказ на чекауте
WEEKLY_USERS   = 50_000      # пользователей/нед на чекауте
WEEKLY_REVENUE = 3_000_000   # выручка, ₽/нед
HORIZON_WEEKS  = 52          # 12 месяцев
REVENUE_HORIZON = WEEKLY_REVENUE * HORIZON_WEEKS

# Стандартный дизайн встречи (для модели-опции порог односторонний)
N_PER_GROUP    = 50_000
ALPHA_ONESIDED = 0.05

print(f"Горизонт {HORIZON_WEEKS} недель, затронутая выручка {REVENUE_HORIZON:,.0f} ₽")

# Три гипотезы. μ и σ — среднее и разброс ожидаемого относительного эффекта
# (из истории похожих тестов). Затраты в ₽, поддержка — ₽/мес.
HYP = {
    "H1 Купон отдельной строкой": dict(mu=0.000, sigma=0.030, test=180_000, impl=120_000, support=20_000, rollback=250_000),
    "H2 Редизайн корзины":        dict(mu=0.033, sigma=0.015, test=120_000, impl=80_000,  support=10_000, rollback=100_000),
    "H3 Платёжный провайдер":     dict(mu=-0.010, sigma=0.020, test=400_000, impl=700_000, support=30_000, rollback=2_500_000),
}

hyp_df = pd.DataFrame(HYP).T
hyp_df.columns = ["μ (ожидание)", "σ (разброс)", "Тест, ₽", "Внедрение, ₽", "Поддержка, ₽/мес", "Откат, ₽"]
hyp_df.index.name = "Гипотеза"
hyp_df
```

| Гипотеза | μ | σ | Тест | Внедрение | Поддержка | Откат |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| H1 Купон отдельной строкой | 0% | ±3% | 180 тыс | 120 тыс | 20 тыс/мес | 250 тыс |
| H2 Редизайн корзины | +3.3% | ±1.5% | 120 тыс | 80 тыс | 10 тыс/мес | 100 тыс |
| H3 Платёжный провайдер | −1% | ±2% | 400 тыс | 700 тыс | 30 тыс/мес | 2 500 тыс |

---

## Шаг 2. Точка безубыточности θ_BE

**Интуиция.** Запуск варианта требует фиксированных денег: тест + внедрение + поддержка на весь горизонт. $\theta_{BE}$ — минимальный относительный прирост выручки, который эти деньги отбивает.

**Формула.**

$$\theta_{BE} = \frac{C_{test} + C_{impl} + C_{maint}}{R_{week} \cdot H}$$

где $R_{week}$ — недельная выручка, $H$ — горизонт в неделях.

**Когда применять.** До любого теста: $\theta_{BE}$ задаёт, что такое «успех» в деньгах, а не в p-value.

**Типичная ошибка.** Забыть поддержку. Поддержка 20 тыс ₽/мес × 12 мес = 240 тыс ₽ — **часто больше самого теста** (у H1 тест 180 тыс). В безубыточность входит вся фиксированная сумма, а не только разработка.

```python
def break_even(h):
    """Относительный эффект, при котором выручка за горизонт покрывает фиксированные затраты."""
    fixed = h["test"] + h["impl"] + h["support"] * 12
    return fixed / REVENUE_HORIZON

names = list(HYP.keys())
colors = [H1_COLOR, H2_COLOR, H3_COLOR]

print("Гипотеза                  θ_BE      фикс. затраты    ₽/нед")
for name in names:
    h = HYP[name]
    be = break_even(h)
    fixed = h["test"] + h["impl"] + h["support"] * 12
    print(f"{name:24s} {be:.3%}   {fixed:>11,.0f} ₽   {be * WEEKLY_REVENUE:>9,.0f}")
```

Ожидаемый вывод:

| Гипотеза | θ_BE | Фикс. затраты | ₽/нед |
| :--- | :--- | ---: | ---: |
| H1 Купон отдельной строкой | **0.346%** | 540 тыс ₽ | 10 385 |
| H2 Редизайн корзины | **0.205%** | 320 тыс ₽ | 6 154 |
| H3 Платёжный провайдер | **0.936%** | 1 460 тыс ₽ | 28 077 |

Безубыточность почти всегда — доли процента: затраты на тест малы против выручки, которую вариант будет приносить целый год.

```python
from matplotlib.ticker import PercentFormatter

be_vals = [break_even(HYP[name]) for name in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# Левый график: lollipop θ_BE
y = np.arange(len(names))[::-1]
for yi, be, c in zip(y, be_vals, colors):
    ax1.hlines(yi, 0, be, color=c, lw=3, zorder=3)
    ax1.plot(be, yi, "o", color=c, ms=11, zorder=4)
    ax1.annotate(f"{be:.3%}\n({be * WEEKLY_REVENUE:,.0f} ₽/нед)",
                 xy=(be, yi), xytext=(be + 0.00015, yi + 0.22),
                 fontsize=9, color=c, ha="left")
ax1.set_yticks(y, names)
ax1.set_xlim(0, max(be_vals) * 1.45)
ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=3))
ax1.set_xlabel("θ_BE — минимальный относительный прирост выручки")
ax1.set_title("Точка безубыточности трёх гипотез")
ax1.tick_params(axis="y", length=0)

# Правый график: состав фиксированных затрат
x = np.arange(len(names))
bottom = np.zeros(len(names))
parts = [("Тест", [HYP[name]["test"] for name in names], INK),
         ("Внедрение", [HYP[name]["impl"] for name in names], "#8c8c8c"),
         ("Поддержка (12 мес)", [HYP[name]["support"] * 12 for name in names], "#c9c9c9")]
for label, vals, c in parts:
    ax2.bar(x, np.array(vals) / 1e3, bottom=np.array(bottom) / 1e3,
            color=c, width=0.6, label=label)
    for xi, v, b in zip(x, vals, bottom):
        ax2.text(xi, (b + v / 2) / 1e3, f"{v/1e3:,.0f}",
                 ha="center", va="center", fontsize=8.5,
                 color="white" if label == "Тест" else INK)
    bottom += np.array(vals)
ax2.set_xticks(x, ["H1", "H2", "H3"])
ax2.set_ylabel("тыс ₽")
ax2.set_title("Состав фиксированных затрат")
ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.show()
```

---

## Шаг 3. MDE в деньгах: разрешающая способность, а не порог

**Интуиция.** MDE — минимальный эффект, который тест данного размера «увидит» с заданной вероятностью (обычно 80%). Это **разрешающая способность прибора**, а не порог решения.

**Формула** (биномиальная метрика, база $p$, $n$ на группу, двусторонний $\alpha$):

$$MDE_{rel} = \frac{(z_{1-\alpha/2} + z_{1-\beta})\sqrt{2p(1-p)/n}}{p}$$

**В деньгах:** $MDE_{rel} \cdot R_{week}$ — сколько ₽/нед мы в принципе способны различить.

**Контраст MDE vs break-even — это нормально.** При $n = 50\,000$: $MDE = 7.72\%$ относительных, то есть $0.386$ п.п., или $\approx 231.7$ тыс ₽/нед — за год $\approx 12$ млн ₽. А $\theta_{BE}(H1) = 0.346\% \approx 0.54$ млн ₽. MDE в ~22 раза больше порога — **и тест всё равно выгоден** (шаг 5). Ценность теста создаёт не детектирование конкретной точки, а интегральный отбор хороших вариантов из всего распределения ожиданий.

**Когда применять.** Чтобы понять, не смешон ли размер выборки по сравнению с экономически значимой величиной.

**Типичная ошибка.** Сравнивать MDE с нулём («увидим любой эффект»). При базе 5% MDE 7.72% относительных — это всего 0.386 п.п. абсолютных: маленькая база требует большого относительного сдвига.

```python
def mde_rel(n, alpha=0.05, power=0.80):
    """Относительный MDE для сравнения двух конверсий (двусторонний тест)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se = np.sqrt(2 * BASELINE_CR * (1 - BASELINE_CR) / n)
    return (z_a + z_b) * se / BASELINE_CR

mde = mde_rel(N_PER_GROUP)
print(f"MDE (n={N_PER_GROUP:,}/группу): {mde:.2%} относительный = "
      f"{mde * BASELINE_CR * 100:.3f} п.п. = {mde * WEEKLY_REVENUE:,.0f} ₽/нед")
print(f"За горизонт это {mde * REVENUE_HORIZON:,.0f} ₽  (θ_BE H1 = {break_even(HYP['H1 Купон отдельной строкой']) * REVENUE_HORIZON:,.0f} ₽)")
```

```python
n_grid = np.logspace(3.3, 5.8, 200)   # от ~2 000 до ~630 000 на группу
mde_grid = mde_rel(n_grid)

fig, ax = plt.subplots(figsize=(9.8, 5.4))
ax.semilogx(n_grid, mde_grid * 100, color=INK, lw=2,
            label="MDE (двусторонний α=0.05, мощность 80%)")

for name, c in zip(names, colors):
    ax.axhline(break_even(HYP[name]) * 100, color=c, ls="--", lw=1.5)

ax.axvline(N_PER_GROUP, color=ACCENT, ls=":", lw=1.4)
ax.plot(N_PER_GROUP, mde * 100, "o", color=ACCENT, ms=9, zorder=5)
ax.annotate(f"n=50k → MDE {mde:.2%}\n= {mde*BASELINE_CR*100:.3f} п.п.\n= {mde*WEEKLY_REVENUE:,.0f} ₽/нед",
            xy=(N_PER_GROUP, mde * 100), xytext=(N_PER_GROUP, mde * 100 * 2.3),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
            fontsize=9, color=ACCENT, ha="center")

ax.set_xlabel("n на группу (логарифмическая шкала)")
ax.set_ylabel("MDE, % относительный")
ax.set_title("Разрешающая способность (MDE) vs порог решения (θ_BE)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, max(mde_grid * 100) * 1.05)

# Зум: три порога почти совпадают и сливаются у нуля — покажем их крупно.
axins = ax.inset_axes([0.04, 0.08, 0.40, 0.30])
for name, c in zip(names, colors):
    be = break_even(HYP[name])
    axins.axhline(be * 100, color=c, ls="--", lw=1.8)
    axins.text(0.02, be * 100 * 1.04, f"{name.split()[0]}  θ_BE = {be:.3%}",
               color=c, fontsize=8, va="bottom")
axins.set_xlim(0, 1)
axins.set_ylim(0, max(be_vals) * 100 * 1.6)
axins.set_xticks([])
axins.set_yticks([])
axins.set_title("θ_BE крупно", fontsize=8, pad=4)

plt.tight_layout()
plt.show()
```

Читаем график: кривая MDE уходит вниз с ростом $n$ (больше данных — выше разрешение), но три порога $\theta_{BE}$ лежат далеко внизу. Тест при $n=50$k «не различает» саму точку безубыточности — и это нормально: решение принимается по порогу $\theta_{BE}$ односторонним тестом, а выгода считается интегрально по всем эффектам.

---

## Шаг 4. Тест как опция (ядро)

### (а) Ожидания эффекта: распределение, а не одно число

**Интуиция.** Истинный эффект $\theta$ до теста неизвестен. Опыт прошлых похожих тестов даёт не «точку», а **распределение** $\pi(\theta) = \mathcal{N}(\mu, \sigma)$: $\mu$ — типичный эффект, $\sigma$ — насколько эффекты разбросаны от гипотезы к гипотезе.

**Зачем распределение, а не одно число.** Если взять только $\mu$, мы не увидим шанс на крупный успех: у H1 $\mu = 0$, но есть ненулевая вероятность, что эффект окажется +5% или −5%. Именно хвост хороших исходов, взвешенный вероятностью, создаёт ценность теста.

**Когда применять.** Всегда, когда оцениваем тест до запуска: $\pi(\theta)$ — вход для ожидаемой прибыли.

**Типичная ошибка.** Заменять $\pi(\theta)$ «точечной оценкой» $\mu$ и решать по ней. Для H1 это дало бы «ожидание ноль → ничего не делаем», а корректный расчёт (шаг 5) — «тестировать».

```python
from matplotlib.lines import Line2D

theta_axis = np.linspace(-0.06, 0.06, 1200)

fig, ax = plt.subplots(figsize=(10, 5.4))
peak = 0
curve_handles = []
for name, c in zip(names, colors):
    h = HYP[name]
    pdf = stats.norm.pdf(theta_axis, h["mu"], h["sigma"])
    peak = max(peak, pdf.max())
    (line,) = ax.plot(theta_axis, pdf, color=c, lw=2.2,
                      label=f"{name}: N(μ={h['mu']:+.1%}, σ={h['sigma']:.1%})")
    curve_handles.append(line)

ax.axvline(0, color=INK, ls="--", lw=1.2, alpha=0.7)
ax.text(0, peak * 0.02, "0", color=INK, fontsize=10, ha="center")

be_handles = []
for name, c in zip(names, colors):
    be = break_even(HYP[name])
    ax.axvline(be, color=c, ls=":", lw=1.5, alpha=0.9)
    be_handles.append(Line2D([0], [0], color=c, ls=":", lw=1.5))

ax.legend(curve_handles + be_handles,
          [h.get_label() for h in curve_handles]
          + [f"θ_BE {n.split()[0]} = {break_even(HYP[n]):.2%}" for n in names],
          fontsize=9, loc="upper left")

ax.set_xlabel("θ — относительный эффект (рост выручки)")
ax.set_ylabel("плотность π(θ)")
ax.set_title("Априорные ожидания эффекта трёх гипотез")
ax.set_ylim(0, peak * 1.2)
plt.tight_layout()
plt.show()
```

Обратите внимание на масштаб: все три $\theta_{BE}$ лежат в полосе 0.2–0.9% и на графике прижаты к нулю. Почти вся масса H2 выше безубыточности, у H3 — в основном ниже, у H1 — примерно пополам.

### (б) Вероятность внедрения POW(θ)

**Интуиция.** Решение привязывается не к нулю, а к точке безубыточности: внедряем, только если оценка эффекта **значимо выше** $\theta_{BE}$.

$$H_0:\ \theta \le \theta_{BE}, \qquad H_1:\ \theta > \theta_{BE}$$

Внедряем при $\hat\delta > \delta_{BE} + z_{1-\alpha}\cdot SE$, где $\delta_{BE} = p \cdot \theta_{BE}$.

**Формула.** Вероятность внедрения при истинном эффекте $\theta$:

$$POW(\theta) = \Phi\!\left(\frac{\delta_{true}(\theta) - \delta_{BE}}{SE} - z_{1-\alpha}\right), \qquad
SE = \sqrt{\frac{p(1-p)}{n} + \frac{p_t(1-p_t)}{n}}$$

где $\delta_{true}(\theta) = p_t - p = p\cdot\theta$.

**Свойство.** При $\theta = \theta_{BE}$ мощность равна $\alpha$ (одностороннему): если эффект ровно на границе безубыточности, вероятность внедрить его — это и есть уровень значимости. $\alpha$ здесь **односторонний** (двусторонний 0.05 = односторонний 0.025 — строже).

**Когда применять.** Это «передаточная функция» теста: она связывает истинный эффект с шансом, что мы его внедрим.

**Типичная ошибка.** Считать, что при $\alpha = 0.5$ «внедряем как монетку». Нет: правило «внедряем, если оценка выше порога» отбирает хорошие варианты чаще плохих — $POW(\theta)$ растёт с эффектом.

```python
def adoption_prob(h, theta, n, alpha):
    """P(внедрения | истинный эффект θ). Односторонняя гипотеза θ > θ_BE."""
    p_c = BASELINE_CR
    p_t = np.clip(p_c * (1 + theta), 1e-9, 1 - 1e-9)
    delta_true = p_t - p_c
    delta_be = p_c * break_even(h)
    se = np.sqrt(p_c * (1 - p_c) / n + p_t * (1 - p_t) / n)
    return stats.norm.sf(stats.norm.ppf(1 - alpha) - (delta_true - delta_be) / se)

h1 = HYP["H1 Купон отдельной строкой"]
be_h1 = break_even(h1)
print(f"Проверка: POW(θ_BE) = {adoption_prob(h1, be_h1, N_PER_GROUP, ALPHA_ONESIDED):.3f}  (должно быть α = {ALPHA_ONESIDED:.2f})")
```

```python
theta_pow = np.linspace(-0.02, 0.10, 800)
pow_h1 = adoption_prob(h1, theta_pow, N_PER_GROUP, ALPHA_ONESIDED)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(theta_pow, pow_h1, color=H1_COLOR, lw=2.5)
ax.axvline(be_h1, color=INK, ls="--", lw=1.3)
ax.axhline(ALPHA_ONESIDED, color=ACCENT, ls=":", lw=1.4)
ax.plot(be_h1, ALPHA_ONESIDED, "o", color=ACCENT, ms=9, zorder=5)
ax.annotate(f"при θ = θ_BE = {be_h1:.2%}\nPOW = α = {ALPHA_ONESIDED:.0%}",
            xy=(be_h1, ALPHA_ONESIDED), xytext=(0.008, 0.15),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
            fontsize=10, color=ACCENT)

ax.set_xlabel("θ — истинный относительный эффект")
ax.set_ylabel("P(внедрения | θ)")
ax.set_title("Вероятность внедрения H1 (n=50 000, односторонний α=0.05)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
```

Кривая растёт с эффектом: при $\theta = 0$ внедрим с шансом $\approx 3.8\%$, при $\theta = +3\%$ — уже $\approx 25\%$, при $\theta = +8\%$ — $\approx 86\%$, а при $\theta = +10\%$ — практически наверняка. Тест — это **фильтр**, который тем охотнее пропускает вариант, чем он выгоднее.

### (в)–(д) Выгоды, затраты, Net и RR

**Интуиция.** Ожидаемая ценность теста складывается из четырёх денежных потоков, взвешенных по $\pi(\theta)$ и по $POW(\theta)$.

Во время теста ($T_{test} = 2n / U_{week}$ недель) вариант видит только половина трафика, после внедрения — весь трафик $T_{post} = H - T_{test}$ недель, но только с вероятностью $POW(\theta)$.

**Выгоды.**

$$Gain_{RA} = \int \pi(\theta)\Big[ \underbrace{\max(\theta R_{test}, 0)}_{\text{во время теста}} + \underbrace{\max(\theta R_{post}, 0)\cdot POW(\theta)}_{\text{после внедрения}} \Big] d\theta$$

**Затраты.**

$$Cost_{RA} = C_{test} + \int \pi(\theta)\Big[ \underbrace{\max(-\theta R_{test}, 0)}_{\text{потеря во время теста}} + \underbrace{\big(\max(-\theta R_{post}, 0) + C_d(\theta)\big)\cdot POW(\theta)}_{\text{потеря после внедрения + затраты решения}} \Big] d\theta$$

где $R_{test} = R_{week}\cdot T_{test}\cdot 0.5$, $R_{post} = R_{week}\cdot T_{post}$, а затраты решения $C_d(\theta) = C_{impl} + C_{maint} + C_{rollback}\cdot\mathbb{I}(\theta < \theta_{BE})$ — внедрение и поддержку платим всегда при внедрении, откат — только если внедрили вариант, чей истинный эффект оказался ниже безубыточности.

**Итоги.** Два числа отвечают на разные вопросы:

- **Чистая прибыль** $Net = Gain_{RA} - Cost_{RA}$ — сколько в среднем заработаем на этом решении. Смотрим, когда решение одно и важен абсолютный результат.
- **Отдача на рубль** $RR = Gain_{RA} / Cost_{RA}$ — сколько ₽ выгоды на ₽ затрат. Смотрим, когда тестов много, ресурсы ограничены и нужна лучшая отдача на вложенный рубль ($RR = 1$ означает $ROI = 0$).

**Когда применять.** Перед любым тестом, у которого есть цена и история похожих эффектов.

**Типичная ошибка.** Сравнивать тесты по «мощности» или «красоте дизайна», а не по $Net$/$RR$ — и не считать альтернативу «внедрить без теста».

### (е) Правило вердикта

Для каждой гипотезы считаем три числа и берём максимум:

| Действие | Формула |
| :--- | :--- |
| Тестировать | $Net$ (формулы выше) |
| Не внедрять | $0$ |
| Внедрять без теста | $\mu \cdot R_{week} \cdot H - (C_{impl} + C_{maint})$ |

Если максимум — «тестировать», запускаем тест; если «внедрять без теста» — экономим на тесте и внедряем сразу; если «0» — не делаем ничего.

Сначала — наглядная «опция»: ценность трёх действий как функция истинного эффекта $\theta$ (до интегрирования по $\pi$).

```python
def test_value_given_theta(h, theta, n, alpha):
    """V_test(θ): ценность теста при известном истинном эффекте θ."""
    test_weeks = 2 * n / WEEKLY_USERS
    post_weeks = max(HORIZON_WEEKS - test_weeks, 0)
    r_test = WEEKLY_REVENUE * test_weeks * 0.5
    r_post = WEEKLY_REVENUE * post_weeks
    pow_t = adoption_prob(h, theta, n, alpha)
    c_d = h["impl"] + h["support"] * 12 + h["rollback"] * (theta < break_even(h))
    return -h["test"] + theta * r_test + pow_t * (theta * r_post - c_d)

def ship_value_given_theta(h, theta):
    return theta * REVENUE_HORIZON - (h["impl"] + h["support"] * 12)

theta_pay = np.linspace(-0.04, 0.08, 800)
v_test = test_value_given_theta(h1, theta_pay, N_PER_GROUP, ALPHA_ONESIDED)
v_ship = ship_value_given_theta(h1, theta_pay)

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(theta_pay, v_test / 1e6, color=H1_COLOR, lw=2.4, label="Тестировать: V(θ)")
ax.plot(theta_pay, v_ship / 1e6, color=H3_COLOR, lw=2.0, label="Внедрять без теста")
ax.axhline(0, color=INK, lw=1.2, ls="--", label="Не внедрять: 0")
ax.axvline(be_h1, color=INK, ls=":", lw=1.3)
ax.text(be_h1, -5.4, f" θ_BE={be_h1:.2%}", color=INK, fontsize=9)

ax.annotate("downside ограничен\nзатратами теста",
            xy=(-0.03, test_value_given_theta(h1, -0.03, N_PER_GROUP, ALPHA_ONESIDED) / 1e6),
            xytext=(-0.038, -1.6),
            arrowprops=dict(arrowstyle="->", color=H1_COLOR, lw=1.2),
            fontsize=9, color=H1_COLOR)
ax.annotate("внедрение без теста\nуходит в минус",
            xy=(-0.03, ship_value_given_theta(h1, -0.03) / 1e6),
            xytext=(-0.038, -9.0),
            arrowprops=dict(arrowstyle="->", color=H3_COLOR, lw=1.2),
            fontsize=9, color=H3_COLOR)

ax.set_xlabel("θ — истинный относительный эффект")
ax.set_ylabel("Ценность решения, млн ₽")
ax.set_title("Тест как опция (H1): ограниченный downside, отбор upside")
ax.legend(fontsize=10)
ax.set_ylim(-11, 13)
plt.tight_layout()
plt.show()
```

Видна асимметрия опции: «внедрять без теста» — прямая линия (при плохом эффекте убыток неограничен), «тестировать» — кривая с плоским дном и ростом на хороших эффектах. Это и есть та самая опция из шага 1.

Теперь модель целиком.

```python
effect_grid = np.linspace(-0.10, 0.10, 2001)

def evaluate(h, n, alpha):
    """Ожидаемые выгоды, затраты, Net, RR и декомпозиция по компонентам."""
    pdf = stats.norm.pdf(effect_grid, h["mu"], h["sigma"])
    pdf /= trapz(pdf, effect_grid)          # сетка обрезает хвосты — нормируем
    pow_t = adoption_prob(h, effect_grid, n, alpha)

    test_weeks = 2 * n / WEEKLY_USERS
    post_weeks = max(HORIZON_WEEKS - test_weeks, 0)
    r_test = WEEKLY_REVENUE * test_weeks * 0.5
    r_post = WEEKLY_REVENUE * post_weeks

    gain_test = np.maximum(r_test * effect_grid, 0)
    gain_post = np.maximum(r_post * effect_grid, 0) * pow_t
    loss_test = np.maximum(-r_test * effect_grid, 0)
    loss_post = np.maximum(-r_post * effect_grid, 0) * pow_t
    decision_cost = (h["impl"] + h["support"] * 12
                     + h["rollback"] * (effect_grid < break_even(h))) * pow_t

    gain = trapz(pdf * (gain_test + gain_post), effect_grid)
    loss = (h["test"]
            + trapz(pdf * loss_test, effect_grid)
            + trapz(pdf * loss_post, effect_grid)
            + trapz(pdf * decision_cost, effect_grid))
    p_adopt = trapz(pdf * pow_t, effect_grid)

    return dict(
        gain=gain, loss=loss, net=gain - loss, rr=gain / loss, p_adopt=p_adopt,
        gain_test=trapz(pdf * gain_test, effect_grid),
        gain_post=trapz(pdf * gain_post, effect_grid),
        loss_test=trapz(pdf * loss_test, effect_grid),
        loss_post=trapz(pdf * loss_post, effect_grid),
        loss_decision=trapz(pdf * decision_cost, effect_grid),
        cost_test=h["test"],
    )

def verdict(h, n, alpha):
    r = evaluate(h, n, alpha)
    ship = h["mu"] * REVENUE_HORIZON - (h["impl"] + h["support"] * 12)
    if r["net"] <= 0 and ship <= 0:
        return "не делать"
    return "тестировать" if r["net"] >= ship else "внедрять без теста"
```

---

## Шаг 5. Практика: три гипотезы — три вердикта

```python
rows = []
for name in names:
    h = HYP[name]
    r = evaluate(h, N_PER_GROUP, ALPHA_ONESIDED)
    ship = h["mu"] * REVENUE_HORIZON - (h["impl"] + h["support"] * 12)
    rows.append((name, r["p_adopt"], r["gain"], r["loss"], r["net"], r["rr"], ship, verdict(h, N_PER_GROUP, ALPHA_ONESIDED)))

res_df = pd.DataFrame(rows, columns=[
    "Гипотеза", "P(внедр.)", "Выгоды, ₽", "Затраты, ₽",
    "Чистая прибыль (тест), ₽", "RR", "Внедрить без теста, ₽", "Вердикт"])
res_df.round(0)
```

Ожидаемые числа:

| Гипотеза | P(внедр.) | Чистая прибыль (тест) | RR | Внедрить без теста | Вердикт |
| :--- | ---: | ---: | ---: | ---: | :--- |
| H1 Купон отдельной строкой | 11.4% | **+408 тыс ₽** | 2.54 | −360 тыс ₽ | **тестировать** |
| H2 Редизайн корзины | 31.9% | +1.88 млн ₽ | 11.2 | **+4.95 млн ₽** | **внедрять без теста** |
| H3 Платёжный провайдер | 2.9% | −412 тыс ₽ | 0.18 | −2.62 млн ₽ | **не делать** |

### H1 — тестировать: тест превращает ноль в прибыль

У H1 ожидание эффекта ровно ноль. «Внедрять без теста» даёт −360 тыс ₽ (платим внедрение и поддержку, а эффекта в среднем нет). Тест превращает это в **+408 тыс ₽** — хотя вероятность внедрения всего 11.4%.

```python
r1 = evaluate(h1, N_PER_GROUP, ALPHA_ONESIDED)
print(f"Выгода во время теста : {r1['gain_test']:>+12,.0f} ₽")
print(f"Выгода после внедрения: {r1['gain_post']:>+12,.0f} ₽")
print(f"Потеря во время теста : {-r1['loss_test']:>+12,.0f} ₽")
print(f"Потеря после внедрения: {-r1['loss_post']:>+12,.0f} ₽")
print(f"Внедрение/откат (ожид.): {-r1['loss_decision']:>+12,.0f} ₽")
print(f"Тест (фикс.)          : {-r1['cost_test']:>+12,.0f} ₽")
print(f"ИТОГО чистая прибыль  : {r1['net']:>+12,.0f} ₽   при P(внедрения) = {r1['p_adopt']:.1%}")
```

```python
# Waterfall-диаграмма разложения прибыли H1
labels = ["Выгода во время теста", "Выгода после внедрения",
          "Потеря во время теста", "Потеря после внедрения",
          "Внедрение/откат (ожид.)", "Тест (фикс.)", "ИТОГО"]
values = [r1["gain_test"], r1["gain_post"],
          -r1["loss_test"], -r1["loss_post"],
          -r1["loss_decision"], -r1["cost_test"], r1["net"]]

running = 0.0
bottoms, tops = [], []
for v in values[:-1]:
    tops.append(max(running + v, running))
    bottoms.append(min(running + v, running))
    running += v
tops.append(max(0, running)); bottoms.append(0)

fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(labels))
bar_colors = [H2_COLOR if v >= 0 else H3_COLOR for v in values]
bar_colors[-1] = H1_COLOR

for xi, (v, b, t, c) in enumerate(zip(values, bottoms, tops, bar_colors)):
    ax.bar(xi, (t - b) / 1e3, bottom=b / 1e3, color=c, width=0.62,
           edgecolor="white", linewidth=0.5, zorder=3)
    label_y = t / 1e3 if v >= 0 else b / 1e3
    ax.text(xi, label_y + (0.12 if v >= 0 else -0.18), f"{v/1e3:+,.0f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5,
            fontweight="bold" if xi == len(values) - 1 else "normal", color=c)

# аккуратные соединительные линии между барами
cum = 0.0
for xi, v in enumerate(values[:-1]):
    cum += v
    if xi < len(values) - 1:
        ax.plot([xi + 0.31, xi + 0.69], [cum / 1e3, cum / 1e3],
                color="#999999", lw=1, zorder=2)

ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x, labels, rotation=12, ha="right")
ax.set_ylabel("тыс ₽")
ax.set_title(f"H1: из чего складывается {fmt_rub(r1['net'])} чистой прибыли")
ax.set_ylim(min(bottoms) / 1e3 - 60, max(tops) / 1e3 + 80)
plt.tight_layout()
plt.show()
```

Читаем декомпозицию. Во время теста мы трогаем пол-трафика и эффект в среднем нулевой, поэтому выгода и потеря почти симметричны (+35.8 / −35.8 тыс ₽) и гасят друг друга. Главный вклад — **выгода после внедрения: +637.6 тыс ₽**. Она взвешена вероятностью внедрения: мы внедряем только 11.4% вариантов, но именно те, где оценка оказалась сильно выше порога. Тест работает как **фильтр**: отсекает ~88.6% «мусора», оставляя хвост хороших исходов. Вычитаем фиксированные 180 тыс на тест и ожидаемые ~42.8 тыс на внедрение/откат — остаётся +408 тыс ₽.

```python
# «Тест как фильтр»: какая доля априорной плотности доходит до внедрения
theta_f = np.linspace(-0.06, 0.06, 1200)
pdf_f = stats.norm.pdf(theta_f, h1["mu"], h1["sigma"])
pdf_f /= trapz(pdf_f, theta_f)
kept = pdf_f * adoption_prob(h1, theta_f, N_PER_GROUP, ALPHA_ONESIDED)
p_adopt_area = trapz(kept, theta_f)

fig, ax = plt.subplots(figsize=(9.8, 5.4))
ax.fill_between(theta_f, pdf_f, color=H1_COLOR, alpha=0.13,
                label="π(θ) — вся априорная масса (100%)")
ax.fill_between(theta_f, kept, color=H1_COLOR, alpha=0.75,
                label=f"π(θ)·POW(θ) — доходит до внедрения ({p_adopt_area:.1%})")
ax.plot(theta_f, pdf_f, color=H1_COLOR, lw=1.4, alpha=0.45)
ax.plot(theta_f, kept, color=H1_COLOR, lw=2.2)

ax.axvline(0, color=INK, ls="--", lw=1.1, alpha=0.7)
ax.axvline(be_h1, color=INK, ls=":", lw=1.4)
ax.text(be_h1, pdf_f.max() * 0.92, f"  θ_BE={be_h1:.2%}", color=INK, fontsize=9)

# аннотация «мусора» и «пропущенного»
ax.annotate("отсекается ≈ 88.6%\n(мусор)", xy=(-0.02, 4.2), xytext=(-0.052, 7.5),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=1.2), fontsize=9.5, color="#555555")
ax.annotate("пропускается ≈ 11.4%\n(хвост хороших исходов)", xy=(0.028, 3.1), xytext=(0.028, 8.5),
            arrowprops=dict(arrowstyle="->", color=H1_COLOR, lw=1.2), fontsize=9.5, color=H1_COLOR)

ax.set_xlabel("θ — истинный относительный эффект")
ax.set_ylabel("плотность")
ax.set_title("Тест как фильтр: тест пропускает только хвост хороших исходов")
ax.legend(fontsize=9.5, loc="upper left")
ax.set_ylim(0, pdf_f.max() * 1.18)
plt.tight_layout()
plt.show()
```

Светлая область — всё, что может произойти с эффектом по нашему априорному ожиданию. Тёмная — то, что реально доедет до внедрения после фильтра-теста. Площадь тёмной области и есть $P(\text{внедрения}) = 11.4\%$: тест выбрасывает почти 9/10 массы, но выбрасывает именно «мусор» (нулевые и отрицательные эффекты).

### H2 — внедрять без теста: тест не окупает свою задержку

У H2 ожидание +3.3% при узком разбросе ±1.5% — мы почти уверены в эффекте. «Внедрять без теста» даёт **+4.95 млн ₽**, тогда как тест — только +1.88 млн ₽.

```python
r2 = evaluate(HYP["H2 Редизайн корзины"], N_PER_GROUP, ALPHA_ONESIDED)
ship2 = HYP["H2 Редизайн корзины"]["mu"] * REVENUE_HORIZON - (HYP["H2 Редизайн корзины"]["impl"] + HYP["H2 Редизайн корзины"]["support"] * 12)
print(f"H2 тест: Net = {fmt_rub(r2['net'])}, P(внедр.) = {r2['p_adopt']:.1%}")
print(f"H2 внедрить без теста = {fmt_rub(ship2)}")
print(f"Проигрыш теста: {fmt_rub(ship2 - r2['net'])} (тест теряет {r2['p_adopt']:.0%} внедрений и откладывает выгоду)")
```

Почему тест хуже: он внедряет только 31.9% вариантов (остальные ~68% — ложные «нет» при таком уверенном априорном эффекте) и откладывает выгоду на время теста. Задержка и неполная уверенность стоят дороже, чем сам тест. Когда эффект настолько надёжен, тест — это дорогая страховка от риска, которого почти нет.

### H3 — не делать: отрицательное ожидание не спасает ни тест, ни внедрение

```python
r3 = evaluate(HYP["H3 Платёжный провайдер"], N_PER_GROUP, ALPHA_ONESIDED)
ship3 = HYP["H3 Платёжный провайдер"]["mu"] * REVENUE_HORIZON - (HYP["H3 Платёжный провайдер"]["impl"] + HYP["H3 Платёжный провайдер"]["support"] * 12)
print(f"H3 тест: Net = {fmt_rub(r3['net'])}, P(внедр.) = {r3['p_adopt']:.1%}")
print(f"H3 внедрить без теста = {fmt_rub(ship3)}")
```

У H3 ожидание −1%: и тест (−412 тыс ₽), и внедрение без теста (−2.62 млн ₽) в среднем убыточны. Единственное разумное действие — не делать ничего (0 ₽). Тест не «переворачивает» отрицательное априорное ожидание: он лишь иногда ошибочно внедрит плохой вариант, за что мы заплатим откатом 2.5 млн ₽.

```python
# Итоговая визуальная сводка
actions = ["Тест", "Внедрять без теста", "Не внедрять"]
res = {name: evaluate(h, N_PER_GROUP, ALPHA_ONESIDED) for name, h in HYP.items()}
ships = {name: h["mu"] * REVENUE_HORIZON - (h["impl"] + h["support"] * 12) for name, h in HYP.items()}

from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(10.5, 5.6))
x = np.arange(len(names))
width = 0.26
offsets = [-width, 0, width]

all_vals = []
for xi, (name, c) in enumerate(zip(names, colors)):
    vals = [res[name]["net"], ships[name], 0.0]
    all_vals.append(vals)
    best_idx = int(np.argmax(vals))
    for k, off in enumerate(offsets):
        v = vals[k]
        color = c if k == best_idx else "#c7c7c7"
        ax.bar(xi + off, v / 1e6, width * 0.92, color=color,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(xi + off, v / 1e6 + (0.12 if v >= 0 else -0.24),
                f"{v/1e6:+,.2f}" if abs(v) >= 1e5 else "0",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.3, color=color if k == best_idx else "#8a8a8a")
    vd = verdict(HYP[name], N_PER_GROUP, ALPHA_ONESIDED)
    ax.text(xi, max(vals) / 1e6 + 0.62, vd.upper(), ha="center",
            fontweight="bold", fontsize=11, color=c)

ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n[Тест | Внедрить | Ничего]" for n in names], fontsize=10)
ax.set_ylabel("Чистая прибыль, млн ₽")
ax.set_title("Три гипотезы — три вердикта (n=50 000, односторонний α=0.05)")
ax.legend(handles=[
    Patch(color=H1_COLOR, label="H1 · тестировать"),
    Patch(color=H2_COLOR, label="H2 · внедрять без теста"),
    Patch(color=H3_COLOR, label="H3 · не делать"),
    Patch(color="#c7c7c7", label="альтернативные действия"),
], loc="upper left", fontsize=9)
ax.set_ylim(-3.4, max(max(v) for v in all_vals) / 1e6 + 1.6)
plt.tight_layout()
plt.show()
```

---

## Шаг 6. Выводы

1. **Сначала экономика, потом статистика.** Порог решения — не p-value и не MDE, а точка безубыточности $\theta_{BE}$. MDE — только разрешающая способность теста; то, что MDE (≈12 млн ₽) много больше $\theta_{BE}$ (≈0.54 млн ₽), нормально и не отменяет выгоду теста.
2. **Тест — это опция.** Он ценен не «знанием», а отбором: H1 при нулевом ожидании и вероятности внедрения 11.4% превращает −360 тыс ₽ (внедрять вслепую) в +408 тыс ₽ чистой прибыли.
3. **Один инструмент — три действия.** H1 — тестировать, H2 — внедрять без теста, H3 — не делать. Вердикт всегда выбирается из трёх чисел: $Net$ теста, $0$, и $\mu \cdot R \cdot H - (C_{impl}+C_{maint})$.
4. **Считайте деньги, а не ритуал.** Для каждого решения сравнивайте $Net$ и $RR$ и помните про «внедрять без теста»: уверенный эффект (H2) дешевле внедрить сразу, а отрицательное ожидание (H3) не спасает даже тест.

---

## Интерактив: покрутите рычаги

Два слайдера. Первый — размер теста $n$: как меняется MDE и вероятность внедрения H1. Второй — ожидание $\mu$ для H1: как меняются прибыль и вердикт.

```python
def _p_adopt_h1(n):
    return evaluate(h1, n, ALPHA_ONESIDED)["p_adopt"]

def _plot_mde_and_padopt(n):
    mde_n = mde_rel(n)
    padopt_n = _p_adopt_h1(n)
    ns = np.logspace(3.3, 5.8, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.semilogx(ns, mde_rel(ns) * 100, color=INK, lw=2)
    ax1.axvline(n, color=ACCENT, ls=":", lw=1.4)
    ax1.plot(n, mde_n * 100, "o", color=ACCENT, ms=9, zorder=5)
    ax1.set_xlabel("n на группу (лог. шкала)")
    ax1.set_ylabel("MDE, % относительный")
    ax1.set_title(f"n={n:,} → MDE = {mde_n:.2%}")

    ax2.semilogx(ns, [_p_adopt_h1(nn) for nn in ns], color=H1_COLOR, lw=2)
    ax2.axvline(n, color=ACCENT, ls=":", lw=1.4)
    ax2.plot(n, padopt_n, "o", color=ACCENT, ms=9, zorder=5)
    ax2.set_xlabel("n на группу (лог. шкала)")
    ax2.set_ylabel("P(внедрения H1)")
    ax2.set_title(f"n={n:,} → P(внедрения) = {padopt_n:.1%}")

    plt.tight_layout()
    plt.show()

def _plot_h1_net(mu):
    h = dict(h1)
    h["mu"] = mu
    r = evaluate(h, N_PER_GROUP, ALPHA_ONESIDED)
    ship = mu * REVENUE_HORIZON - (h1["impl"] + h1["support"] * 12)

    mus = np.linspace(-0.05, 0.05, 200)
    test_nets = [evaluate(dict(h1, mu=m), N_PER_GROUP, ALPHA_ONESIDED)["net"] for m in mus]
    ship_nets = [m * REVENUE_HORIZON - (h1["impl"] + h1["support"] * 12) for m in mus]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(mus, np.array(test_nets) / 1e6, color=H1_COLOR, lw=2.4, label="Тест (Net)")
    ax.plot(mus, np.array(ship_nets) / 1e6, color=H3_COLOR, lw=2.0, label="Внедрять без теста")
    ax.axhline(0, color=INK, lw=1.2, ls="--", label="Не внедрять")
    ax.axvline(mu, color=ACCENT, ls=":", lw=1.5)
    ax.plot(mu, r["net"] / 1e6, "o", color=H1_COLOR, ms=10, zorder=5)
    ax.plot(mu, ship / 1e6, "o", color=H3_COLOR, ms=8, zorder=5)

    vd = verdict(h, N_PER_GROUP, ALPHA_ONESIDED)
    ax.set_xlabel("μ — ожидание эффекта H1")
    ax.set_ylabel("Чистая прибыль, млн ₽")
    ax.set_title(f"μ={mu:+.1%} → Net(тест)={r['net']/1e6:+.2f} млн ₽, вердикт: {vd}")
    ax.legend(fontsize=10)
    ax.set_ylim(-4.5, 5.5)
    plt.tight_layout()
    plt.show()

try:
    import ipywidgets as widgets
    from ipywidgets import interact, IntSlider, FloatSlider
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False

if _HAS_WIDGETS:
    interact(_plot_mde_and_padopt,
             n=IntSlider(min=10_000, max=300_000, step=5_000, value=50_000,
                         description="n/группу"))
    interact(_plot_h1_net,
             mu=FloatSlider(min=-0.05, max=0.05, step=0.0025, value=0.0,
                            description="μ (H1)", readout_format="+.1%"))
else:
    print("ipywidgets недоступны — показываю статические точки.")
    _plot_mde_and_padopt(N_PER_GROUP)
    _plot_h1_net(0.0)
```
