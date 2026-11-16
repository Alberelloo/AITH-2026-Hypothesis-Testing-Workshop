# Встреча 1. Стоит ли тестировать: цена и ценность A/B-теста в деньгах

Один сквозной вопрос на всю встречу: **стоит ли запускать A/B-тест, и если да — какой дизайн принесёт больше денег**. Сквозной кейс — интернет-магазин, который тестирует изменение страницы оформления заказа (checkout).

Почему это не тривиально. Традиционный рецепт «$\alpha = 0.05$, мощность $80\%$, MDE на глаз» игнорирует экономику конкретного решения: сколько стоит тест, сколько стоит ошибка и сколько мы заработаем, если вариант окажется удачным. Мы построим модель, которая подбирает $\alpha$ и $n$ под финансовый результат, и честно посмотрим, где её можно обмануть.

```python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

try:
    from ipywidgets import FloatSlider, IntSlider, interact
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

BLUE, RED, GREEN, GRAY, INK = '#4C72B0', '#C44E52', '#55A868', '#7F7F7F', '#1b1b1b'


def spaced(value: float) -> str:
    """Форматирует число с пробелами между разрядами."""
    return f'{value:,.0f}'.replace(',', ' ')
```

---

## Часть 1. Финансовая модель: из чего складывается ценность теста

### 1.1 Успех теста — это деньги, а не p-value

**Интуиция.** В науке корректно проведённый эксперимент ценен сам по себе — он увеличивает знание. В бизнесе критерий другой. Ключевой вопрос перед тестом звучит так:

> Основываясь на наших ограниченных ресурсах, какое действие улучшит KPI наиболее экономически эффективным и устойчивым способом?

Тест ценен не «знанием», а **решениями, которые он меняет**. Поэтому перед любым существенным действием нужно ответить на три вопроса с учётом всех рисков:

1. Каков общий доход, на который может повлиять запланированное действие?
2. Каков ожидаемый положительный эффект от этого действия?
3. Каковы ожидаемые суммарные затраты — включая постоянные и связанные с риском?

**Формула.** Оптимизируя первые два пункта и минимизируя третий, мы приближаемся к максимальной отдаче:

$$ROI = \frac{Gain - Costs}{Costs}, \qquad Gain = Effect \times Impact$$

Здесь $Gain$ — эффект, помноженный на масштаб влияния (затронутую выручку). Мы используем стандартное определение ROI: при $ROI = 0$ выгоды лишь покрыли затраты. Далее удобнее работать не с самим ROI, а с отношением $RR = Gain / Costs$: тогда $RR = 1$ означает $ROI = 0$, а $ROI = RR - 1$.

**Пример на кейсе.** Если изменение checkout-страницы поднимает конверсию на $0.5\%$ при годовой выручке, затронутой решением, в $\approx 156$ млн ₽, то $Gain \approx 0.005 \times 156\,\text{млн} \approx 0.78$ млн ₽ — и это надо сравнивать с затратами на тест, внедрение и поддержку.

**Типичная ошибка.** Мерить успех «нашёл значимый эффект». Значимый эффект может не окупать затраты на внедрение — тогда тест успешен статистически и убыточен экономически.

### 1.2 Затраты и выгоды на шкале времени

**Интуиция.** Чтобы ничего не потерять, разложим деньги по времени. Затраты возникают на этапе планирования и разработки, во время теста и после внедрения выбранного решения. Выгоды возникают только **во время теста и после внедрения** — до запуска варианта их не существует.

```python
phases = ['Планирование\nи разработка', 'Тест\n(T недель)', 'Решение\n(внедрить / нет)', 'После внедрения\n(H − T недель)']

fig, ax = plt.subplots(figsize=(12.5, 3.6))
x = np.arange(len(phases))

ax.hlines([1, 0], -0.42, len(phases) - 0.58, color=GRAY, lw=1.2, alpha=0.5)
ax.text(-0.42, 1.05, 'Затраты', va='bottom', fontsize=11, color=INK, fontweight='bold')
ax.text(-0.42, 0.05, 'Выгоды', va='bottom', fontsize=11, color=INK, fontweight='bold')

for xi in x:                                    # затраты есть на всех фазах
    ax.plot(xi, 1, 'o', ms=12, color=GRAY, zorder=3)
for xi in [1, 3]:                               # выгоды — только на тесте и после внедрения
    ax.plot(xi, 0, 'o', ms=12, color=GREEN, zorder=3)

ax.annotate('', xy=(len(phases) - 0.52, -0.55), xytext=(-0.42, -0.55),
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.4))
ax.text((len(phases) - 1) / 2, -0.82, 'время →', ha='center', fontsize=10, color=INK)

ax.set_xticks(x, phases, fontsize=9.5)
ax.set_ylim(-1.05, 1.5)
ax.set_yticks([])
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.set_title('Когда возникают затраты и выгоды', loc='left')
plt.tight_layout()
plt.show()
```

**Два типа затрат.**

1. **Фиксированные** — возникают с вероятностью $P = 1$ независимо от результата теста (разработка, подписка на сервис тестирования). «Фиксированные» — про вероятность, а не про неизменность суммы: итоговая сумма может зависеть от длительности, но наступает гарантированно.
2. **Связанные с риском** — возникают только при определённом исходе. Учтённые в деньгах с вероятностью, мы называем их **затратами с поправкой на риск** (risk-adjusted). Выгоды в этой модели всегда связаны с риском: если выгоду можно получить независимо от результата теста, она к тесту не относится.

**Стоимость отмены решения.** Отдельно считаем **откат** — что делать, если внедрённый вариант окажется плохим. Тривиальные изменения текста откатить дёшево; решения, затрагивающие партнёров или инфраструктуру, практически необратимы. Чем дороже откат, тем строже должен быть порог внедрения.

**Типичная ошибка.** Считать только «стоимость теста» и забыть, что внедрение, поддержка и откат — тоже деньги, которые платятся не всегда, а с вероятностью решения.

### 1.3 Кейс и класс `BusinessCase`

Синтетический кейс: интернет-магазин тестирует изменение checkout-страницы, которое влияет на вероятность покупки. Допущение на всю встречу: **относительный рост конверсии = относительный рост выручки** (для checkout это разумно: корзина уже сформирована, мы лишь доводим пользователя до заказа).

| Параметр | Значение |
| :--- | :--- |
| Базовая конверсия | 5% |
| Пользователей в неделю | 50 000 |
| Недельная выручка | 3 000 000 ₽ |
| Фиксированные затраты на тест | 180 000 ₽ |
| Стоимость внедрения | 120 000 ₽ |
| Стоимость отката | 250 000 ₽ |
| Дополнительная поддержка | 20 000 ₽/мес |
| Горизонт планирования | 12 месяцев от старта теста |

```python
@dataclass
class BusinessCase:
    """Параметры бизнес-кейса."""
    baseline_cr: float = 0.05          # базовая конверсия
    weekly_users: int = 50_000         # пользователей в неделю
    weekly_revenue: float = 3_000_000  # недельная выручка, ₽
    fixed_test_cost: float = 180_000   # фиксированные затраты на тест, ₽
    implementation_cost: float = 120_000  # стоимость внедрения, ₽
    rollback_cost: float = 250_000     # стоимость отката ошибочного решения, ₽
    monthly_maintenance: float = 20_000   # поддержка, ₽/мес
    horizon_months: float = 12         # горизонт планирования, мес

    @property
    def horizon_weeks(self) -> float:
        return self.horizon_months * 52 / 12

    @property
    def maintenance_cost(self) -> float:
        return self.monthly_maintenance * self.horizon_months

    @property
    def break_even(self) -> float:
        """Эффект, при котором выручка за горизонт покрывает затраты на тест, внедрение и поддержку."""
        costs = self.fixed_test_cost + self.implementation_cost + self.maintenance_cost
        return costs / (self.weekly_revenue * self.horizon_weeks)


case = BusinessCase()
print(f"Горизонт: {case.horizon_weeks:.0f} недель")
print(f"Поддержка за горизонт: {spaced(case.maintenance_cost)} ₽")
```

### 1.4 Что можно крутить: $\alpha$ и $n$

У теста есть пять параметров. Два статистических — **порог значимости $\alpha$** и **размер выборки $n$**. Размер выборки задаёт мощность $(1-\beta)$, минимальный надёжно обнаруживаемый эффект MDE и длительность теста (поток пользователей ограничен). Три дизайнерских — нулевая гипотеза и набор метрик, число вариантов, схема мониторинга.

Нулевая гипотеза, метрики и варианты обычно диктуются внешней информацией о продукте, а последовательная схема разобрана в отдельной лекции. Поэтому **сосредоточимся на двух параметрах: $n$ и $\alpha$**.

**Типичная ошибка.** Менять $n$, забыв, что оно тянет за собой два эффекта сразу: маленькая выборка дешевле, но слабее; большая выборка мощнее, но **откладывает внедрение**. Оптимум приходится искать перебором, потому что эти эффекты тянут в разные стороны.

### 1.5 Распределение ожидаемых эффектов $\pi(\theta)$

**Интуиция.** Истинный эффект $\theta$ заранее неизвестен. Вместо «точки» зададим **априорное распределение** $\pi(\theta)$: какие значения эффекта мы считаем более или менее правдоподобными до эксперимента. Обычно его строят по истории похожих тестов.

**Три предостережения.**

1. Это априорное распределение, но сам анализ остаётся частотным: $\pi(\theta)$ используется только для выбора $\alpha$ и $n$ на этапе дизайна.
2. Равномерное распределение — не «я ничего не знаю», а сильное утверждение, что все эффекты равновероятны.
3. Проблема референтного класса: слишком узкий круг похожих тестов даёт мало данных, слишком широкий — неточную оценку для конкретного случая.

**Пример на кейсе.** Три сценария ожиданий, различающихся формой распределения:

```python
x_eff = np.linspace(-0.22, 0.22, 600)
scenarios = [
    ('Консервативный\n(изменение CTA-кнопки)', 0.00, 0.04),
    ('Умеренный\n(редизайн корзины)', 0.02, 0.06),
    ('Оптимистичный\n(убрать поле купона)', 0.033, 0.015),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, loc, scale) in zip(axes, scenarios):
    prior = norm(loc, scale)
    y = prior.pdf(x_eff)
    ax.plot(x_eff * 100, y, color=GRAY, linewidth=1.5)
    ax.fill_between(x_eff * 100, y, where=x_eff >= 0, color=GREEN, alpha=0.35,
                    label=f'θ > 0: {prior.sf(0):.1%}')
    ax.fill_between(x_eff * 100, y, where=x_eff <= 0, color=RED, alpha=0.35,
                    label=f'θ < 0: {prior.cdf(0):.1%}')
    ax.axvline(0, color=GRAY, linestyle=':', linewidth=1.5)
    ax.set_title(f'{title}\nμ = {loc:.1%}, σ = {scale:.1%}', loc='left', fontsize=12)
    ax.set_xlabel('Относительный эффект θ, %')
    ax.set_xlim(-22, 22)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=10)
axes[0].set_ylabel('Плотность вероятности')
plt.tight_layout()
plt.show()
```

Чем правее и уже распределение, тем выше $P(\theta > 0)$: у консервативного сценария она $50\%$, у умеренного $63.1\%$, у оптимистичного $98.6\%$. Выбор распределения сильно влияет на оценку рисков — ниже проверим чувствительность результата к нему.

**Априорное распределение нашего кейса.** Для checkout-страницы считаем, что эффект распределён нормально вокруг нуля с $\sigma = 3\%$: положительные и отрицательные эффекты заранее одинаково возможны.

```python
effect_grid = np.linspace(-0.10, 0.10, 1000)  # сетка возможных истинных эффектов
prior_mu = 0.0
prior_sigma = 0.03

density = norm.pdf(effect_grid, loc=prior_mu, scale=prior_sigma)
density /= np.trapezoid(density, effect_grid)  # сетка обрезает хвосты, нормируем заново
```

### 1.6 Точка безубыточности $\theta_{BE}$

**Интуиция.** Запуск варианта требует фиксированных денег: тест + внедрение + поддержка на весь горизонт. Точка безубыточности $\theta_{BE}$ — минимальный относительный прирост выручки, который отбивает эти деньги. Она почти всегда ниже $1\%$, потому что затраты на большинство тестов малы по сравнению с их влиянием за время использования.

**Формула.** В точке безубыточности выгода равна затратам ($ROI = 0$):

$$\theta_{BE} \times Revenue = Fixed\,Costs \qquad\Rightarrow\qquad
\theta_{BE} = \frac{Fixed\,Costs}{Revenue} = \frac{C_{test} + C_{impl} + C_{maint}}{R_{week} \cdot H}$$

где $R_{week}$ — недельная выручка, $H$ — горизонт в неделях. Точка равна нулю только в редком случае, когда побочные эффекты изменения в точности компенсируют затраты.

**Пример на кейсе.**

```python
fixed_costs = case.fixed_test_cost + case.implementation_cost + case.maintenance_cost
break_even = case.break_even
print(f"Суммарные фиксированные издержки: {spaced(fixed_costs)} ₽")
print(f"Fixed-cost break-even point: {break_even:.3%} относительного uplift")
```

**Типичная ошибка.** Забыть поддержку: $20\,000$ ₽/мес × $12$ мес = $240\,000$ ₽ — **больше, чем сам тест** ($180\,000$ ₽). В безубыточность входит вся фиксированная сумма, а не только разработка.

Наложим точку безубыточности на распределение ожиданий:

```python
prior = norm(prior_mu, prior_sigma)
x = effect_grid * 100

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(x, density, color=GRAY, linewidth=2)
ax.fill_between(x, density, where=effect_grid >= break_even, color=GREEN, alpha=0.4,
                label=f'Прибыль, θ ≥ {break_even:.2%}: вероятность {prior.sf(break_even):.1%}')
ax.fill_between(x, density, where=(effect_grid > 0) & (effect_grid < break_even),
                color='#DD8452', alpha=0.6,
                label=f'Эффект есть, но не окупается: {prior.cdf(break_even) - prior.cdf(0):.1%}')
ax.fill_between(x, density, where=effect_grid <= 0, color=RED, alpha=0.4,
                label=f'Негативный эффект: {prior.cdf(0):.1%}')
ax.axvline(break_even * 100, color=GREEN, linestyle='--', linewidth=2, label='Точка безубыточности')
ax.set_title('Больше половины априорной вероятности не окупает затраты', loc='left')
ax.set_xlabel('Относительный эффект θ, %')
ax.set_ylabel('Плотность вероятности')
ax.set_xlim(-10, 10)
ax.set_ylim(bottom=0)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

**Как читать график.** Зелёная зона справа от $\theta_{BE}$ — варианты, которые окупаются. Оранжевая полоса $0 < \theta < \theta_{BE}$ — эффект положительный, но **не окупается** (внедрять такой вариант — убыток). Красная зона — отрицательные эффекты. Важно: точка безубыточности определяет, что такое «успех» — не «$\theta > 0$», а «$\theta > \theta_{BE}$».

### 1.7 Затраты и выгоды с поправкой на риск: $Cost_{RA}$ и $Gain_{RA}$

**Интуиция.** Ожидаемая ценность теста — это интеграл по всем возможным эффектам $\theta$, взвешенный вероятностью $\pi(\theta)$ и вероятностью того, что мы этот эффект внедрим. Каждый эффект даёт денежный поток во время теста и (если внедрили) после теста.

Обозначения для денежных потоков:

- $Rev_{test}$ — выручка, затронутая во время теста: половина недельной выручки × длительность теста $T$ (вариант видит только половину трафика).
- $Rev_{post}$ — выручка после теста: недельная выручка × оставшееся время $H - T$. Каждая неделя теста **съедает** неделю внедрения — именно это делает слишком длинные тесты невыгодными.
- $Cost_d(\theta)$ — затраты решения при внедрении: внедрение + поддержка, а если $\theta$ ниже безубыточности — ещё и откат.
- $Gain_d$ — разовая выгода от внедрения; в нашем кейсе её нет.
- $POW(\alpha; \theta)$ — вероятность внедрения при истинном эффекте $\theta$ (введём её в 1.8).

**Формула затрат.**

$$Cost_{RA} = Cost_f + \int_{a}^{b} f_c(\theta)\, d\theta$$

$$f_c(\theta) = \pi(\theta)\cdot\Big(\underbrace{\max(-\theta \cdot Rev_{test},\,0)}_{\text{потеря во время теста}}
+ \big[\underbrace{\max(-\theta \cdot Rev_{post},\,0)}_{\text{потеря после внедрения}}
+ \underbrace{Cost_d(\theta)}_{\text{затраты решения}}\big]\cdot \underbrace{POW(\alpha;\theta)}_{\text{шанс внедрения}}\Big)$$

**Формула выгод.**

$$Gain_{RA} = \int_{a}^{b} f_g(\theta)\, d\theta$$

$$f_g(\theta) = \pi(\theta)\cdot\Big(\underbrace{\max(\theta \cdot Rev_{test},\,0)}_{\text{выгода во время теста}}
+ \big[\underbrace{\max(\theta \cdot Rev_{post},\,0)}_{\text{выгода после внедрения}}
+ \underbrace{Gain_d}_{\text{разовая выгода}}\big]\cdot POW(\alpha;\theta)\Big)$$

**Разбор каждого члена (интуиция, а не просто формула).**

- $\max(\cdot, 0)$ и $\max(-\cdot, 0)$ — мы **не обязаны** получать убыток от плохого варианта целиком: во время теста плохой вариант видит только половина трафика, а после теста плохой вариант чаще всего просто не внедряется. Поэтому выгода отсекается снизу нулём, а потеря — сверху.
- Умножение послевнедренческих слагаемых на $POW(\alpha;\theta)$ — выгоды и затраты после теста возникают **только если мы решили внедрить**.
- Взвешивание на $\pi(\theta)$ — редкие, но крупные эффекты учитываются с малой вероятностью; частые и маленькие — с большой.
- $Cost_f$ в $Cost_{RA}$ — фиксированные затраты на тест платим всегда, поэтому они вне интеграла.

**Типичная ошибка.** Считать $Rev_{post}$ константой и «расти» выборку бесконечно. $Rev_{post}$ **убывает** с ростом $T$: каждая неделя теста — это отложенная неделя прибыли. Без этого члена оптимальная выборка всегда стремилась бы к бесконечности.

### 1.8 Вероятность внедрения $POW(\alpha; \theta)$

**Интуиция.** Решение привязываем не к проверке нулевого эффекта, а к проверке «эффект выше точки безубыточности». Новый вариант достоин внедрения, только если есть статистические основания считать, что он **экономически оправдан**:

$$H_0:\ \theta \le \theta_{BE}, \qquad H_1:\ \theta > \theta_{BE}$$

**Формула.** Пусть $\hat\delta$ — оценка абсолютной разницы конверсий, $\delta_{BE} = p \cdot \theta_{BE}$ — разница, соответствующая безубыточности, $SE$ — стандартная ошибка. Внедряем, если:

$$\frac{\hat\delta - \delta_{BE}}{SE} > z_{1-\alpha} \quad\Rightarrow\quad \hat\delta > \delta_{BE} + z_{1-\alpha}\cdot SE$$

Здесь $\alpha$ **односторонний** и равен вероятности внедрения при $\theta = \theta_{BE}$. Привычный двусторонний уровень $0.05$ соответствует одностороннему $0.025$.

**Типичная ошибка.** Читать $\alpha$ как «вероятность ошибиться в любом решении». Это неверно: $\alpha$ — вероятность внедрения **при эффекте ровно на границе безубыточности**. Доля ошибочных внедрений зависит ещё и от того, какие эффекты реально встречаются (вернёмся к этому в части 2).

```python
def implementation_probability(
    p_control: float,
    relative_effect: np.ndarray,
    break_even_effect: float,
    n_per_group: np.ndarray,
    alpha: np.ndarray = 0.05
) -> np.ndarray:
    """Вероятность внедрения при истинном эффекте relative_effect; аргументы могут быть массивами."""
    p_true = np.clip(p_control * (1 + relative_effect), 1e-6, 1 - 1e-6)

    delta_true = p_true - p_control            # истинная разница конверсий
    delta_be = p_control * break_even_effect   # разница, соответствующая безубыточности
    se = np.sqrt(p_control * (1 - p_control) / n_per_group + p_true * (1 - p_true) / n_per_group)

    z_crit = norm.ppf(1 - alpha)               # порог решения
    return norm.sf(z_crit - (delta_true - delta_be) / se)
```

```python
fig, ax = plt.subplots(figsize=(14, 5.5))
for n_val, color, style in [(2_000, RED, '-'), (20_000, BLUE, '--'), (200_000, GREEN, '-.')]:
    probability = implementation_probability(case.baseline_cr, effect_grid, break_even, n_val)
    ax.plot(effect_grid * 100, probability * 100, color=color, linestyle=style, linewidth=2.5,
            label=f'n = {spaced(n_val)}')
ax.axvline(break_even * 100, color=GREEN, linestyle=':', linewidth=2, label=f'Точка безубыточности: {break_even:.2%}')
ax.axhline(5, color=GRAY, linestyle=':', linewidth=1.5, label='α = 0.05')
ax.set_title('Большая выборка делает решение почти пороговым', loc='left')
ax.set_xlabel('Истинный относительный эффект, %')
ax.set_ylabel('Вероятность внедрения, %')
ax.set_xlim(-10, 10)
ax.set_ylim(0, 100)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

**Как читать график.** Все кривые проходят через одну точку: при $\theta = \theta_{BE}$ вероятность внедрения равна $\alpha$ при любом $n$. Чем больше выборка, тем круче кривая: хорошие варианты внедряются почти наверняка, плохие — почти никогда. Маленькая выборка «смазывает» решение, и даже сильные эффекты внедряются редко.

### 1.9 Отношение выгоды к риску и чистая прибыль: `evaluate_design`

**Интуиция.** После интегрирования получаем два числа, отвечающих на разные вопросы:

- **Чистая прибыль** $Net = Gain_{RA} - Cost_{RA}$ — сколько в среднем заработаем на этом решении. Смотрим, когда решение одно и важен абсолютный результат.
- **Отношение выгоды к риску** $RR = Gain_{RA} / Cost_{RA}$ — сколько рублей выгоды на рубль затрат. Смотрим, когда тестов много, ресурсы ограничены и нужна лучшая отдача на вложенный рубль.

$RR > 1$ означает, что ожидаемая выгода превышает ожидаемые затраты (и $ROI = RR - 1$). Георгиев и многие инструменты используют обратное отношение (risk/reward, где чем меньше, тем лучше); мы смотрим на вариант «чем больше, тем лучше».

**Типичная ошибка.** Думать, что максимум $RR$ и максимум $Net$ совпадают. Они выбирают **разные** дизайны: отношение поощряет экономию затрат даже ценой абсолютной прибыли.

```python
def evaluate_design(
    case: BusinessCase,
    effect_grid: np.ndarray,
    density: np.ndarray,
    n_per_group: np.ndarray,
    alpha: np.ndarray
) -> dict:
    """Выгоды и затраты с поправкой на риск.

    n_per_group и alpha можно передать массивами: их формы должны
    согласовываться с осью эффектов, которая всегда идёт последней.
    """
    n_per_group = np.asarray(n_per_group, dtype=float)
    probability = implementation_probability(case.baseline_cr, effect_grid, case.break_even, n_per_group, alpha)

    test_weeks = 2 * n_per_group / case.weekly_users
    post_weeks = np.maximum(case.horizon_weeks - test_weeks, 0)  # длинный тест откладывает внедрение

    # Половина пользователей видит новый вариант во время теста
    test_value = case.weekly_revenue * effect_grid * test_weeks * 0.5
    post_value = case.weekly_revenue * effect_grid * post_weeks * probability

    # Затраты только при внедрении; откат нужен, если вариант не окупается
    decision_costs = (case.implementation_cost + case.maintenance_cost
                      + case.rollback_cost * (effect_grid < case.break_even)) * probability

    gains = (np.maximum(test_value, 0) + np.maximum(post_value, 0)) * density
    losses = (np.maximum(-test_value, 0) + np.maximum(-post_value, 0) + decision_costs) * density

    expected_gain = np.trapezoid(gains, effect_grid, axis=-1)
    expected_loss = case.fixed_test_cost + np.trapezoid(losses, effect_grid, axis=-1)

    return {
        'probability': probability,
        'gains': gains,
        'losses': losses,
        'expected_gain': expected_gain,
        'expected_loss': expected_loss,
        'rr': expected_gain / expected_loss,
        'net': expected_gain - expected_loss,
    }


def describe_design(result: dict) -> pd.Series:
    """Итоговые показатели дизайна."""
    return pd.Series({
        'Выгоды, ₽': result['expected_gain'],
        'Затраты, ₽': result['expected_loss'],
        'Чистая прибыль, ₽': result['net'],
        'RR': result['rr'],
    })
```

Сравним два дизайна с одинаковым $\alpha$, но разной выборкой:

```python
def plot_risk_adjusted(designs: list) -> None:
    """Вероятность внедрения и подынтегральные выгоды/затраты для списка дизайнов (n, alpha)."""
    results = [evaluate_design(case, effect_grid, density, n, alpha) for n, alpha in designs]
    limit = max(max(r['gains'].max(), r['losses'].max()) for r in results) * 1.05
    x = effect_grid * 100

    fig, axes = plt.subplots(2, len(designs), figsize=(7 * len(designs), 7.5),
                             sharex=True, sharey='row', squeeze=False)
    for column, ((n, alpha), result) in enumerate(zip(designs, results)):
        top, bottom = axes[0, column], axes[1, column]
        top.plot(x, result['probability'] * 100, color=BLUE, linewidth=2.5)
        top.axvline(break_even * 100, color=GRAY, linestyle=':', linewidth=1.5)
        top.set_ylim(0, 100)
        top.set_title(f'n = {spaced(n)}, α = {alpha}, тест {2 * n / case.weekly_users:.1f} нед.', loc='left')

        bottom.fill_between(x, result['gains'], color=GREEN, alpha=0.5,
                            label=f"Выгоды: {spaced(result['expected_gain'] / 1000)} тыс. ₽")
        bottom.fill_between(x, -result['losses'], color=RED, alpha=0.5,
                            label=f"Затраты без фиксированных: {spaced((result['expected_loss'] - case.fixed_test_cost) / 1000)} тыс. ₽")
        bottom.axhline(0, color=GRAY, linewidth=1)
        bottom.axvline(break_even * 100, color=GRAY, linestyle=':', linewidth=1.5)
        bottom.set_ylim(-0.3 * limit, limit)
        bottom.set_xlabel('Истинный относительный эффект θ, %')
        bottom.legend(loc='upper left', fontsize=10)

    axes[0, 0].set_ylabel('Вероятность внедрения, %')
    axes[1, 0].set_ylabel('Вклад в ожидание, ₽ на единицу θ')
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e6:.1f} млн'))
    plt.xlim(-10, 10)
    plt.tight_layout()
    plt.show()


designs_example = [(5_000, 0.05), (50_000, 0.05)]
plot_risk_adjusted(designs_example)
pd.DataFrame({f'n = {n}': describe_design(evaluate_design(case, effect_grid, density, n, alpha))
              for n, alpha in designs_example}).round(2)
```

**Как читать графики.** Верхний ряд — вероятность внедрения. Нижний — вклад каждого $\theta$ в ожидание, уже взвешенный $\pi(\theta)$: площадь зелёной области равна выгодам, красной — затратам без $180$ тыс. ₽ на тест.

- При $n = 5\,000$ тест почти ничего не решает: даже сильные эффекты внедряются редко, фиксированные затраты не окупаются — убыток $\approx 37$ тыс. ₽, $RR = 0.85$.
- При $n = 50\,000$ тот же уровень значимости даёт прибыль $\approx 408$ тыс. ₽ и $RR = 2.54$.

### 1.10 Проверка формул симуляцией и декомпозиция прибыли

Формулы можно проверить независимо: разыграем весь процесс целиком — случайный истинный эффект из априорного распределения, конверсии в группах, правило внедрения, деньги.

```python
def simulate_design(case: BusinessCase, n_per_group: int, alpha: float,
                    n_sims: int = 400_000, seed: int = 42) -> tuple:
    """Оценивает среднюю прибыль дизайна полным разыгрыванием экспериментов."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(prior_mu, prior_sigma, n_sims)
    theta = theta[np.abs(theta) <= effect_grid.max()]  # та же область интегрирования, что и в формулах

    p_control = case.baseline_cr
    p_treatment = p_control * (1 + theta)
    control = rng.binomial(n_per_group, p_control, theta.size) / n_per_group
    treatment = rng.binomial(n_per_group, p_treatment) / n_per_group

    se = np.sqrt(control * (1 - control) / n_per_group + treatment * (1 - treatment) / n_per_group)
    ship = (treatment - control - p_control * case.break_even) / se > norm.ppf(1 - alpha)

    test_weeks = 2 * n_per_group / case.weekly_users
    post_weeks = case.horizon_weeks - test_weeks
    profit = case.weekly_revenue * theta * test_weeks * 0.5 - case.fixed_test_cost
    profit += ship * (case.weekly_revenue * theta * post_weeks - case.implementation_cost
                      - case.maintenance_cost - case.rollback_cost * (theta < case.break_even))
    return profit.mean(), profit.std() / np.sqrt(theta.size)


for n, alpha in designs_example:
    formula = evaluate_design(case, effect_grid, density, n, alpha)['net']
    simulated, error = simulate_design(case, n, alpha)
    print(f"n = {spaced(n)}: формула {spaced(formula)} ₽, симуляция {spaced(simulated)} ± {spaced(1.96 * error)} ₽")
```

Совпадение формулы и симуляции (в пределах $1.96$ стандартных ошибок) — это самопроверка: интегралы и «физический» процесс описывают одно и то же.

Теперь разложим прибыль одного дизайна на слагаемые — waterfall:

```python
def decompose_design(case, n, alpha):
    """Подынтегральные слагаемые прибыли одного дизайна (те же интегралы, что в evaluate_design)."""
    probability = implementation_probability(case.baseline_cr, effect_grid, case.break_even, n, alpha)
    test_weeks = 2 * n / case.weekly_users
    post_weeks = max(case.horizon_weeks - test_weeks, 0)
    r_test = case.weekly_revenue * test_weeks * 0.5
    r_post = case.weekly_revenue * post_weeks
    decision_costs = case.implementation_cost + case.maintenance_cost \
        + case.rollback_cost * (effect_grid < case.break_even)
    return {
        'gain_test': np.trapezoid(np.maximum(r_test * effect_grid, 0) * density, effect_grid),
        'gain_post': np.trapezoid(np.maximum(r_post * effect_grid, 0) * probability * density, effect_grid),
        'loss_test': np.trapezoid(np.maximum(-r_test * effect_grid, 0) * density, effect_grid),
        'loss_post': np.trapezoid(np.maximum(-r_post * effect_grid, 0) * probability * density, effect_grid),
        'loss_decision': np.trapezoid(decision_costs * probability * density, effect_grid),
        'cost_test': case.fixed_test_cost,
    }


d = decompose_design(case, 50_000, 0.05)
d['net'] = d['gain_test'] + d['gain_post'] - d['loss_test'] - d['loss_post'] - d['loss_decision'] - d['cost_test']

labels = ['Выгода во время теста', 'Выгода после внедрения', 'Потеря во время теста',
          'Потеря после внедрения', 'Внедрение/откат (ожид.)', 'Тест (фикс.)', 'ИТОГО']
values = [d['gain_test'], d['gain_post'], -d['loss_test'], -d['loss_post'],
          -d['loss_decision'], -d['cost_test'], d['net']]

running = 0.0
bottoms, tops = [], []
for v in values[:-1]:
    tops.append(max(running + v, running))
    bottoms.append(min(running + v, running))
    running += v
tops.append(max(0, running)); bottoms.append(0)

fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(labels))
bar_colors = [GREEN if v >= 0 else RED for v in values]
bar_colors[-1] = BLUE

for xi, (v, b, t, c) in enumerate(zip(values, bottoms, tops, bar_colors)):
    ax.bar(xi, (t - b) / 1e3, bottom=b / 1e3, color=c, width=0.62,
           edgecolor='white', linewidth=0.5, zorder=3)
    label_y = t / 1e3 if v >= 0 else b / 1e3
    ax.text(xi, label_y + (0.10 if v >= 0 else -0.16), f"{v/1e3:+,.0f}",
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=9.5,
            fontweight='bold' if xi == len(values) - 1 else 'normal', color=c)

cum = 0.0
for xi, v in enumerate(values[:-1]):
    cum += v
    ax.plot([xi + 0.31, xi + 0.69], [cum / 1e3, cum / 1e3], color='#999999', lw=1, zorder=2)

ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x, labels, rotation=12, ha='right')
ax.set_ylabel('тыс ₽')
ax.set_title(f"Из чего складывается {spaced(d['net'])} ₽ чистой прибыли (n = 50 000, α = 0.05)", loc='left')
ax.set_ylim(min(bottoms) / 1e3 - 60, max(tops) / 1e3 + 80)
plt.tight_layout()
plt.show()
```

**Как читать waterfall.** Во время теста мы трогаем половину трафика, а эффект в среднем нулевой, поэтому выгода и потеря почти симметричны и гасят друг друга. Главный вклад — **выгода после внедрения**: она взвешена вероятностью внедрения, то есть тест пропускает только хвост хороших исходов и отсекает «мусор». Вычитаем фиксированные $180$ тыс. ₽ на тест и ожидаемые затраты на внедрение/откат — остаётся чистая прибыль.

---

## Часть 2. Оптимальные $\alpha$ и $n$

### 2.1 Сетка дизайнов и два максимума

Переберём все сочетания $\alpha$ и $n$ и найдём максимумы. Две типичные ловушки поиска:

- **Край сетки.** Если оптимум лежит на границе сетки, настоящего оптимума мы не нашли. Поэтому сетка по $n$ идёт до $600$ тыс. на группу — почти до половины горизонта.
- **Критерий.** Максимум $RR$ и максимум $Net$ в общем случае не совпадают, поэтому смотрим на оба.

```python
alpha_grid = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8])
n_grid = np.unique(np.round(np.geomspace(1_000, 600_000, 60), -2)).astype(int)


def design_grid(case: BusinessCase, effects: np.ndarray, weights: np.ndarray) -> dict:
    """RR и прибыль для всех сочетаний alpha и n."""
    return evaluate_design(case, effects, weights, n_grid[None, :, None], alpha_grid[:, None, None])


def best_design(grid: dict, metric: str) -> tuple:
    """alpha и n с максимальным значением метрики."""
    row, column = np.unravel_index(np.argmax(grid[metric]), grid[metric].shape)
    return alpha_grid[row], n_grid[column], grid['rr'][row, column], grid['net'][row, column]


grid = design_grid(case, effect_grid, density)
results_df = pd.DataFrame([best_design(grid, 'rr'), best_design(grid, 'net')],
                          index=['Максимум RR', 'Максимум прибыли'],
                          columns=['α', 'n на группу', 'RR', 'Чистая прибыль, ₽'])
results_df['Длительность, дней'] = (results_df['n на группу'] * 2 / case.weekly_users * 7).round(1)
results_df.round(3)
```

### 2.2 Тепловые карты $\alpha \times n$

```python
def plot_design_maps(case: BusinessCase, grid: dict, marks: dict = None) -> None:
    """Тепловые карты RR и чистой прибыли на сетке α×n."""
    n_pos = np.arange(n_grid.size)
    a_pos = np.arange(alpha_grid.size)

    panels = [
        (grid['rr'], 1.0, 'RR — отношение выгоды к риску', 'rr'),
        (grid['net'] / 1e6, 0.0, 'Чистая прибыль, млн ₽', 'net'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (values, center, label, metric) in zip(axes, panels):
        vmin, vmax = values.min(), values.max()
        vmin = min(vmin, center - 1e-6)
        vmax = max(vmax, center + 1e-6)
        mesh = ax.pcolormesh(n_grid, a_pos, values, cmap='RdBu_r', shading='nearest',
                             norm=TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax))
        cb = fig.colorbar(mesh, ax=ax, label=label, fraction=0.046, pad=0.02)
        cb.set_ticks([vmin, center, vmax])
        ax.contour(n_grid, a_pos, values, levels=[center], colors='black', linewidths=1.5)

        best_alpha, best_n, best_rr, best_net = best_design(grid, metric)
        best_row = np.searchsorted(alpha_grid, best_alpha)
        best_col = np.searchsorted(n_grid, best_n)
        ax.scatter(best_n, best_row, marker='*', s=420, color='#FFD166', edgecolor='black', zorder=5)
        ax.annotate(f'α = {best_alpha}, n = {spaced(best_n)}\n{values[best_row, best_col]:.2f}',
                    xy=(best_n, best_row), xytext=(best_n * 1.35, best_row + 1.6),
                    fontsize=9.5, color='black', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFD166', ec='black', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))

        for name, (mark_alpha, mark_n) in (marks or {}).items():
            ax.scatter(mark_n, np.searchsorted(alpha_grid, mark_alpha), marker='o', s=110,
                       color='white', edgecolor='black', zorder=4, label=name)

        ax.set_xscale('log')
        ax.set_xlabel('Размер группы n, логарифмическая шкала')
        ax.grid(False)
        if marks:
            ax.legend(loc='lower left', fontsize=9)

    axes[0].set_yticks(a_pos[::2], [f'{a:g}' for a in alpha_grid[::2]])
    axes[0].set_ylabel('Уровень значимости α, односторонний')
    axes[1].set_ylabel('')
    plt.tight_layout()
    plt.show()


# Стандартный дизайн: 80% мощности для эффекта 3% (одна σ априорного распределения)
mde_relative = 0.03
delta = case.baseline_cr * mde_relative
variance = case.baseline_cr * (1 - case.baseline_cr)
n_standard = int(np.ceil(2 * variance * (norm.ppf(0.975) + norm.ppf(0.8)) ** 2 / delta ** 2))

plot_design_maps(case, grid, {f'Стандартный дизайн: n = {spaced(n_standard)}': (0.025, n_standard)})
```

**Как читать карты.** Единая дивергентная палитра: тёплый цвет — хорошо ($RR > 1$, прибыль $> 0$), холодный — плохо. Чёрная линия — граница безубыточности ($RR = 1$ / $Net = 0$). Жёлтая звезда — оптимум по соответствующему критерию, белая точка — стандартный дизайн.

### 2.3 Почему максимумы не совпадают и когда $\alpha > 0.05$ оправдан

Оптимум по $RR$ даёт $\alpha = 0.2$ и $\approx 106$ тыс. на группу; оптимум по прибыли — $\alpha = 0.5$ и $\approx 147$ тыс. на группу.

- **$RR$ поощряет экономию затрат.** Меньшая выборка → короче тест → меньше отложенная выгода и меньше затрат, даже если абсолютная прибыль чуть ниже. Если тестов много, а ресурсов мало, смотрим на $RR$.
- **$Net$ поощряет абсолютный результат.** Если важен результат конкретного решения, смотрим на прибыль — и тогда оптимальный $\alpha$ ещё выше.

Почему $\alpha > 0.05$ вообще оправдан? Потому что порог сравнивается **не с нулём, а с точкой безубыточности** $\theta_{BE}$. Высокий $\alpha$ означает готовность чаще внедрять варианты, которые лишь «выглядят» выше безубыточности. В нашем кейсе цена ошибки мала относительно выручки за год, поэтому оптимальный порог ослаблен. Для почти необратимых решений порог станет строже.

Модель нейтральна к риску: она смотрит только на среднюю прибыль, а не на разброс исходов отдельного решения. Поэтому $\alpha = 0.5$ здесь — не «рецепт на все случаи», а следствие конкретной экономики.

### 2.4 Чувствительность к ожиданиям и затратам

**Интуиция.** «Оптимум» верен настолько, насколько верны входы: распределение ожиданий, затраты, горизонт. Поэтому любой вывод нужно проверять на чувствительность.

<details>
<summary><b>Интерактив: покрутите бизнес-параметры</b></summary>

```python
def optimum_demo(prior_mean: float = 0.0, prior_std: float = 3.0,
                 horizon_months: int = 12, rollback: int = 250) -> None:
    """Пересчитывает карты дизайнов для других априорных ожиданий и бизнес-параметров (проценты и тыс. ₽)."""
    custom_case = BusinessCase(horizon_months=horizon_months, rollback_cost=rollback * 1_000)
    effects = np.linspace(-0.15, 0.15, 400)
    weights = norm.pdf(effects, prior_mean / 100, prior_std / 100)
    weights /= np.trapezoid(weights, effects)
    plot_design_maps(custom_case, design_grid(custom_case, effects, weights))


try:
    interact(optimum_demo,
             prior_mean=FloatSlider(value=0, min=-2, max=2, step=0.25, description='μ, %'),
             prior_std=FloatSlider(value=3, min=1, max=6, step=0.5, description='σ, %'),
             horizon_months=IntSlider(value=12, min=3, max=36, step=3, description='Горизонт, мес'),
             rollback=IntSlider(value=250, min=0, max=2000, step=50, description='Откат, тыс ₽'))
except Exception:
    print('Виджеты недоступны — показываю базовый сценарий.')
    optimum_demo()
```

</details>

Как оптимум реагирует на входы:

```python
sensitivity = []
for name, mean, std, months, rollback in [
    ('Базовый кейс', 0.0, 0.03, 12, 250_000),
    ('Ждём ухудшение: μ = −1%', -0.01, 0.03, 12, 250_000),
    ('Ждём улучшение: μ = +1%', 0.01, 0.03, 12, 250_000),
    ('Узкие ожидания: σ = 1.5%', 0.0, 0.015, 12, 250_000),
    ('Горизонт 36 месяцев', 0.0, 0.03, 36, 250_000),
    ('Дорогой откат: 2 млн ₽', 0.0, 0.03, 12, 2_000_000),
]:
    custom_case = BusinessCase(horizon_months=months, rollback_cost=rollback)
    effects = np.linspace(-0.15, 0.15, 600)
    weights = norm.pdf(effects, mean, std)
    weights /= np.trapezoid(weights, effects)
    best_alpha, best_n, best_rr, best_net = best_design(design_grid(custom_case, effects, weights), 'net')
    sensitivity.append({'Сценарий': name, 'α': best_alpha, 'n на группу': best_n,
                        'Недель': round(2 * best_n / custom_case.weekly_users, 1),
                        'Чистая прибыль, ₽': round(best_net)})

pd.DataFrame(sensitivity).set_index('Сценарий')
```

**Что видим.**

- **Ожидания сдвигают порог.** Если вариант скорее плох ($\mu = -1\%$), оптимальный $\alpha$ падает до $0.4$; если скорее хорош — растёт до $0.55$.
- **Дорогой откат делает порог строже** ($\alpha = 0.35$ при откате $2$ млн ₽).
- **Длинный горизонт разрешает длинный тест:** при $36$ месяцах оптимум — $10$ недель, потому что задержка стоит относительно дешевле.
- **Узкие ожидания требуют большей выборки:** эффекты малы, тест длится $9$ недель, прибыль падает примерно в $5$ раз.

### 2.5 Провокация: что значит $\alpha = 0.5$

**Интуиция.** На первый взгляд $\alpha = 0.5$ абсурдно — кажется, что это подбрасывание монетки. Это **неверное прочтение** $\alpha$:

- $\alpha$ — вероятность внедрения **при эффекте на границе безубыточности**, а не вероятность ошибки в любом решении. Доля ошибочных внедрений зависит ещё и от распределения реальных эффектов.
- Монетка внедряла бы вариант с вероятностью $50\%$ **независимо от эффекта**. Тест с односторонним $\alpha = 0.5$ внедряет вариант, если его **оценка** эффекта положительна, поэтому вероятность внедрения растёт с эффектом.

**Формула.** Сравнение конверсий контрольной ($p_0$) и тестовой ($p_1$) групп:

$$z = \frac{\hat p_1 - \hat p_0}{SE}, \qquad SE = \sqrt{\frac{p_0(1-p_0)}{n} + \frac{p_1(1-p_1)}{n}}$$

Если эффекта нет, $z \sim N(0, 1)$. Если эффект есть, распределение сдвигается на **параметр нецентральности** $NCP = (p_1 - p_0)/SE$ (со знаком). Внедряем при $z > z_{1-\alpha}$, поэтому

$$P(\text{внедрение}) = 1 - \Phi(z_{1-\alpha} - NCP)$$

При $p_1 > p_0$ ($NCP > 0$) вероятность стремится к $1$ — это мощность. При $p_1 < p_0$ ($NCP < 0$) вероятность стремится к $0$ — это шанс внедрить плохой вариант. **Скорость обоих переходов задаёт выборка.**

**Пример на кейсе.** Для сравнения с нулём ($\theta_{BE} = 0$):

```python
effects_table = np.array([-0.05, -0.03, 0.0, 0.03, 0.05])
table = pd.DataFrame(index=[f'{value:+.0%}' for value in effects_table])
table.index.name = 'Истинный эффект'
for n_val in [10_000, 100_000]:
    for alpha_val in [0.05, 0.5]:
        table[f'n = {n_val}, α = {alpha_val}'] = implementation_probability(
            case.baseline_cr, effects_table, 0.0, n_val, alpha_val).round(3)
table
```

Всё зависит от выборки. При $n = 10\,000$ тест слабый: вариант на $3\%$ хуже контроля при $\alpha = 0.5$ внедрится с вероятностью $31\%$. При $n = 100\,000$ эта вероятность падает до $6\%$. При нулевом эффекте вероятность внедрения равна $\alpha$ при любом $n$.

Оценим риск целиком — долю ошибочных внедрений и средний эффект на одно решение, для априорного распределения (без затрат, $\theta_{BE} = 0$):

```python
def rollout_quality(n_per_group: int, alpha: float) -> pd.Series:
    """Доля внедрений, доля ошибочных внедрений и средний эффект для априорного распределения."""
    if alpha is None:  # монетка
        probability = np.full_like(effect_grid, 0.5)
    else:
        probability = implementation_probability(case.baseline_cr, effect_grid, 0.0, n_per_group, alpha)
    p_ship = np.trapezoid(probability * density, effect_grid)
    p_bad_ship = np.trapezoid(probability * density * (effect_grid < 0), effect_grid)
    return pd.Series({
        'Доля внедрений': p_ship,
        'Доля плохих среди внедрённых': p_bad_ship / p_ship,
        'Средний эффект на решение, %': np.trapezoid(effect_grid * probability * density, effect_grid) * 100,
    })


pd.DataFrame({
    'Монетка': rollout_quality(0, None),
    'n = 10 000, α = 0.05': rollout_quality(10_000, 0.05),
    'n = 10 000, α = 0.5': rollout_quality(10_000, 0.5),
    'n = 100 000, α = 0.05': rollout_quality(100_000, 0.05),
    'n = 100 000, α = 0.5': rollout_quality(100_000, 0.5),
}).T.round(3)
```

**Что видим.**

- **Монетка** внедряет половину вариантов, половина внедрённых плоха, средний эффект ноль.
- **Тест с $\alpha = 0.5$** внедряет столько же, но плохих среди внедрённых заметно меньше: $36\%$ при $n = 10\,000$ и $18\%$ при $n = 100\,000$. Средний эффект положителен.
- **Строгий порог** даёт меньше плохих внедрений ($18\%$ и $3\%$), но пропускает хорошие варианты — средний эффект ниже.

Итак, $\alpha = 0.5$ — не монетка, но и не бесплатный подарок: среди внедрённых остаётся заметная доля плохих вариантов, а защищает от них **только выборка**. Этот пример не означает, что тестировать на $50\%$ лучше, чем на $95\%$: оптимальный порог зависит от всех параметров.

<details>
<summary><b>Интерактив: асимметрия вероятности внедрения</b></summary>

```python
def asymmetry_demo(alpha: float = 0.5, n_thousands: int = 10) -> None:
    """Сравнивает вероятность внедрения при выбранных α и n с дизайном α = 0.05."""
    n = n_thousands * 1_000
    effects = np.linspace(-0.15, 0.15, 500)
    chosen = implementation_probability(case.baseline_cr, effects, 0.0, n, alpha)
    reference = implementation_probability(case.baseline_cr, effects, 0.0, n, 0.05)
    worse, better = implementation_probability(case.baseline_cr, np.array([-0.03, 0.03]), 0.0, n, alpha)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(effects * 100, reference * 100, color=GRAY, linewidth=2, linestyle='--', label='α = 0.05')
    ax.plot(effects * 100, chosen * 100, color=BLUE, linewidth=2.5, label=f'α = {alpha:.2f}')
    ax.axhline(50, color=RED, linewidth=1.5, linestyle=':', label='Монетка')
    ax.axvline(0, color=GRAY, linestyle=':', linewidth=1.5)
    ax.set_title(f'n = {n}: вариант хуже на 3% внедрим с вероятностью {worse:.1%}, '
                 f'лучше на 3% с вероятностью {better:.1%}', loc='left', fontsize=12)
    ax.set_xlabel('Истинный относительный эффект, %')
    ax.set_ylabel('Вероятность внедрения, %')
    ax.set_ylim(0, 100)
    ax.set_xlim(-15, 15)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


try:
    interact(asymmetry_demo,
             alpha=FloatSlider(value=0.5, min=0.01, max=0.8, step=0.01, description='α'),
             n_thousands=IntSlider(value=10, min=1, max=300, step=1, description='n, тыс'))
except Exception:
    print('Виджеты недоступны — показываю базовый сценарий.')
    asymmetry_demo()
```

</details>

---

## Часть 3. Вердикт: стоит ли тестировать

### 3.1 Сравним все варианты решения

Соберём в одну таблицу: оптимальные дизайны, стандартный дизайн и два решения без теста. Решение «внедрить без теста» считает прибыль за весь горизонт — нет ни затрат на тест, ни задержки.

```python
horizon_revenue = case.weekly_revenue * case.horizon_weeks
ship_without_test = (np.trapezoid(horizon_revenue * effect_grid * density, effect_grid)
                     - case.implementation_cost - case.maintenance_cost
                     - case.rollback_cost * np.trapezoid(density * (effect_grid < case.break_even), effect_grid))

rr_alpha, rr_n, *_ = best_design(grid, 'rr')
net_alpha, net_n, *_ = best_design(grid, 'net')

options = {'Внедрить без теста': (ship_without_test, np.nan, np.nan), 'Ничего не делать': (0.0, np.nan, np.nan)}
for name, (alpha, n) in {
    'Стандартный дизайн': (0.025, n_standard),
    'Максимум RR': (rr_alpha, rr_n),
    'Максимум прибыли': (net_alpha, net_n),
}.items():
    result = evaluate_design(case, effect_grid, density, n, alpha)
    options[name] = (float(result['net']), float(result['rr']), 2 * n / case.weekly_users)

comparison = pd.DataFrame(options, index=['Чистая прибыль, ₽', 'RR', 'Длительность, недель']).T
comparison.round(2)
```

```python
ordered = comparison['Чистая прибыль, ₽'].sort_values()
fig, ax = plt.subplots(figsize=(12, 4.4))
colors = [GREEN if value > 0 else RED for value in ordered.values]
ax.barh(ordered.index, ordered.values / 1e6, color=colors, height=0.65)
for position, value in enumerate(ordered.values):
    ax.text(value / 1e6 + (0.02 if value >= 0 else -0.02), position, f'{value / 1e6:.2f} млн ₽',
            va='center', ha='left' if value >= 0 else 'right')
ax.axvline(0, color=GRAY, linewidth=1)
ax.set_xlim(ordered.min() / 1e6 * 1.6, ordered.max() / 1e6 * 1.25)
ax.set_title('Ожидаемая чистая прибыль решений', loc='left')
ax.set_xlabel('Млн ₽')
ax.grid(axis='y', visible=False)
plt.tight_layout()
plt.show()
```

**Что видим.**

- **Оптимум лежит внутри сетки.** Лучшие дизайны длятся $4$–$6$ недель; дальше потери от задержки внедрения перевешивают пользу от точности.
- **Стандартный дизайн прибылен, но не оптимален.** Здесь он длится $\approx 13$ недель и приносит $\approx 0.71$ млн ₽ против $\approx 1.08$ млн ₽ у лучшего варианта — проигрывает и из-за строгого порога, и из-за отложенного внедрения.
- **Внедрение без теста убыточно** ($\approx -0.5$ млн ₽): при нулевом среднем ожидании половина вариантов вредна. Именно отбор вариантов и создаёт ценность теста.

### 3.2 Требуемый $n$ → длительность → доступный трафик

Теперь переведём оптимум в практический вопрос: **хватит ли трафика?**

```python
net_alpha, net_n, *_ = best_design(grid, 'net')
total_users = 2 * net_n
weeks = total_users / case.weekly_users

print(f"Оптимум по прибыли: α = {net_alpha}, n = {spaced(int(net_n))} на группу")
print(f"Всего пользователей нужно: {spaced(int(total_users))}")
print(f"При {spaced(case.weekly_users)} в неделю тест займёт {weeks:.1f} недель (~{weeks * 7:.0f} дней)")
print(f"Это {(weeks / case.horizon_weeks):.1%} горизонта планирования")
```

**Как читать.** Оптимальный дизайн по прибыли требует $\approx 293$ тыс. пользователей суммарно (две группы по $\approx 147$ тыс.). При потоке $50$ тыс. в неделю это $\approx 5.9$ недель — чуть больше $11\%$ горизонта. Трафик есть с запасом: тест короткий относительно горизонта, а задержка внедрения уже учтена в модели как стоимость.

### 3.3 Ответ на вопрос встречи

**Стоит ли тестировать изменение checkout-страницы? Да — тестировать.**

- При нулевом априорном ожидании ($\mu = 0$, $\sigma = 3\%$) «внедрять без теста» в среднем убыточно ($\approx -0.50$ млн ₽), а «ничего не делать» даёт $0$.
- Оптимальный по прибыли тест ($\alpha = 0.5$ односторонний, $n \approx 147$ тыс. на группу) приносит **$\approx 1.08$ млн ₽ ожидаемой чистой прибыли** и длится $\approx 5.9$ недель — трафика хватает.
- Стандартный дизайн ($\alpha = 0.025$ односторонний, $n \approx 331$ тыс.) тоже прибылен, но хуже: $\approx 0.71$ млн ₽ за $\approx 13$ недель.
- Ценность теста создаёт не «детектирование эффекта», а **асимметричный отбор**: тест пропускает хвост хороших вариантов и отсекает вредные.

Важно: $\alpha = 0.5$ здесь — не универсальный совет, а следствие экономики кейса (цена ошибки мала против годовой выручки). Для дорогого и почти необратимого решения оптимальный порог строже — расчёт нужно делать под конкретный тест.

---

## Ограничения модели

<details>
<summary><b>Развёрнуто</b></summary>

1. **Упрощённое распределение ожиданий.** На практике $\pi(\theta)$ может быть сложнее и асимметричнее.
2. **Постоянный эффект.** Эффект может затухать со временем (novelty effect).
3. **Долгосрочные эффекты.** Некоторые изменения влияют на LTV, бренд и лояльность — их трудно измерить в краткосрочном эксперименте.
4. **Альтернативные издержки.** Время и ресурсы на тест можно потратить иначе; ответ помогает сравнение с решением без теста (мы его сделали выше).
5. **Проблема референтного класса.** Выбор априорного распределения субъективен.
6. **Упрощённый откат.** Модель считает откат платным, но потери после ошибочного внедрения идут весь горизонт, хотя откат их бы прервал — откат считается пессимистично.
7. **Фиксированные затраты в пороге.** К моменту решения затраты на тест уже понесены, поэтому в точке безубыточности для решения их можно не учитывать; мы следуем исходной постановке, где они включены.
8. **Постоянные изменения продукта.** Продукты с большим числом тестов могут страдать от самого факта частых изменений независимо от результатов.

Точность расчёта равна точности входов: если распределение ожиданий искажено, затраты недооценены или выгоды преувеличены, «оптимальный» дизайн на самом деле не оптимален. Существует «достаточно хороший» уровень оценки, который уравновешивает сложность получения точных оценок и их пользу.

</details>

## Что вынести

- Стандартные $\alpha = 0.05$ и мощность $80\%$ — это соглашение, а не оптимум для бизнеса.
- Дизайн можно подбирать под деньги: нужны затраты, выручка, горизонт и распределение ожидаемых эффектов.
- Цель нужно выбирать явно: максимум отношения выгоды к риску и максимум прибыли дают разные дизайны.
- Модель обязана учитывать цену времени, иначе «оптимальная» выборка будет бесконечно расти.
- Высокое $\alpha$ не равно монетке, но и не безопасно само по себе: защита от плохих вариантов зависит от размера выборки.
- Чем дороже и необратимее ошибка, тем строже оптимальный порог, поэтому результат нужно проверять на чувствительность.
