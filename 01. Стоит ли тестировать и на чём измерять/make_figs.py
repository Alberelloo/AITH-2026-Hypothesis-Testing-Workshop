# -*- coding: utf-8 -*-
"""
Генерация PNG-графиков для встречи 1 «Стоит ли тестировать».
Код воспроизводит ячейки ноутбука
  «1. Дизайн и анализ экспериментов / 06. Подбор уровня значимости и размера выборки теста».
Запуск:  MPLBACKEND=Agg python make_figs.py
"""
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BLUE, RED, GREEN, GRAY, INK = '#4C72B0', '#C44E52', '#55A868', '#7F7F7F', '#1b1b1b'
ORANGE = '#DD8452'
GOLD = '#FFD166'

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved  ', path)


def spaced(value: float) -> str:
    return f'{value:,.0f}'.replace(',', ' ')


# ----------------------------------------------------------------------------
# Бизнес-кейс
# ----------------------------------------------------------------------------
@dataclass
class BusinessCase:
    """Параметры бизнес-кейса."""
    baseline_cr: float = 0.05
    weekly_users: int = 50_000
    weekly_revenue: float = 3_000_000
    fixed_test_cost: float = 180_000
    implementation_cost: float = 120_000
    rollback_cost: float = 250_000
    monthly_maintenance: float = 20_000
    horizon_months: float = 12

    @property
    def horizon_weeks(self) -> float:
        return self.horizon_months * 52 / 12

    @property
    def maintenance_cost(self) -> float:
        return self.monthly_maintenance * self.horizon_months

    @property
    def break_even(self) -> float:
        costs = self.fixed_test_cost + self.implementation_cost + self.maintenance_cost
        return costs / (self.weekly_revenue * self.horizon_weeks)


case = BusinessCase()

# Сетка эффектов и априорное распределение
effect_grid = np.linspace(-0.10, 0.10, 1000)
prior_mu = 0.0
prior_sigma = 0.03
density = norm.pdf(effect_grid, loc=prior_mu, scale=prior_sigma)
density /= np.trapezoid(density, effect_grid)

break_even = case.break_even
fixed_costs = case.fixed_test_cost + case.implementation_cost + case.maintenance_cost
print(f'Горизонт: {case.horizon_weeks:.0f} недель')
print(f'Поддержка за горизонт: {spaced(case.maintenance_cost)} ₽')
print(f'Суммарные фиксированные издержки: {spaced(fixed_costs)} ₽')
print(f'Fixed-cost break-even point: {break_even:.3%} относительного uplift')


# ----------------------------------------------------------------------------
# Базовые функции (из ноутбука)
# ----------------------------------------------------------------------------
def implementation_probability(
    p_control: float,
    relative_effect: np.ndarray,
    break_even_effect: float,
    n_per_group: np.ndarray,
    alpha: np.ndarray = 0.05,
) -> np.ndarray:
    p_true = np.clip(p_control * (1 + relative_effect), 1e-6, 1 - 1e-6)
    delta_true = p_true - p_control
    delta_be = p_control * break_even_effect
    se = np.sqrt(p_control * (1 - p_control) / n_per_group + p_true * (1 - p_true) / n_per_group)
    z_crit = norm.ppf(1 - alpha)
    return norm.sf(z_crit - (delta_true - delta_be) / se)


def evaluate_design(case, effect_grid, density, n_per_group, alpha) -> dict:
    n_per_group = np.asarray(n_per_group, dtype=float)
    probability = implementation_probability(case.baseline_cr, effect_grid, case.break_even, n_per_group, alpha)

    test_weeks = 2 * n_per_group / case.weekly_users
    post_weeks = np.maximum(case.horizon_weeks - test_weeks, 0)

    test_value = case.weekly_revenue * effect_grid * test_weeks * 0.5
    post_value = case.weekly_revenue * effect_grid * post_weeks * probability

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
    return pd.Series({
        'Выгоды, ₽': result['expected_gain'],
        'Затраты, ₽': result['expected_loss'],
        'Чистая прибыль, ₽': result['net'],
        'RR': result['rr'],
    })


alpha_grid = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8])
n_grid = np.unique(np.round(np.geomspace(1_000, 600_000, 60), -2)).astype(int)


def design_grid(case, effects, weights) -> dict:
    return evaluate_design(case, effects, weights, n_grid[None, :, None], alpha_grid[:, None, None])


def best_design(grid, metric) -> tuple:
    row, column = np.unravel_index(np.argmax(grid[metric]), grid[metric].shape)
    return alpha_grid[row], n_grid[column], grid['rr'][row, column], grid['net'][row, column]


# ----------------------------------------------------------------------------
# FIG 1. Распределение ожиданий + точка безубыточности
# ----------------------------------------------------------------------------
prior = norm(prior_mu, prior_sigma)
x = effect_grid * 100

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.plot(x, density, color=GRAY, linewidth=2)
ax.fill_between(x, density, where=effect_grid >= break_even, color=GREEN, alpha=0.4,
                label=f'Прибыль, θ ≥ {break_even:.2%}: вероятность {prior.sf(break_even):.1%}')
ax.fill_between(x, density, where=(effect_grid > 0) & (effect_grid < break_even), color=ORANGE, alpha=0.6,
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
save(fig, 'expectation_break_even.png')


# ----------------------------------------------------------------------------
# FIG 2. Вероятность внедрения
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.2))
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
save(fig, 'implementation_probability.png')


# ----------------------------------------------------------------------------
# FIG 3. Затраты и выгоды с поправкой на риск (n=5000 vs n=50000)
# ----------------------------------------------------------------------------
def plot_risk_adjusted(designs, fname):
    results = [evaluate_design(case, effect_grid, density, n, alpha) for n, alpha in designs]
    limit = max(max(r['gains'].max(), r['losses'].max()) for r in results) * 1.05
    x = effect_grid * 100

    fig, axes = plt.subplots(2, len(designs), figsize=(6.8 * len(designs), 7.2),
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
    save(fig, fname)


designs_example = [(5_000, 0.05), (50_000, 0.05)]
plot_risk_adjusted(designs_example, 'risk_adjusted.png')
for n, alpha in designs_example:
    d = describe_design(evaluate_design(case, effect_grid, density, n, alpha))
    print(f'n={n}, α={alpha}: gains={d["Выгоды, ₽"]:.0f}, costs={d["Затраты, ₽"]:.0f}, '
          f'net={d["Чистая прибыль, ₽"]:.0f}, RR={d["RR"]:.2f}')


# ----------------------------------------------------------------------------
# FIG 4. Тепловые карты α×n (прибыль и RR)
# ----------------------------------------------------------------------------
def plot_design_maps(case, grid, fname, marks=None):
    n_pos = np.arange(n_grid.size)
    a_pos = np.arange(alpha_grid.size)
    panels = [
        (grid['rr'], 1.0, 'RR — отношение выгоды к риску', 'rr'),
        (grid['net'] / 1e6, 0.0, 'Чистая прибыль, млн ₽', 'net'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.9), sharey=True)
    for ax, (values, center, label, metric) in zip(axes, panels):
        vmin, vmax = values.min(), values.max()
        vmin = min(vmin, center - 1e-6)
        vmax = max(vmax, center + 1e-6)
        mesh = ax.pcolormesh(n_grid, a_pos, values, cmap='RdBu', shading='nearest',
                             norm=TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax))
        cb = fig.colorbar(mesh, ax=ax, label=label, fraction=0.046, pad=0.02)
        ax.contour(n_grid, a_pos, values, levels=[center], colors='black', linewidths=1.5)

        best_alpha, best_n, best_rr, best_net = best_design(grid, metric)
        best_row = np.searchsorted(alpha_grid, best_alpha)
        best_col = np.searchsorted(n_grid, best_n)
        best_val = values[best_row, best_col]
        ax.scatter(best_n, best_row, marker='*', s=430, color=GOLD, edgecolor='black', zorder=5)
        annot_val = f'{best_rr:.2f}' if metric == 'rr' else f'{best_net / 1e6:.2f} млн ₽'
        ax.annotate(f'α = {best_alpha}, n = {spaced(best_n)}\n{annot_val}',
                    xy=(best_n, best_row), xytext=(best_n * 1.35, best_row + 1.6),
                    fontsize=9.5, color='black', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc=GOLD, ec='black', alpha=0.9),
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
    save(fig, fname)


mde_relative = 0.03
delta = case.baseline_cr * mde_relative
variance = case.baseline_cr * (1 - case.baseline_cr)
n_standard = int(np.ceil(2 * variance * (norm.ppf(0.975) + norm.ppf(0.8)) ** 2 / delta ** 2))
print(f'n_standard (стандартный дизайн): {spaced(n_standard)}')

grid = design_grid(case, effect_grid, density)
plot_design_maps(case, grid, 'design_maps.png',
                 marks={f'Стандартный дизайн: n = {spaced(n_standard)}': (0.025, n_standard)})

rr_alpha, rr_n, rr_rr, rr_net = best_design(grid, 'rr')
net_alpha, net_n, net_rr, net_net = best_design(grid, 'net')
print(f'Максимум RR:      α={rr_alpha}, n={rr_n}, RR={rr_rr:.3f}, net={rr_net:.0f}, '
      f'{2 * rr_n / case.weekly_users:.2f} нед, {2 * rr_n / case.weekly_users * 7:.1f} дней')
print(f'Максимум прибыли: α={net_alpha}, n={net_n}, RR={net_rr:.3f}, net={net_net:.0f}, '
      f'{2 * net_n / case.weekly_users:.2f} нед, {2 * net_n / case.weekly_users * 7:.1f} дней')

# Стандартный дизайн для сравнения
std_result = evaluate_design(case, effect_grid, density, n_standard, 0.025)
print(f'Стандартный дизайн: net={std_result["net"]:.0f}, RR={std_result["rr"]:.2f}, '
      f'{2 * n_standard / case.weekly_users:.2f} нед')


# ----------------------------------------------------------------------------
# FIG 5. Статическая версия optimum_demo (дорогой откат)
# ----------------------------------------------------------------------------
rollback_case = BusinessCase(rollback_cost=2_000_000)
rb_effects = np.linspace(-0.15, 0.15, 600)
rb_weights = norm.pdf(rb_effects, 0.0, 0.03)
rb_weights /= np.trapezoid(rb_weights, rb_effects)
rb_grid = design_grid(rollback_case, rb_effects, rb_weights)
plot_design_maps(rollback_case, rb_grid, 'optimum_demo_rollback.png')
rb_alpha, rb_n, rb_rr, rb_net = best_design(rb_grid, 'net')
print(f'Дорогой откат 2 млн: α={rb_alpha}, n={rb_n}, net={rb_net:.0f}, {2 * rb_n / rollback_case.weekly_users:.1f} нед')


# ----------------------------------------------------------------------------
# FIG 6. Статическая версия asymmetry_demo (α=0.5, n=100 000)
# ----------------------------------------------------------------------------
n_asym = 100_000
alpha_asym = 0.5
effects = np.linspace(-0.15, 0.15, 500)
chosen = implementation_probability(case.baseline_cr, effects, 0.0, n_asym, alpha_asym)
reference = implementation_probability(case.baseline_cr, effects, 0.0, n_asym, 0.05)
worse, better = implementation_probability(case.baseline_cr, np.array([-0.03, 0.03]), 0.0, n_asym, alpha_asym)

fig, ax = plt.subplots(figsize=(12.5, 5.2))
ax.plot(effects * 100, reference * 100, color=GRAY, linewidth=2, linestyle='--', label='α = 0.05')
ax.plot(effects * 100, chosen * 100, color=BLUE, linewidth=2.5, label=f'α = {alpha_asym:.2f}')
ax.axhline(50, color=RED, linewidth=1.5, linestyle=':', label='Монетка')
ax.axvline(0, color=GRAY, linestyle=':', linewidth=1.5)
ax.set_title(f'n = {spaced(n_asym)}: вариант хуже на 3% внедрим с вероятностью {worse:.1%}, '
             f'лучше на 3% с вероятностью {better:.1%}', loc='left', fontsize=12)
ax.set_xlabel('Истинный относительный эффект, %')
ax.set_ylabel('Вероятность внедрения, %')
ax.set_ylim(0, 100)
ax.set_xlim(-15, 15)
ax.legend(loc='upper left')
plt.tight_layout()
save(fig, 'asymmetry_demo.png')


# ----------------------------------------------------------------------------
# FIG 7. Качество роллаута
# ----------------------------------------------------------------------------
def rollout_quality(n_per_group: int, alpha) -> pd.Series:
    if alpha is None:
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


designs_rq = [
    ('Монетка', rollout_quality(0, None)),
    ('n = 10k, α = 0.05', rollout_quality(10_000, 0.05)),
    ('n = 10k, α = 0.5', rollout_quality(10_000, 0.5)),
    ('n = 100k, α = 0.05', rollout_quality(100_000, 0.05)),
    ('n = 100k, α = 0.5', rollout_quality(100_000, 0.5)),
]
names = [d[0] for d in designs_rq]
bad_share = [100 * d[1]['Доля плохих среди внедрённых'] for d in designs_rq]
mean_effect = [d[1]['Средний эффект на решение, %'] for d in designs_rq]
ship_share = [100 * d[1]['Доля внедрений'] for d in designs_rq]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
y = np.arange(len(names))[::-1]

axes[0].barh(y, bad_share, color=[GRAY if n == 'Монетка' else BLUE for n in names], height=0.62)
axes[0].set_yticks(y, names)
axes[0].set_title('Доля плохих среди внедрённых, %', loc='left')
axes[0].axvline(50, color=RED, linestyle=':', linewidth=1.5)
for yi, v in zip(y, bad_share):
    axes[0].text(v + 1, yi, f'{v:.0f}%', va='center', fontsize=9.5, fontweight='bold')
axes[0].set_xlim(0, 62)

colors2 = [GRAY if n == 'Монетка' else GREEN for n in names]
axes[1].barh(y, mean_effect, color=colors2, height=0.62)
axes[1].set_yticks(y, names)
axes[1].set_title('Средний эффект на решение, %', loc='left')
axes[1].axvline(0, color=GRAY, linewidth=1)
for yi, v in zip(y, mean_effect):
    axes[1].text(v + (0.02 if v >= 0 else -0.02), yi, f'{v:.2f}%', va='center',
                 ha='left' if v >= 0 else 'right', fontsize=9.5, fontweight='bold')
axes[1].set_xlim(-0.25, 1.25)

plt.tight_layout()
save(fig, 'rollout_quality.png')
print('rollout_quality:')
for n, d in designs_rq:
    print(f'  {n}: внедрений={d["Доля внедрений"]:.3f}, плохих={d["Доля плохих среди внедрённых"]:.3f}, '
          f'средний эффект={d["Средний эффект на решение, %"]:.3f}%')


# ----------------------------------------------------------------------------
# FIG 8. Сравнение решений (вердикт)
# ----------------------------------------------------------------------------
horizon_revenue = case.weekly_revenue * case.horizon_weeks
ship_without_test = (np.trapezoid(horizon_revenue * effect_grid * density, effect_grid)
                     - case.implementation_cost - case.maintenance_cost
                     - case.rollback_cost * np.trapezoid(density * (effect_grid < case.break_even), effect_grid))

options = {'Внедрить без теста': (ship_without_test, np.nan, np.nan), 'Ничего не делать': (0.0, np.nan, np.nan)}
for name, (alpha, n) in {
    'Стандартный дизайн': (0.025, n_standard),
    'Максимум RR': (rr_alpha, rr_n),
    'Максимум прибыли': (net_alpha, net_n),
}.items():
    result = evaluate_design(case, effect_grid, density, n, alpha)
    options[name] = (float(result['net']), float(result['rr']), 2 * n / case.weekly_users)

comparison = pd.DataFrame(options, index=['Чистая прибыль, ₽', 'RR', 'Длительность, недель']).T
print('\ncomparison:')
print(comparison.round(2).to_string())

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
save(fig, 'comparison_bars.png')

print('\nDone. Figures in', FIG_DIR)
