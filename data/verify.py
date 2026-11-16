# -*- coding: utf-8 -*-
"""Проверка четырёх CSV: схема, примеры строк, целостность зашитых дефектов."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ok, bad = [], []


def check(cond, msg):
    (ok if cond else bad).append(msg)


def size(name):
    return (D / name).stat().st_size / 1024 ** 2


# ============================ checkout =====================================
co = pd.read_csv(D / "checkout_ab.csv")
print("=" * 78)
print(f"checkout_ab.csv  {size('checkout_ab.csv'):.2f} МБ  {co.shape[0]:,} x {co.shape[1]}")
print(co.dtypes.to_string())
print("\nпример строк:")
print(co.head(3).to_string(index=False))
print("\nпоследние колонки (предпериод):")
print(co[["user_id", "group", "revenue", "orders_prev28", "revenue_prev28"]].head(3).to_string(index=False))

check(co.shape == (105330, 9), f"shape 105330x9 (got {co.shape})")
check(list(co.columns) == ["user_id", "date", "group", "converted", "revenue",
                           "checkout_time_sec", "payment_error",
                           "orders_prev28", "revenue_prev28"], "порядок колонок")
check(co.isna().sum().sum() == 0, "нет пропусков")
check(set(co.group.unique()) == {"A", "B"}, "группы A/B")
check((co.converted.isin([0, 1])).all() and (co.payment_error.isin([0, 1])).all(), "бинарные 0/1")
check(co.converted.mean() < co[co.group == "B"].converted.mean(), "конверсия B выше A")
check((co.loc[co.converted == 0, "revenue"] == 0).all(), "revenue=0 при converted=0")
check((co.orders_prev28 >= 0).all() and (co.revenue_prev28 >= 0).all(), "предпериод ≥ 0")
d1, d2 = pd.to_datetime(co.date).min(), pd.to_datetime(co.date).max()
check((d1 == pd.Timestamp("2026-09-07")) and (d2 == pd.Timestamp("2026-09-20")), f"даты {d1.date()}..{d2.date()}")
both = co.groupby("user_id").group.nunique()
check((both > 1).sum() == 60, f"юзеров в обеих группах: {(both > 1).sum()} (ожидается 60)")
dups = co.duplicated().sum()
check(dups == 250, f"полных дублей: {dups} (ожидается 250)")
# у задвоенных пользователей предпериод обязан совпадать
mism = 0
for uid, g in co[co.user_id.isin(both[both > 1].index)].groupby("user_id"):
    if g.orders_prev28.nunique() != 1 or g.revenue_prev28.round(6).nunique() != 1:
        mism += 1
check(mism == 0, f"предпериод одинаков у задвоенных (расхождений {mism})")
check(co.user_id.nunique() == 105020, "уникальных user_id: 105 020")
rho = np.corrcoef(co.revenue, co.revenue_prev28)[0, 1]
check(0.5 <= rho <= 0.7, f"corr(revenue, revenue_prev28) = {rho:.3f} в [0.5, 0.7]")

# ============================ store_panel ==================================
sp = pd.read_csv(D / "store_panel.csv", parse_dates=["week"])
tr = pd.read_csv(D / "store_panel_truth.csv")
print("\n" + "=" * 78)
print(f"store_panel.csv  {size('store_panel.csv'):.2f} МБ  {sp.shape[0]:,} x {sp.shape[1]}")
print(sp.dtypes.to_string())
print("\nпример строк:")
print(sp.head(3).to_string(index=False))
print(f"store_panel_truth.csv  {size('store_panel_truth.csv'):.2f} МБ  {tr.shape[0]} x {tr.shape[1]}")

check(sp.shape == (23400, 7), f"shape 23400x7 (got {sp.shape})")
check(list(sp.columns) == ["store", "region", "format", "week", "sales", "traffic", "promo"], "колонки панели")
check(sp.isna().sum().sum() == 0, "нет пропусков")
check(sp.store.nunique() == 300 and sp.week.nunique() == 78, "300 магазинов × 78 недель")
check(len(sp) == 300 * 78, "панель полная, без дыр")
check(set(sp.week.dt.dayofweek) == {0}, "все недели — понедельники")
check(set(sp.promo.unique()) <= {0, 1}, "promo 0/1")
check((sp.sales > 0).all() and (sp.traffic > 0).all(), "sales/traffic > 0")
check(set(sp.format.unique()) <= {"a", "b", "c"}, "форматы a/b/c")
check(sp.region.nunique() == 6, "6 регионов")
check(tr.shape[0] == 300 and tr.treated.sum() == 60, "truth: 300 магазинов, 60 леченых")
check(set(tr.treat_week[tr.treated == 1].unique()) == {"2024-10-07"}, "окно лечения 2024-10-07")
check((tr.treat_week[tr.treated == 0] == "").all() or tr.treat_week[tr.treated == 0].isna().all(),
      "у контроля дата лечения пустая")
check((tr[tr.treated == 1].true_sales_lift.round(4) == 0.0353).all(), "истинный эффект +3.53%")

# справочно: как ведут себя наивный DiD, плацебо и мэтчинг (материал встречи 5)
p2 = sp.assign(y=np.log(sp.sales))
weeks = sorted(p2.week.unique())
t0 = weeks.index(pd.Timestamp("2024-10-07"))
window = weeks[t0:t0 + 12]
pre = weeks[t0 - 12:t0]
tid = set(tr.loc[tr.treated == 1, "store"])


def did(t_ids, c_ids, w_after, w_before):
    """DiD в логах: T = t_ids, C = c_ids."""
    g = p2.assign(t=p2.store.isin(t_ids), c=p2.store.isin(c_ids), after=p2.week.isin(w_after))
    g = g[(g.after | g.week.isin(w_before)) & (g.t | g.c)]
    return ((g[g.t & g.after].y.mean() - g[g.t & ~g.after].y.mean())
            - (g[g.c & g.after].y.mean() - g[g.c & ~g.after].y.mean()))


base_mean = p2[p2.week.isin(pre)].groupby("store").y.mean()
ctrl = [s for s in p2.store.unique() if s not in tid]
matched = {min(ctrl, key=lambda c: abs(base_mean[c] - base_mean[t])) for t in tid}
print("\nсправочно (лог-выручка):")
print(f"  истинный эффект        {np.log(1.0353):+.4f}")
print(f"  наивный DiD            {did(tid, ctrl, window, pre):+.4f}")
print(f"  плацебо по датам       {did(tid, ctrl, weeks[t0 - 20:t0 - 8], weeks[t0 - 32:t0 - 20]):+.4f}")
print(f"  DiD с мэтчингом        {did(tid, matched, window, pre):+.4f}  (контроль: {len(matched)} магазинов)")

# ============================ segments =====================================
sg = pd.read_csv(D / "hotdeals_segmented.csv", parse_dates=["date", "reg_date"])
print("\n" + "=" * 78)
print(f"hotdeals_segmented.csv  {size('hotdeals_segmented.csv'):.2f} МБ  {sg.shape[0]:,} x {sg.shape[1]}")
print(sg.dtypes.to_string())
print("\nпример строк:")
print(sg.head(3).to_string(index=False))

check(list(sg.columns) == ["user_id", "date", "group", "booked", "revenue", "cancelled",
                           "device", "reg_date", "tenure_days", "is_new",
                           "orders_90d", "segment"], "колонки логов с сегментами")
check(sg.isna().sum().sum() == 0, "нет пропусков")
check(sg.date.nunique() == 28, "28 дней")
check(set(sg.device.unique()) == {"web", "mobile"}, "device web/mobile")
check(set(sg.segment.unique()) == {"low", "mid", "high"}, "segment low/mid/high")
check(set(sg.is_new.unique()) == {0, 1}, "is_new 0/1")
check((sg.reg_date <= sg.date).all(), "reg_date ≤ date")
check(((sg.date - sg.reg_date).dt.days == sg.tenure_days).all(), "tenure_days = date − reg_date")
check((sg.loc[sg.is_new == 1, "tenure_days"] <= 30).all(), "новички ≤ 30 дней")
check((sg.booked.isin([0, 1])).all(), "booked 0/1")
check((sg.loc[sg.booked == 0, "revenue"] == 0).all(), "нет выручки без брони")

# ============================ portfolio ====================================
pf = pd.read_csv(D / "portfolio_tests.csv")
print("\n" + "=" * 78)
print(f"portfolio_tests.csv  {size('portfolio_tests.csv'):.3f} МБ  {pf.shape[0]} x {pf.shape[1]}")
print(pf.dtypes.to_string())
print("\nпример строк:")
print(pf.head(3).to_string(index=False))

check(pf.shape == (6, 10), f"shape 6x10 (got {pf.shape})")
check(list(pf.columns) == ["test_id", "name", "metric", "n", "base_rate", "effect",
                           "se", "ci_low", "ci_high", "p_value"], "колонки портфеля")
check((pf.effect.abs() <= 0.1).sum() == 1, "ровно 1 нулевой тест (|effect| ≤ 0.1 п.п.)")
check(pf.effect.max() > 3 * pf.effect.sort_values().iloc[-2], "есть гетерогенный выброс (>3x)")
check(np.allclose(pf.ci_low, pf.effect - 1.959963984540054 * pf.se, atol=5e-3), "ci_low = effect − 1.96·se")
check(np.allclose(pf.ci_high, pf.effect + 1.959963984540054 * pf.se, atol=5e-3), "ci_high = effect + 1.96·se")
check((pf.p_value < 0.05).sum() == 2, "2 значимых теста из 6")

print("\n" + "=" * 78)
print(f"ПРОЙДЕНО: {len(ok)}; ПРОВАЛЕНО: {len(bad)}")
for b in bad:
    print("  FAIL:", b)
