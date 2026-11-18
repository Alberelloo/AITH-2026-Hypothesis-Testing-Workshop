"""Подсчёт входного и выходного тестов воркшопа.

Запуск:
    python assessment/score.py assessment/sample_responses.csv [--key assessment/answer_key.json]

CSV: participant_id, test, q1..qN. test — entry или exit. Пустая ячейка — нет ответа.
Ответы: одна буква (A–E); набор букв для множественного выбора ("BCE", "B,C,E", "B;C;E");
число (десятичная точка или запятая, знак % и пробелы отбрасываются).

Выход:
  1) балл каждого участника по каждому тесту и «прошёл / нет» для выходного (порог из ключа);
  2) сдвиг по якорям: доля верных по 6 якорям после − до, в п.п., по участникам с обоими тестами;
  3) сдвиг по блокам (ILO): доля верных в блоке на выходе − на входе, в п.п.
"""
import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_number(raw):
    s = re.sub(r"[%\s₽]|п\.п\.|дн(я|ей|ь)?", "", raw.strip().lower()).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def letters(raw):
    s = raw.upper().translate(str.maketrans("АВСЕ", "ABCE"))  # кириллица, похожая на латиницу
    return frozenset(re.findall(r"[A-E]", s))


def is_correct(spec, raw):
    if raw is None or not raw.strip():
        return False
    if spec["type"] == "numeric":
        x = parse_number(raw)
        return x is not None and abs(x - spec["answer"]) <= spec["tol"] + 1e-9
    given = letters(raw)
    if spec["type"] == "single" and len(given) != 1:
        return False
    return given == frozenset(spec["answer"])


def pct(x):
    return f"{100 * x:5.1f}%" if x is not None else "   — "


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("responses", type=Path)
    ap.add_argument("--key", type=Path, default=HERE / "answer_key.json")
    args = ap.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    tests, threshold = key["tests"], key["pass_threshold"]

    # marks[pid][test][q] = True/False
    marks = {}
    with args.responses.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid, test = row["participant_id"].strip(), row["test"].strip().lower()
            if test not in tests:
                raise SystemExit(f"Неизвестный test={test!r} у {pid}: ожидается entry или exit")
            marks.setdefault(pid, {})[test] = {q: is_correct(spec, row.get(q)) for q, spec in tests[test].items()}

    # 1. Баллы
    print("Баллы")
    print(f"{'участник':<12}{'вход':>10}{'выход':>10}  прошёл (≥ {threshold:.0%})")
    passed = []
    for pid in sorted(marks):
        cells = []
        for test in ("entry", "exit"):
            m = marks[pid].get(test)
            cells.append(f"{sum(m.values())}/{len(m)}" if m else "—")
        ex = marks[pid].get("exit")
        ok = None if ex is None else sum(ex.values()) / len(ex) >= threshold
        if ok is not None:
            passed.append(ok)
        print(f"{pid:<12}{cells[0]:>10}{cells[1]:>10}  {'—' if ok is None else ('да' if ok else 'нет')}")
    if passed:
        print(f"Прошли порог выходного теста: {sum(passed)} из {len(passed)} ({sum(passed) / len(passed):.0%})")

    paired = [pid for pid in sorted(marks) if {"entry", "exit"} <= marks[pid].keys()]
    if not paired:
        print("\nНет участников с обоими тестами — сдвиги не считаются.")
        return
    print(f"\nСдвиги — по {len(paired)} участникам с обоими тестами")

    # 2. Якоря
    anchors = key["anchors"]
    def anchor_share(pid, test):
        return mean(marks[pid][test][a[test]] for a in anchors)
    before = mean(anchor_share(p, "entry") for p in paired)
    after = mean(anchor_share(p, "exit") for p in paired)
    shift = 100 * (after - before)
    target = key.get("anchor_target_pp", 30)
    print(f"\nЯкоря ({len(anchors)} пар): до {pct(before)}, после {pct(after)}, "
          f"сдвиг {shift:+.1f} п.п. (цель ≥ +{target} п.п.: {'выполнена' if shift >= target else 'не выполнена'})")
    print(f"{'пара':<6}{'блок':>5}{'вход':>7}{'выход':>7}{'до':>9}{'после':>9}{'сдвиг, п.п.':>14}")
    for a in anchors:
        b = mean(marks[p]["entry"][a["entry"]] for p in paired)
        e = mean(marks[p]["exit"][a["exit"]] for p in paired)
        print(f"{a['id']:<6}{a['block']:>5}{a['entry']:>7}{a['exit']:>7}{pct(b):>9}{pct(e):>9}{100 * (e - b):>+14.1f}")

    # 3. Блоки
    print("\nБлоки (все вопросы блока; входной и выходной наборы разные, сдвиг — для матрицы выравнивания)")
    print(f"{'блок':<42}{'до':>9}{'после':>9}{'сдвиг, п.п.':>14}")
    for b, name in key["blocks"].items():
        share = {}
        for test in ("entry", "exit"):
            qs = [q for q, spec in tests[test].items() if str(spec["block"]) == b]
            share[test] = mean(marks[p][test][q] for p in paired for q in qs)
        print(f"{b + '. ' + name:<42}{pct(share['entry']):>9}{pct(share['exit']):>9}"
              f"{100 * (share['exit'] - share['entry']):>+14.1f}")


if __name__ == "__main__":
    main()
