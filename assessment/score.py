"""Подсчёт входного и выходного тестов воркшопа.

Запуск:
    python assessment/score.py assessment/sample_responses.csv [--key assessment/answer_key.json]

CSV: participant_id, test, q1..qN. test — entry или exit. Пустая ячейка — нет ответа.
Ответы: одна буква (A–E) или «D. текст варианта»; набор букв для множественного выбора
("BCE", "B,C,E", "B; C; E"); число — берётся первое число в ячейке, единицы и знак % отбрасываются,
запятая = точка. participant_id сравнивается без учёта регистра и лишних пробелов.
Неразобранные ответы и повторы участника печатаются в stderr как «Внимание: …».

Выход:
  1) балл каждого участника по каждому тесту и «прошёл / нет» для выходного (порог из ключа);
  2) сдвиг по якорям: доля верных по 6 якорям после − до, в п.п., по участникам с обоими тестами;
  3) сдвиг по блокам (ILO): доля верных в блоке на выходе − на входе, в п.п.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


NUMBER = re.compile(r"[-−]?(?:\d+(?:[.,]\d+)?|[.,]\d+)")
THOUSANDS = re.compile(r"(?<=\d)[\s\u00a0\u202f](?=\d{3}(?!\d))")  # «83 333» → «83333»
CYR_TO_LAT = str.maketrans("АВСЕавсе", "ABCEABCE")  # кириллица, похожая на латиницу

WARNINGS = []


def parse_number(raw, where=""):
    """Первое число в ответе: «69 наблюдений» → 69, «12,5 дня» → 12.5, «≈ 64%» → 64."""
    s = THOUSANDS.sub("", raw.strip())
    found = NUMBER.findall(s)
    if not found:
        return None
    if len(found) > 1:
        WARNINGS.append(f"{where}: в {raw!r} несколько чисел, взято первое — {found[0]}")
    return float(found[0].replace("−", "-").replace(",", "."))


def letters(raw):
    """Набор букв A–E или None, если ответ не разобран.

    «D», «B, C, E», «BCE», «B;C;E» — буквы; «D. текст варианта» / «D) …» — буква перед точкой или скобкой;
    слова вроде «Вариант», «ответ», «и» пропускаются, поэтому «Вариант C» → {C}, а не {B, A, C}.
    """
    s = raw.strip().translate(CYR_TO_LAT).upper()
    m = re.match(r"([A-E])\s*[.)]\s+\S", s)
    if m:
        return frozenset(m.group(1))
    got = set()
    for tok in filter(None, re.split(r"[\W_]+", s)):  # разделитель — любой не буквенно-цифровой символ
        if re.fullmatch(r"[A-E]+", tok):
            got |= set(tok)
        elif not re.fullmatch(r"[^\W\d_]+", tok):  # не слово и не буквы A–E: «1», «A2», «C?»
            return None
    return frozenset(got) or None


def is_correct(spec, raw, where=""):
    if raw is None or not raw.strip():
        return False
    if spec["type"] == "numeric":
        x = parse_number(raw, where)
        if x is None:
            WARNINGS.append(f"{where}: не число {raw!r}, засчитано как неверное")
        return x is not None and abs(x - spec["answer"]) <= spec["tol"] + 1e-9
    given = letters(raw)
    if given is None:
        WARNINGS.append(f"{where}: не разобраны буквы в {raw!r}, засчитано как неверное")
        return False
    if spec["type"] == "single" and len(given) != 1:
        WARNINGS.append(f"{where}: в одиночном выборе несколько букв {''.join(sorted(given))}, засчитано как неверное")
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
        for line, row in enumerate(csv.DictReader(f), start=2):
            pid = " ".join((row["participant_id"] or "").split()).lower()  # «Ivan@X.ru » == «ivan@x.ru»
            test = (row["test"] or "").strip().lower()
            if not pid:
                WARNINGS.append(f"строка {line}: пустой participant_id, строка пропущена")
                continue
            if test not in tests:
                raise SystemExit(f"Неизвестный test={test!r} у {pid}: ожидается entry или exit")
            if test in marks.get(pid, {}):
                WARNINGS.append(f"строка {line}: повтор {pid}/{test}, учтена последняя попытка")
            marks.setdefault(pid, {})[test] = {
                q: is_correct(spec, row.get(q), f"строка {line}, {pid}/{test}/{q}") for q, spec in tests[test].items()
            }
    for w in WARNINGS:
        print(f"Внимание: {w}", file=sys.stderr)

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
