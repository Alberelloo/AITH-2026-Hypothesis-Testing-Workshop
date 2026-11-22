# 4. Когда останавливать: последовательный анализ и futility

## О чём встреча

Четвёртый шаг жизненного цикла эксперимента: тест запущен, на него смотрят каждый день, и нужно честно решить, когда его остановить — по победе или по бесперспективности. Базовый курс дал обзор последовательного анализа; здесь разбираем устройство методов: цену подглядывания, SPRT, mSPRT и always-valid вывод, расчёт границ GST через alpha-spending и futility-анализ (conditional power, PPoS, β-spending, binding и non-binding границы). Отдельно — мост к встрече 2: почему эффект теста, остановленного рано, завышен. Итог встречи — политика остановки, которую можно зафиксировать в дизайне до старта.

## Из каких лекций курса

- «Последовательное тестирование» (первая часть встречи).
- «Futility-анализ» (вторая часть встречи).

## Ключевые идеи

- Подглядывание с обычным порогом кратно завышает долю ложных побед: около 19.5% при 10 взглядах и больше половины при 1000.
- SPRT Вальда экономит наблюдения, но требует простых гипотез и известного базового значения; mSPRT снимает эти ограничения смесью альтернатив и одной границей 1/α.
- Always-valid вывод опирается на супермартингалы и неравенство Вилле; минимум обычных p-value такой гарантии не даёт, а доверительные последовательности шире обычных интервалов.
- В GST значение alpha-spending функции — накопленная ошибка, а не порог текущего взгляда; границы рассчитываются численно. Для пакетных данных и известного размера выборки обычно лучше GST, для потока — always-valid методы.
- Эффект теста, остановленного рано по эффективности, систематически завышен — тем сильнее, чем раньше остановка (та же механика, что у ошибки M).
- Futility не раздувает ошибку первого рода, немного снижает мощность и экономит трафик прежде всего тогда, когда эффекта нет.
- Conditional power считается под MDE, а не под наблюдаемым трендом; β-spending — единственный из методов futility, который контролирует ошибку второго рода по построению; non-binding — безопасный дефолт.
- Futility-стоп означает «эффект размера MDE вряд ли докажем», а не «эффекта нет».

## Вопросы для обсуждения

1. Продакт предлагает компромисс: «Смотрю на дашборд каждый день, но остановлю тест, только если p-value ниже 0.05 три дня подряд». Это решает проблему подглядывания? Что вы ему ответите?
2. Команда с небольшим трафиком прочитала, что SPRT в среднем тратит заметно меньше наблюдений, чем фиксированный тест той же точности, и хочет перевести на него все A/B-тесты. Какие возражения вы выдвинете?
3. Продакт просит: «Давайте GST, но на ранних взглядах обычный порог 1.96, чтобы быстро ловить победы, а в конце построже». Можно ли так, и чем команда за это заплатит?
4. В сравнении методов mSPRT при 10 взглядах заметно проиграл GST по мощности. Значит ли это, что mSPRT хуже и его стоит выкинуть из платформы?
5. GST с O'Brien–Fleming остановил тест на третьем взгляде из десяти: прирост конверсии +3 п.п. при MDE 2 п.п. Продакт хочет заложить +3 п.п. в прогноз выручки. Что вы ему скажете?
6. Тест прошёл половину запланированной выборки, наблюдаемый эффект вдвое меньше MDE. Продакт: «Посчитай шанс, что тест станет значимым, если дальше всё пойдёт так же, как сейчас, и если шанс маленький — закрываем». Какой эффект вы подставите в такой расчёт и почему?
7. Руководитель аналитики хочет binding-границы: при том же α efficacy-граница ниже, значит, тест чаще побеждает. Но бизнес иногда просит «дать тесту ещё недельку» после futility-стопа. Что выберете и почему?
8. Две команды думают о futility. У первой большинство гипотез «пустые» (по истории тестов около трети дают эффект), у второй каждая гипотеза долго прорабатывается, и эффект почти всегда есть. Кому futility выгоднее и что каждая теряет?
9. Тест остановлен по futility. Продакт пишет в итоговом отчёте: «Доказали, что фича не влияет на метрику, выпиливаем». Что вы исправите в формулировке и в решении?

## Правило для команды

1. Политика остановки фиксируется до старта и записывается в дизайн: метод, расписание взглядов в долях информации, spending-функции, максимальный размер выборки. Дашборд с обычным p-value — для поиска поломок, а не для решения.
2. Метод — под режим данных. Взгляды пакетами и предсказуемый размер выборки — GST с O'Brien–Fleming-подобной функцией; поток, неизвестный горизонт и быстрый отлов деградаций — mSPRT или always-valid p-value.
3. Futility — non-binding β-spending с поздним расходом β (по умолчанию βt³), первый futility-взгляд не раньше 30–50% информации; если считаем conditional power, то только под MDE, а не под наблюдаемым трендом.
4. Futility-стоп в отчёте — «эффект размера MDE вряд ли докажем», а не «эффекта нет»; для «эффекта нет» — тест эквивалентности.
5. Эффект рано остановленного теста в прогноз как есть не берём: используем осторожную оценку или интервал, при больших ставках подтверждаем на holdout.
6. Операционные характеристики проверяем симуляцией до запуска: α, мощность, доля futility-стопов и средний объём данных при отсутствии эффекта и при эффекте MDE.

## Источники

Последовательное тестирование:
- Wald A. [Sequential Tests of Statistical Hypotheses](https://www.jstor.org/stable/2235829), 1945; Wallis A. [The Statistical Research Group, 1942–1945](https://doi.org/10.1080/01621459.1980.10477469).
- [Sequential Probability Ratio Test (конспект лекции)](https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture9.pdf); [Experiments, peeking and optimal stopping](https://towardsdatascience.com/experiments-peeking-and-optimal-stopping-954506cec665/).
- Johari R., Pekelis L., Walsh D. [Always Valid Inference: Bringing Sequential Analysis to A/B Testing](https://arxiv.org/pdf/1512.04922); [презентация](https://simons.berkeley.edu/sites/default/files/docs/5572/simonsrj.pdf).
- Johari R. et al. [Peeking at A/B Tests: Why It Matters, and What to Do About It](https://kdd.org/kdd2017/papers/view/peeking-at-ab-tests-why-it-matters-and-what-to-do-about-it), KDD 2017.
- Howard S. et al. [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240).
- [Последовательный анализ в AB-тестировании: mSPRT](https://habr.com/ru/articles/991582/); [mSPRT: практическое объяснение](https://websitelytics.top/blog/post/msprt/); [mSPRT в Райффайзенбанке (видео)](https://www.youtube.com/watch?v=B5CfU2jV-nY&t=483s).
- [Group Sequential Testing: объяснение](https://habr.com/ru/companies/tbank/articles/876984/); Courthoud M. [Group Sequential Testing](https://matteocourthoud.github.io/post/group_sequential_testing/).
- Spotify. [Choosing a Sequential Testing Framework — Comparisons and Discussions](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions); [Fixed-Power Design](https://arxiv.org/abs/2405.03487); Zalando. [YEAST](https://arxiv.org/pdf/2406.16523v1); [Corrected-alpha Approach](https://arxiv.org/pdf/1905.10493).

Futility-анализ:
- Jennison C., Turnbull B. W. [Group Sequential Methods with Applications to Clinical Trials](https://doi.org/10.1201/9780367805326).
- Pampallona S., Tsiatis A. A. [Group sequential designs with provision for early stopping in favor of the null hypothesis](https://zbmath.org/pdf/00696996.pdf).
- Lan K. K. G., DeMets D. L. [Discrete sequential boundaries for clinical trials](https://zbmath.org/pdf/0543.62059.pdf).
- Lan K. K. G., Wittes J. T. The B-value: a tool for monitoring data. Biometrics, 1988.
- Spiegelhalter D. J., Freedman L. S., Parmar M. K. B. [Bayesian approaches to randomized trials](https://academic.oup.com/jrsssa/article/157/3/357/7106848).
- Proschan M. A., Lan K. K. G., Wittes J. T. [Statistical Monitoring of Clinical Trials: A Unified Approach](https://doi.org/10.1007/978-0-387-44970-8).
- Wassmer G., Brannath W. [Group Sequential and Confirmatory Adaptive Designs in Clinical Trials](https://doi.org/10.1007/978-3-319-32562-0).
- [Одинаковые данные, разные p-value: почему важно правило остановки](https://habr.com/ru/articles/1084294/); [Проблема подглядывания и последовательное А/Б тестирование](https://temofeev.com/info/articles/problema-podglyadyvaniya-i-posledovatelnoe-a-b-testirovanie/).
