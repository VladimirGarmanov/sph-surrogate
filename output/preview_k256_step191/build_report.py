"""Bundle the measured rollout diagnostics into an offline report."""
from pathlib import Path
import csv
import html
import json
import zipfile

OUT=Path(__file__).resolve().parent
s=json.loads((OUT/"summary_metrics.json").read_text())
a=json.loads((OUT/"audit_metrics.json").read_text())
assert s["sha256"] == a["sha256"]
assert abs(s["per_frame"][-1]["position_rmse_mm"]-a["position_m"]["all"]["final"]["error_vector_rmse"]*1000)<1e-12
last=s["per_frame"][-1]
vel=a["velocity_m_s"]["all"]["final"]
stress=a["increments"]["stress_6vector_Pa"]["all"]
q=s["final_position_percentiles_mm"]

sections=[
("scope","Что именно проверяем",[
    "Это прежняя модель K256, checkpoint 191, материал phi30_c500. В файле 46 464 частицы и 10 предсказанных кадров: 101–110, время 2,02–2,20 с. Сеть получила истинную историю до кадра 100 (2,00 с), затем использовала собственный прогноз грунта. Известное движение границ продолжает задаваться извне.",
    "Таким образом, проверено только 0,20 с самостоятельного прогноза: первые две секунды модель здесь не моделировала. По присланному журналу завершено 191 из 3927 обновлений — 4,86% первой эпохи; это ещё ранний результат. Нового K32 в этом архиве нет."]),
("position","0,503 мм в среднем скрывают редкие большие ошибки",[
    f"В конце окна векторная RMSE положения всех частиц равна {last['position_rmse_mm']:.3f} мм. Для подвижных частиц она уже {last['active_position_rmse_mm']:.3f} мм. Подвижность определена только по истинной скорости: не меньше 1 мм/с в рассматриваемом кадре. {100-s['active_fraction_percent']:.2f}% всех пар «частица–кадр» не достигают этого порога. Усреднение по всем частицам уменьшает видимую величину ошибки активной области; при этом именно подвижные частицы дают около 98% суммы квадратов ошибок положения на всём окне.",
    f"У половины частиц конечная ошибка не больше {q['p50']:.3f} мм, у 95% — {q['p95']:.3f} мм, у 99% — {q['p99']:.3f} мм. Но максимум составляет {q['max']:.2f} мм; {s['final_position_above_10mm']} частиц превышают 10 мм. Строка терминала «10/10 до порога 10 мм» относится к общей RMSE, а не к каждой частице.",
    "Худшая частица — ID 21551. Между кадрами 101 и 110 её истинный z вырос на 42,20 мм, предсказанный — на 3,35 мм. Здесь большая ошибка отражает пропущенное быстрое движение эталона; она сама по себе не означает, что модель разбрасывает весь грунт. Три примера на графиках выбраны по явному правилу: медиана и 95-й перцентиль ошибки среди подвижных частиц, затем максимум среди всех."]),
("velocity","Модель слабо обновляет скорость",[
    "Прогноз не равен нулевому движению. На девяти доступных переходах 101→102, …, 109→110 ошибка приращений положения по MSE на 30,59% ниже контроля Δx = 0. Но это диагностика изменений между кадрами, а не доказательство превосходства полного rollout над частицами, замороженными в момент старта.",
    f"RMS собственных изменений скорости модели — {s['velocity_increment_rms_mm_s']['model']:.3f} мм/с за переход, у солвера — {s['velocity_increment_rms_mm_s']['truth']:.3f} мм/с. Модель почти переносит скорость из предыдущего кадра. Ошибка изменения скорости по MSE на {-s['velocity_increment_zero_baseline_skill_percent']:.2f}% хуже контроля Δv = 0.",
    f"На последнем кадре RMSE скорости всех частиц — {vel['model_rmse']*1000:.3f} мм/с. Простой прогноз v = 0 даёт {vel['zero_baseline_rmse']*1000:.3f} мм/с: MSE модели выше на {-vel['mse_skill_vs_zero']*100:.2f}%. Для подвижной области результат тоже хуже нулевой скорости: 15,900 против 14,239 мм/с. Это относится к концу окна; в среднем по всем десяти кадрам модель ещё лучше этого простого контроля.",
    "Появляется и лишняя скорость на спокойном фоне. Среди частиц с истинной скоростью меньше 1 мм/с на кадре 110 медиана модуля скорости равна 0,062 мм/с у солвера и 0,429 мм/с у модели. На пространственном срезе этот дрейф виден как положительный фон вертикальной скорости. Причину по одному rollout установить нельзя; проверка на истинной истории поможет отделить локальную ошибку от накопления ошибок."]),
("pressure","Давление: 1,27% к оценке, 28,43% к кривой солвера",[
    "p_pred и p_gt вычислены одной приближённой формулой: минус среднее напряжение p33 частиц в диске радиуса 0,15 м и слое толщиной 0,04 м под плитой. В первом случае используются предсказанные частицы, во втором — истинные. p_reference — отдельная кривая давления из CSV солвера. Она не является лабораторным измерением.",
    "Ошибка модели относительно оценки по истинным частицам — 1,272%, или 0,695 кПа MAE. Относительно сохранённой кривой солвера — 28,430%, или 12,126 кПа MAE. При этом сама формула на истинном грунте отличается от кривой на 28,047%. Даже до первого прогноза, в кадре 100, оценка равна 57,320 кПа, а CSV — 44,018 кПа.",
    "Поэтому малые 1,27% показывают хорошее совпадение с конкретной оценкой по напряжениям. Они не означают точность итогового давления беваметра 1,27%. Источник расхождения формулы и CSV следует проверять отдельно; исходного CSV, данных плиты и их временного выравнивания в скачанном архиве нет.",
    f"Шесть компонент напряжения на последнем кадре имеют RMSE от 0,422 до 1,904 кПа. Однако MSE их приращений на {-stress['mse_skill_vs_zero']*100:.2f}% хуже контроля «напряжения не меняются». Близость абсолютных напряжений ещё не доказывает правильность динамики. Нулевые ошибки rho, pc, Ev и Sv здесь тривиальны: в эталоне rho = 1600, остальные три поля равны нулю на всём окне."]),
("cost","За 0,20 секунды процесса потрачено около девяти минут",[
    f"Все десять шагов заняли {s['compute_seconds_total']:.1f} с вычислений. Основное среднее в CLI исключает первый шаг: {s['mean_seconds_per_frame_excluding_first']:.2f} с/кадр. В архиве нет стадийного профиля: это запуск без --profile. Экспорт NPZ и построение картинки в это время не входят.",
    "При ранее указанном времени солвера около 4,72 с/кадр текущий K256 примерно в 11,6 раза медленнее. Число 4,72 взято из контекста предыдущей проверки, а не из этого NPZ. Новый K32 ещё нужно измерить отдельно. По предыдущему сообщению обучение перед проверкой было остановлено; сам архив модель GPU и фоновую нагрузку не сохраняет."]),
("next","Что проверять дальше",[
    "1. Сохранить этот результат как контроль K256. Для тех же весов, материала и кадров выполнить teacher forcing: каждый шаг получает истинную историю. Это покажет, насколько ошибки растут именно от использования собственного прогноза.",
    "2. В следующую выгрузку добавить истинное состояние кадра 100. Тогда можно честно сравнить полный rollout с замороженным состоянием и постоянной начальной скоростью от одного и того же старта. Сейчас файл содержит частицы только с кадра 101; подменять начальный кадр нельзя.",
    "3. Новый K32 сравнить на тех же окнах, с тем же горизонтом и шумом. Проверять не только среднюю ошибку: отдельно подвижные частицы, скорости, приращения напряжений, p99/max и давление относительно обеих сохранённых кривых. После короткой проверки нужны более длинные траектории и другие отложенные материалы."]),
]

metrics=[
    ("RMSE положения, все частицы",f"{last['position_rmse_mm']:.3f} мм"),
    ("RMSE положения, подвижные",f"{last['active_position_rmse_mm']:.3f} мм"),
    ("95-й / 99-й перцентиль ошибки положения",f"{q['p95']:.3f} / {q['p99']:.3f} мм"),
    ("Максимальная ошибка положения",f"{q['max']:.2f} мм"),
    ("Частиц с ошибкой больше 10 мм",str(s['final_position_above_10mm'])),
    ("RMSE скорости модели / контроль v = 0",f"{vel['model_rmse']*1000:.3f} / {vel['zero_baseline_rmse']*1000:.3f} мм/с"),
    ("Медиана скорости спокойного фона, модель / солвер","0,429 / 0,062 мм/с"),
]

md=["# Проверка суррогата: K256, шаг 191", "",
    "**Прогноз воспроизводит часть движения, но пока недостаточно точно обновляет скорости и напряжения. Малый общий RMSE скрывает ошибки активных частиц.**", "",
    "[Открыть все графики в браузере](index.html) · [Анимация](motion.gif)", "",
    "![Обзор](overview.png)", "", "## Конечный кадр 110", "",
    "| Показатель | Значение |", "|---|---:|"]
md += [f"| {label} | {value} |" for label,value in metrics]
for ident,title,paras in sections:
    md += ["",f"## {title}",""]
    for para in paras: md += [para,""]
    if ident=="position": md += ["![Траектории выбранных частиц](particle_tracks.png)",""]
    if ident=="velocity": md += ["![Динамика](dynamics.png)","", "![Пространственный срез](spatial.png)",""]
md += ["## Методика и файлы", "",
       "RMSE положения и скорости вычислена по норме трёхмерной ошибки: sqrt(mean(sum(error²))). Начальный истинный кадр исключён. Порог активности задан по солверу отдельно в каждом кадре. Процентили относятся к нормам ошибок частиц, а не к ошибкам отдельных координат.", "",
       "Все проценты давления в основном отчёте соответствуют CLI: 100 × mean(abs(error)) / mean(abs(reference)). Это не среднее покадровых процентов MAPE. Подробный независимый аудит также содержит MAPE, поэтому его числа немного отличаются.", "",
       "Срез включает 1054 фиксированных ID, выбранных по истинному кадру 101: 0 ≤ y ≤ 20 мм. Реальный масштаб координат, одинаковая цветовая шкала для истинного и предсказанного vz во всех кадрах. Плита и стенки не дорисованы: их координат в NPZ нет. В анимации темп показа условный; перемещения не усилены.", "",
       "Новый анализ не запускал нейросеть и не менял веса. Поля проверены на конечность, ID — на уникальность; независимый пересчёт совпал с сохранёнными RMSE. По единственному короткому rollout нельзя доказать устойчивость всей траектории или улучшение относительно другого checkpoint.", "",
       "- [Метрики основного отчёта](summary_metrics.json)",
       "- [Независимый численный аудит](audit_metrics.json)",
       "- [Показатели по кадрам](per_frame.csv)",
       "- [Ошибки всех 16 полей](features.csv)",
       "- [Три выбранные частицы](selected_particles.csv)", "",
       "Источник: `checkpoints/preview_k256.npz`.",f"SHA-256: `{s['sha256']}`.", "",
       "Воспроизведение из корня проекта:", "", "```bash",
       ".venv/bin/python output/preview_k256_step191/audit_metrics.py",
       "MPLCONFIGDIR=/tmp/sph-preview-mpl .venv/bin/python output/preview_k256_step191/analysis.py",
       "MPLCONFIGDIR=/tmp/sph-preview-mpl .venv/bin/python output/preview_k256_step191/spatial_analysis.py",
       ".venv/bin/python output/preview_k256_step191/build_report.py", "```", ""]
(OUT/"REPORT_RU.md").write_text("\n".join(md))

def paragraphs(items):
    return "".join(f"<p>{html.escape(item)}</p>" for item in items)


def figure(stem,caption):
    return f'<figure><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(caption)}" loading="lazy"></a><figcaption>{html.escape(caption)} <a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></figcaption></figure>'

body='''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Грунт: модель против солвера · K256 / 191</title>
<style>
:root{color-scheme:light;--ink:#24323f;--muted:#607182;--blue:#207b9b;--paper:#f8fafb}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 system-ui,-apple-system,sans-serif}
main{max-width:1420px;margin:auto;padding:42px 28px 70px}header{max-width:1000px;margin-bottom:30px}h1{font-size:clamp(30px,4vw,52px);line-height:1.14;letter-spacing:-.025em;margin:.25em 0 .45em}
h2{font-size:clamp(23px,2.4vw,31px);line-height:1.3;margin-top:2.2em}p{max-width:960px;margin:.9em 0}.eyebrow{color:var(--muted);font-size:14px;letter-spacing:.06em;text-transform:uppercase}
.lead{font-size:21px;line-height:1.5}a{color:var(--blue);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:15px;margin:22px 0}figure{margin:28px 0 40px}figure img{display:block;width:100%;height:auto}figcaption{font-size:14px;line-height:1.5;color:var(--muted);margin-top:10px;max-width:1050px}figcaption a{white-space:nowrap}
table{border-collapse:collapse;width:100%;max-width:1000px;font-size:16px;margin:24px 0}th,td{padding:11px 12px;border-bottom:1px solid #dce3e8;text-align:left;vertical-align:top}th:last-child,td:last-child{text-align:right;font-variant-numeric:tabular-nums}th{font-weight:600;color:var(--muted)}
details{margin:30px 0}summary{cursor:pointer;color:var(--blue);font-weight:600}code{font-size:.85em;overflow-wrap:anywhere}footer{margin-top:48px;padding-top:20px;border-top:1px solid #dce3e8;font-size:14px;color:var(--muted)}.note{border-left:3px solid #a3b6c5;padding-left:18px}.links{display:flex;gap:10px 22px;flex-wrap:wrap}section{scroll-margin-top:20px}
@media(max-width:600px){main{padding:24px 14px 44px}body{font-size:16px}th,td{padding:9px 5px}table{font-size:14px}.lead{font-size:18px}}
@media print{nav,details,footer .links{display:none}body{background:white}main{padding:0}figure{break-inside:avoid}h2{break-after:avoid}}
</style></head><body><main><header><div class="eyebrow">SPH surrogate · диагностика раннего обучения</div>
<h1>Что модель уже повторяет,<br>а где теряет движение</h1>
<p class="lead">На коротком прогнозе средняя ошибка положения невелика. Но скорости обновляются слабо, в спокойном грунте появляется дрейф, а редкие быстрые движения пропускаются.</p>
<p>K256 · checkpoint 191 · phi30_c500 · 46 464 частицы · 0,20 с после истинного старта</p>
<nav aria-label="Разделы"><a href="#position">Ошибки частиц</a><a href="#velocity">Динамика</a><a href="#spatial">Срез и анимация</a><a href="#pressure">Давление</a><a href="#cost">Скорость расчёта</a><a href="#next">Следующая проверка</a></nav></header>'''
body+=figure("overview","Главные показатели: общая и активная ошибки, два разных эталона давления, контроль нулевой скорости и хвост распределения ошибок.")
body+='<h2>Конечный кадр 110</h2><table><thead><tr><th>Показатель</th><th>Значение</th></tr></thead><tbody>'
body+=''.join(f'<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>' for label,value in metrics)
body+='</tbody></table>'
for ident,title,paras in sections:
    body+=f'<section id="{ident}"><h2>{html.escape(title)}</h2>'+paragraphs(paras)
    if ident=="position": body+=figure("particle_tracks","Три выбранных ID: истинная и предсказанная вертикальная траектория и скорость. Масштабы колонок различаются.")
    if ident=="velocity":
        body+=figure("dynamics","Сравнение с простыми прогнозами: нулевые приращения, нулевая скорость, распределение скоростей спокойного фона и время расчёта.")
        body+='<div id="spatial"><h2>Срез грунта: одинаковые ID и шкалы</h2>'
        body+=paragraphs(["Показан фиксированный слой 0 ≤ y ≤ 20 мм по истинному кадру 101: 1054 частицы. Слева истинная вертикальная скорость, в центре прогноз, справа ошибка положения. Одинаковая цветовая шкала показывает положительный дрейф модели на почти спокойном фоне."])
        body+=figure("spatial","Геометрия в реальном масштабе; координат плиты и стенок в архиве нет, поэтому они не дорисованы. Ошибки нанесены на истинные координаты.")
        body+='<details><summary>Посмотреть анимацию среза: солвер и модель рядом</summary><figure><img src="motion.gif" alt="Десять кадров центрального среза: истинная и предсказанная вертикальная скорость"><figcaption>Кадры 101–110; время 2,02–2,20 с. Темп показа условный, перемещения не усилены. Общая шкала скорости во всех кадрах. <a href="motion.gif">Открыть GIF отдельно</a></figcaption></figure></details></div>'
    body+='</section>'
body+='<footer><p class="note">Анализ рассчитан из сохранённого NPZ без запуска модели. Все числовые массивы конечны; ID уникальны; независимый пересчёт подтверждает экспортированные RMSE. Исходный кадр 100 не содержит сохранённых состояний частиц. Выводы ограничены одним материалом и десятью прогнозами.</p>'
body+='<p>Проценты давления: 100 × mean(|ошибка|) / mean(|эталон|). Векторная RMSE: sqrt(mean(sum(ошибка²))). Подробный аудит дополнительно содержит MAPE, чьи значения немного отличаются.</p>'
body+='<div class="links"><a href="REPORT_RU.md">Полный текст отчёта</a><a href="summary_metrics.json">Основные метрики JSON</a><a href="audit_metrics.json">Независимый аудит JSON</a><a href="per_frame.csv">По кадрам CSV</a><a href="features.csv">Все 16 полей CSV</a><a href="selected_particles.csv">Три частицы CSV</a><a href="../preview_k256_step191.zip">Скачать весь отчёт ZIP</a></div>'
body+=f'<p>Источник: checkpoints/preview_k256.npz<br>SHA-256: <code>{s["sha256"]}</code></p></footer></main></body></html>'
(OUT/"index.html").write_text(body)
with zipfile.ZipFile(OUT.parent/(OUT.name+".zip"),"w",compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.iterdir()):
        if path.is_file() and not path.name.startswith("motion_"):
            archive.write(path,arcname=OUT.name+"/"+path.name)
print(f"Report: {OUT/'index.html'}")
