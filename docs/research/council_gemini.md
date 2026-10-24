# Gemini, раунд 1 (Авто, free) — ключевое
Ответ содержит ДВА черновика подряд (видимо, перегенерация): 
- черновик A: Python + FastAPI + React/PixiJS (WebGL/WebGPU) в Tauri или локальном браузере; превью шейдерами на фронте, экспорт тайлами на бэкенде; пакет .lpk (zip: manifest.json, rules.json, brushes/, textures/, symbols/); правила с wildcards/regex, атрибуты ВИД/D.
- черновик B (итоговый): ДЕСКТОП PySide6/Qt6. Аргумент против веба: гигантские растры A1 300 dpi, файловая система, вызов ODA CLI. Превью: wgpu-py/PyOpenGL, экспорт: NumPy/Numba/OpenCV тайлами.
- Пакет .ls-pack: manifest.json, rules.json, brushes/, papers/, symbols/trees|shrubs (SVG/PNG).
- Правила: каскад — точные атрибуты → regex слоя → fallback по свойствам DWG; шаблоны Land F/X, Vectorworks, ГОСТ.
- Обратно в DWG: PNG рядом с чертежом + IMAGE в слое _LANDSCAPE_RENDER_UNDERLAY по bbox VIEWPORT; при пересборке PNG перезаписывается, CAD обновляет ссылку.
- NPR: отказ от полного Kubelka-Munk → Beer-Lambert + краевое затемнение; jitter перлином; 20–30% белой бумаги; толщины: здания 0.5–0.7, кроны 0.25–0.35, штриховка 0.09–0.18 мм; десатурация до 35–50%.
- Пайплайн: paper → area fill → planting → cast shadow (multiply) → line art.
- Roadmap: MVP (PySide6, DXF, NumPy рендер, таблица правил, PNG 300dpi) → v1 (ODA DWG, IMAGE в DWG, паки, пресеты LandFX/ГОСТ, wgpu превью) → v2 (нейросеть для вариаций символов, libmypaint, кистевое подрисовывание).
- Риски: DWG (ODA, иначе «сохраните как DXF»), OOM на A0 (тайлы), контрибьюторы (художники делают паки без кода).

# Gemini, раунд 2 — окончательное решение
- Стек: Tauri (React+TS+PixiJS) + Python FastAPI sidecar (PyInstaller). Аргументы: больше фронтенд/графических контрибьюторов чем Qt; один .msi; тяжёлый тайловый экспорт на Python; ODA через subprocess.
- manifest.json: id, name, version, author, description, target_scale_range [50,500], compatibility ">=1.0.0", assets{rules,brushes,textures,symbols}.
- rules.json: rules[{id, priority, selector{layer_regex | block_name regex | attributes{D:{op,value}}}, action{element_type: polygon|texture_fill|organic_fill|watercolor_pool|plant_symbol|line, symbol_path, size_source:"attribute", size_attr:"D", style{...}}}], fallback ".*" priority 10.
- Проект .ls-proj (JSON): format_version, dwg_path, layout_name, viewport_handle, style_pack, render_settings{dpi, seed, sun_angle_deg, paper_texture}, overrides{<handle>: {...} | {hidden:true}}.
- Автотесты качества: доля белой бумаги ≥65–75% (L≥245) — СЛИШКОМ много для полноцветной акварели, спорно; медиана S ≤40%; иерархия толщин через distance transform.

# Gemini, раунд 3 — ответ на позиции ChatGPT
1) Стек: ПРИНЯЛ PySide6 (один разработчик + ИИ: нет IPC Node↔Python, нет sidecar, один .exe).
2) DWG: частично — против AutoLISP как единственного пути (боль под каждый CAD). Предлагает: DWG→DXF (ODA) → ezdxf добавляет IMAGE/IMAGEDEF в paper space по углам viewport → пользователю готовый DXF, который откроют большинство CAD.
3) Качество: ПРИНЯЛ критику; белая бумага (L≥240) 25–40%, golden + SSIM ≥0.88, экспертная оценка.
4) Правила: ПРИНЯЛ glob, приоритет→специфичность, 3 статуса, профили отдельными пакетами.
5) Добавить: статусы существующие/демонтируемые/новые; автоэкспликация растений (шт, D, виды); сезонность палитры; ручные оверрайды по клику (символ, тень); надписи и размеры CAD поверх подачи.
