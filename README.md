# CAIDA AS-граф: иерархия Интернета и точки подключения DNS-anycast

Проект исследует структуру Интернета на уровне автономных систем (AS) и готовит кандидатов для будущей оптимизации точек подключения авторитативного DNS-anycast с помощью ACO.

**Важно:** выбранная ASN — потенциальная **сетевая точка подключения**, а не автоматически отдельный DNS-сервер или конкретный город. Полный CSR используется для маршрутов; BFS-дерево — только для изучения топологии.

## 1. Подготовка окружения

Команды выполняются **из корня репозитория**:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Если `.venv` ещё не создано: `python3 -m venv .venv`.

## 2. Построение полного графа (один раз для каждого снимка CAIDA)

Положить исходный `*.as-rel2.txt` или `*.as-rel2.txt.gz` в `data/raw/`, затем:

```bash
python -m src.graph build data/raw/20261101.as-rel2.txt.gz --out data/processed
python -m src.graph inspect --dir data/processed
```

При необходимости заменить имя файла в команде на фактическое. Получаются `graph.npz`, `edge_types.npy`, `asn_ids.npy` и `metadata.json` в `data/processed/`. Если снимок уже обработан, повторно строить CSR не нужно.

## 3. Полный цикл аналитики: три CSV

Выполнять **в указанном порядке**.

### Шаг 1. Метрики AS и дополнение CAIDA AS Rank

```bash
python -m src.as_analysis metrics --asrank-top 500
```

Создаёт `data/analysis/as_metrics.csv`: прямых клиентов, провайдеров и peers для всех AS. Для AS, совпавших с топ-500 CAIDA AS Rank, дополнительно заполняются рейтинг и Customer Cone. **Без интернета** можно выполнить `python -m src.as_analysis metrics`, но поля AS Rank останутся пустыми, а при отборе будут использоваться прямые клиенты вместо Customer Cone.

### Шаг 2. Первоначальные кандидаты без географии

```bash
python -m src.as_analysis candidates --large 40 --middle 80 --peering 80
```

Создаёт `data/analysis/candidates.csv` из трёх **пересекающихся** групп. Сумма 40 + 80 + 80 не обязана равняться количеству уникальных ASN.

### Шаг 3. Получить географические сведения PeeringDB

```bash
python -m src.peeringdb_sites --timeout 15 --retries 2
```

Создаёт `data/analysis/asn_sites.csv`. Данные берутся **для ASN из текущего `candidates.csv`**. Ответы API кэшируются в `data/analysis/.cache/peeringdb/`. Время выполнения зависит от числа ASN и API; повторный запуск обычно использует кэш.

Для проверки на пяти ASN **не перезаписывать основной результат**:

```bash
python -m src.peeringdb_sites --limit-asns 5 --out data/analysis/asn_sites_sample.csv
```

Для принудительного обновления запросов: `python -m src.peeringdb_sites --refresh`. Параметры `--timeout`, `--retries`, `--limit-asns` поддерживаются исправленной версией `src/peeringdb_sites.py`.

### Шаг 4. Повторно собрать кандидатов с географией

```bash
python -m src.as_analysis candidates \
  --large 40 --middle 80 --peering 80 \
  --sites data/analysis/asn_sites.csv
```

**Итоговые файлы:**

| CSV | Содержимое |
|---|---|
| `data/analysis/as_metrics.csv` | Все AS из CSR и их топологические показатели; частично CAIDA AS Rank |
| `data/analysis/asn_sites.csv` | Заявленное присутствие кандидатов в facilities и IX |
| `data/analysis/candidates.csv` | Отобранные уникальные AS, группы, топологические метрики и географическая сводка |

На **23.11.2026**, по результатам локального запуска: **184 кандидата, 11 441 запись PeeringDB**. Число записей — не число уникальных городов и не число DNS-серверов. При изменении параметров отбора эти значения изменятся.

## 4. Что изучить после получения CSV

### Быстрый просмотр файлов

```bash
head -n 3 data/analysis/as_metrics.csv
head -n 3 data/analysis/candidates.csv
head -n 3 data/analysis/asn_sites.csv
```

Показать первые 15 кандидатов по официальному AS Rank (если поля заполнены):

```bash
python - 1>&2 <<'PY'
import csv

with open('data/analysis/candidates.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r['asrank_rank'].strip()]
rows.sort(key=lambda r: int(r['asrank_rank']))

if not rows:
    print("Нет данных AS Rank в candidates.csv")

for r in rows[:15]:
    print(f"AS{r['asn']:>8}  rank={r['asrank_rank']:>5}  "
          f"cone={r['asrank_cone_asns']:>7}  "
          f"facilities={r['facility_count']:>4}  IX={r['ix_count']:>4}  "
          f"groups={r['groups']}")
PY
```

Показать первые 15 кандидатов по количеству peer-связей:

```bash
python - 1>&2 <<'PY'
import csv

with open('data/analysis/candidates.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: (-int(r['peer_degree']), int(r['asn'])))

for r in rows[:15]:
    print(f"AS{r['asn']:>8}  peers={r['peer_degree']:>5}  "
          f"customers={r['customer_degree']:>5}  "
          f"facilities={r['facility_count']:>4}  "
          f"countries={r['facility_countries']}")
PY
```

Для сравнения географии использовать прежде всего `facility_countries` и `facility_cities`: город IX не гарантирует физическое присутствие оборудования AS в этом городе (возможен remote peering).

### Исследовать иерархию provider → customer

```bash
python -m src.as_analysis hierarchy --as 3356 --depth 2 --max-nodes 60
```

Можно заменить `3356` на любую ASN из `candidates.csv`. Это **дерево обнаружения** по отношениям provider → customer, а не полный BGP Customer Cone.

### Посмотреть окрестность AS в браузере

```bash
python -m src.graph_neighbors --as 3356 --depth 2 --edge-cutoff 0 --max-nodes 300 --static
```

Открыть `out/as_neighbors.html`. Для исследования структуры удобен `--edge-cutoff 0`; фильтр `--edge-cutoff` отсекает связи **в ходе BFS**, поэтому может убирать целые ветви. Он не используется при расчёте маршрутов для ACO.

## 5. Следующий этап курсовой

1. Сопоставить AS с большим **Customer Cone** и большим `peer_degree`; изучить их иерархии и географию.
2. Оценить покрытие данных PeeringDB: не у каждой ASN обязательно есть запись; `facility` и `ix` трактовать раздельно.
3. Зафиксировать `S` (источники DNS-запросов), `D` (существующие точки подключения DNS), `q_s` (веса), `C` (допустимые AS-кандидаты), `k` и `α`.
4. На **полном** CSR посчитать маршрутные стоимости и выигрыш от подключения; затем реализовать Greedy и ACO с сохранением рейтинга **комбинаций** из `k` AS.
5. Для лучших комбинаций проверить географическое разнообразие и практическую реализуемость по PeeringDB. География пока **не влияет** на стоимость AS-level пути напрямую.

Архитектура, модель и ограничения — в [HANDOFF.md](HANDOFF.md).
