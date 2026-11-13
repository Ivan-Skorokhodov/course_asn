# Подготовка AS-графа в CSR

1. Поместите снимок `*.as-rel2.txt` или `*.as-rel2.txt.gz` CAIDA в `data/raw/`.
2. Установите зависимости: `python -m pip install -r requirements.txt`.
3. Постройте граф: `python -m src.graph build data/raw/20261101.as-rel2.txt --out data/processed`.
4. Проверьте: `python -m src.graph inspect --dir data/processed`.
5. Тесты: `python -m unittest discover -s tests -v`.

Описание выбора формата и структуры файлов — в [HANDOFF.md](HANDOFF.md).
