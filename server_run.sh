#!/bin/bash
# server_run.sh
# Универсальный скрипт для запуска обучения на сервере
# Использование: ./server_run.sh [путь_к_скрипту]
# Пример: ./server_run.sh src/train_mixed.py

# 1. Определяем корневую директорию проекта
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR" || exit 1

# 2. Конфигурация
# Скрипт по умолчанию - train_mixed.py (наиболее актуальный)
SCRIPT_NAME=${1:-"src/train_mixed.py"}
SCRIPT="$PROJECT_DIR/$SCRIPT_NAME"

# Создаем имя лог-файла на основе имени скрипта
SCRIPT_BASENAME=$(basename "$SCRIPT_NAME" .py)
LOG_FILE="$PROJECT_DIR/logs/${SCRIPT_BASENAME}_$(date +%s).log"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

# 3. Проверка окружения
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ОШИБКА: Виртуальное окружение не найдено по пути $VENV_PYTHON"
    echo "Пожалуйста, создайте окружение:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 4. Настройка переменных окружения
export PYTHONPATH="$PROJECT_DIR/src"
# Отключаем видеодрайвер для работы без GUI
export SDL_VIDEODRIVER=dummy

# 5. Создаем директорию для логов
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== Запуск обучения ==="
echo "Скрипт:      $SCRIPT"
echo "Лог:         $LOG_FILE"
echo "Директория:  $(pwd)"

# 6. Запуск
if [ ! -f "$SCRIPT" ]; then
    echo "ОШИБКА: Скрипт $SCRIPT не найден!"
    exit 1
fi

# Запускаем через nohup в фоне
nohup "$VENV_PYTHON" "$SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Процесс запущен с PID: $PID"
echo "Для просмотра логов: tail -f $LOG_FILE"
echo "Для остановки: kill $PID"
echo "======================="

