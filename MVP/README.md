# MVP: Prompt Enhancer + Image Generator + Judge (CLIP + Aesthetic)

Учебный / MVP ML-проект для SMM и малого бизнеса.

Сервис принимает «сырой» пользовательский промт, улучшает его до англоязычного описания в бренд-стиле, генерирует изображения, а затем автоматически оценивает и ранжирует результаты по соответствию промту и эстетике.


---


## Общая архитектура

Client (Browser / Postman)
        ↓
FastAPI (app, web)
        ↓
RabbitMQ
        ↓
ML Worker
  ├─ Prompt Enhancer (Ollama / mock)
  ├─ Image Generator (local SD / mock)
  └─ Judge (CLIP + Aesthetic)
        ↓
PostgreSQL


---

## Quickstart (Docker)

### Требования
- Docker + Docker Compose
- Порты: 80 (web), 15672 (RabbitMQ UI), 5432 (Postgres), 11434 (Ollama)

### Запуск
1) Создайте `.env` в корне (пример ниже).  
2) Запустите сервисы:

```bash
docker compose up --build
# масштабирование воркера при необходимости:
# docker compose up --build --scale worker=2
```

3) Настроить миграции

```bash
docker compose exec app alembic upgrade head
```

---

## Где открыть

- Web UI: http://localhost/
- RabbitMQ Management: http://localhost:15672/ (guest/guest)
- Postgres: localhost:5432
- Ollama: http://localhost:11434

---

## Конфигурация (.env)

### Минимально необходимые переменные
- DATABASE_URL
- RABBITMQ_URL
- OLLAMA_URL, OLLAMA_MODEL (если используете enhance_backend=ollama)
- CLIP_MODEL_NAME, AES_WEIGHTS_URL (для Judge)
- LOCAL_SD_MODEL и параметры (если используете image_backend=local)

### Пример .env

```bash
APP_ENV=development
APP_DEBUG=true
APP_SECRET_KEY=supersecret

POSTGRES_USER=ml_user
POSTGRES_PASSWORD=ml_password
POSTGRES_DB=ml_service
POSTGRES_HOST=database
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg2://ml_user:ml_password@database:5432/ml_service

RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_KEEP_ALIVE=30s
OLLAMA_TIMEOUT_SEC=120
ENHANCE_MAX_TOKENS=200

CLIP_MODEL_NAME=openai/clip-vit-base-patch32
AES_WEIGHTS_URL=https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth
AES_MIN=1.0
AES_MAX=10.0
FINAL_W_CLIP=0.7
FINAL_W_AES=0.3

LOCAL_SD_MODEL=runwayml/stable-diffusion-v1-5
LOCAL_SD_WIDTH=512
LOCAL_SD_HEIGHT=512
LOCAL_SD_STEPS=20
LOCAL_SD_GUIDANCE=7.0
LOCAL_SD_SEED=123

```

## Что реализовано на данный момент

### 1. Prompt Enhancer
**Назначение:**
Преобразует короткий, шумный пользовательский промт (RU / mixed)
в чистый, детализированный **English prompt** для image generation.

**Варианты backend:**
- `ollama` — через локальный Ollama (Qwen2.5-1.5B-Instruct)
- `mock` — без LLM, используется `cleaned_prompt`

**Фичи:**
- очистка текста (`clean_text`)
- детекция платформы (`instagram`, `banner`, `website`, и т.д.)
- детекция цели (`promotion`, `branding`, `product_shot`, …)
- единый системный промт с бренд-стилем (фиолетовый минимализм)
- результат сохраняется в БД (`enhanced_prompt`)


---


### 2. Brand Profiles

Пользователь может создавать бренд-профили, которые управляют стилем генерации.

#### Brand Profile содержит:
- name — имя профиля
- style_short — короткое нормализованное описание визуального стиля


#### Особенности:
- лимит профилей зависит от тарифа
- стиль встраивается в system prompt
- если профиль не выбран — используется нейтральный enhancer


---


### 3. Генерация изображений

#### Mock-генерация
- генерирует 1–4 детерминированных PNG
- используется для дешёвого MVP и тестов
- гарантирует разнообразие (без рандомного шума)

#### Local Stable Diffusion
- модель: `runwayml/stable-diffusion-v1-5`
- backend: `local`
- ограничения:
  - ≤ 2 изображения за запрос
  - CPU / GPU auto-detect
- параметры
  - `width`, `height`
  - `steps`
  - `guidance`
  - `seed`
- все изображения **нормализуются через PIL и сохраняются в PNG**


---


### 4. Judge (оценка и ранжирование)

Используется реальный ML-судья, а не заглушка.

#### Модели:
- CLIP: `openai/clip-vit-base-patch32`
- Aesthetic: LAION Aesthetic Predictor (linear head)

#### Метрики:
- `clip_score` → нормализованный [0, 1]
- `aesthetic_score` → raw → нормализованный [0, 1]

#### Финальный скор:

S = 0.7 * clip_n + 0.3 * aes_n

#### Что делает судья:

- считает скоры для каждой картинки
- сортирует по final_score
- назначает rank
- сохраняет всё в БД (predictions)


---


### 5. Хранение данных

#### PostgreSQL
Основные таблицы:
- `ml_requests`
- `ml_tasks`
- `predictions`

В predictions.gen_meta сохраняется полный контекст:
- параметры генерации
- backend (mock / hf)
- скоры (clip_n, aes_raw, aes_n, final)
- информация о judge


---


### 6. Очереди и воркеры
- RabbitMQ
- ml_worker:
    - слушает очередь
    - выполняет inference
    - не зацикливается на фатальных ошибках (404, 401)
    - только CPU для judge (предсказуемо и стабильно)


---


### 7. Docker / Dev-окружение
Контейнеры:
- app — FastAPI
- worker — ML-воркер
- ollama — локальный LLM
- rabbitmq
- postgres
- nginx

Запуск через docker `compose up --build`


--


### 8. Пользователи, биллинг и подписки

#### Аутентификация
- регистрация / логин
- JWT в HttpOnly cookie
- Web + API

#### Wallet
- баланс в рублях
- пополнение
- списание за подписки

#### Подписки (plans)

Примеры фич:
- лимит изображений в месяц
- auto prompt enhance
- auto best image selection
- лимит brand profiles
- лимит LoRA (заготовка под будущее)
- UI биллинга и подписок реализован в `/account`.


---


### 9. Web UI (FastAPI + Jinja2)

Реализован полноценный UI:
- `/` — landing
- `/login`, `/register`
- `/app` — генерация и история
- `/app/tasks/{id}` — статус задачи
- `/app/requests/{id}` — результаты
- `/account` — единая точка управления:
  - профиль
  - кошелёк
  - подписка
  -brand profiles

### 10. Тесты

Покрыты критические части сервиса.


---


## Ограничения MVP
- Judge на CPU (Технические ограничения машины)
- Local SD ограничен по количеству изображений
- Brand style — короткий (style_short), не полный brandbook
