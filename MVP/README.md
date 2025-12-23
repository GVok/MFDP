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


### 3. Judge (оценка и ранжирование)

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


### 4. Хранение данных

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


### 5. Очереди и воркеры
- RabbitMQ
- ml_worker:
    - слушает очередь
    - выполняет inference
    - не зацикливается на фатальных ошибках (404, 401)
    - только CPU для judge (предсказуемо и стабильно)


---


### 6. Docker / Dev-окружение
Контейнеры:
- app — FastAPI
- worker — ML-воркер
- ollama — локальный LLM
- rabbitmq
- postgres
- nginx

Запуск через docker `compose up --build`


--


### 7. Пользователи, биллинг и подписки

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


### 8. Web UI (FastAPI + Jinja2)

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


---

### 9. Тесты

Покрыты критические части сервиса.


---


## Ограничения MVP
- Judge на CPU (Технические ограничения машины)
- Local SD ограничен по количеству изображений
- Brand style — короткий (style_short), не полный brandbook
