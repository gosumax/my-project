# Выбор главного агента

Карта текущей системы, проверенная перед изменениями:

| FILE | SETTING | CURRENT VALUE | WHAT IT CONTROLS |
|---|---|---|---|
| `.codex/config.toml` | `model`, `model_reasoning_effort` | `gpt-6.1-sol`, `medium` | Проектные defaults MAIN |
| `C:\Users\Admin\.codex\config.toml` | те же поля | `gpt-6.1-sol`, `high` | Пользовательские defaults MAIN |
| `.codex/config.toml` | `agents.default_subagent_model`, `agents.default_subagent_reasoning_effort` | `gpt-6.1-sol`, `low` | Default subagent |
| `.codex/config.toml` | `agents.enabled`, `agents.max_concurrent_threads_per_session` | `true`, `2` | Включение и concurrency дочерних агентов |
| `.codex/agents/explorer.toml` | модель, reasoning, роль, instructions | `gpt-6-luna`, `medium`, explorer | Read-only разведка и её prompt |
| `.codex/agents/tester.toml` | модель, reasoning, роль, instructions | `gpt-6.1-sol`, `medium`, tester | Проверки и их contract |
| `.codex/agents/reviewer.toml` | модель, reasoning, роль, instructions | `gpt-6.1-sol`, `medium`, reviewer | Read-only аудит и его prompt |
| `AGENTS.md` | Main agent, Routing | MAIN default и правила explorer/tester/reviewer | Владение задачами, routing, максимум два subagents |
| Codex runtime | registry, spawn | Нативные роли/collaboration tools | Создание и жизненный цикл; repo dispatcher отсутствует |

Для новой CLI-сессии из `D:\Новый парсер`:

```powershell
python scripts/start_codex.py --main-model gpt-6.1-sol --main-reasoning medium
python scripts/start_codex.py --main-model gpt-6-astra --main-reasoning high
```

Необязательное первое сообщение: `--prompt "Описание большой задачи"`.
Проверить команду без запуска сессии: добавить `--check-only`.
Python 3.11+ и `codex` в PATH обязательны.

Defaults MAIN уже отделены в существующем `.codex/config.toml`:

```toml
model = "gpt-6.1-sol"
model_reasoning_effort = "medium"
```

Меняйте только эти два верхнеуровневых поля для постоянного выбора MAIN.
CLI flags имеют приоритет над ними. Launcher всегда передаёт оба значения
нативному Codex: `-m MODEL -c 'model_reasoning_effort="LEVEL"'`.
При ошибке конфигурации или запуска launcher завершается с ошибкой;
собственного fallback на другую модель/уровень нет.
Доступность модели и поддерживаемые уровни проверяет сам Codex при запуске;
`--check-only` проверяет только построение команды.
Каталог установленного CLI: `codex debug models` (JSON).

Subagents остаются в `[agents]` и `.codex/agents/*.toml`:
explorer = `gpt-6-luna` / `medium`, tester и reviewer = `gpt-6.1-sol` / `medium`;
default subagent = `gpt-6.1-sol` / `low`; concurrency = 2 дочерних агента.
Launcher не пишет конфиги, не задаёт `agents.*`, не меняет permissions и
не создаёт собственный dispatcher. Routing и contracts остаются в `AGENTS.md`.

В Codex desktop выбирайте модель и reasoning в интерфейсе перед отправкой
задачи. Launcher открывает отдельную CLI-сессию и не меняет текущий чат desktop.
Запись `MAIN MODEL` в тексте задачи сама по себе не переключает runtime-модель.

Нативные поля MAIN и subagent defaults описаны в
[официальной справке](https://learn.chatgpt.com/docs/config-file/config-reference).
Сохранность файлов конфигурации и точность argv можно проверить локально;
фактическую модель каждого дочернего агента доказывает только runtime telemetry.
