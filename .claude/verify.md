# Verify-Kontrakt — area-occupancy (Fork)

Fork von `Hankanman/Area-Occupancy-Detection`, Home-Assistant-Custom-Component.
Upstream-Remote: `upstream`. Der Fork-Stand liegt auf `main`, Arbeit auf `fix/…`/`feat/…`-Branches.
Bau-, Release- und Ausrollweg im Wiki `area-occupancy-fork`.

## Test und Lint

Lokal seit 2026-09-25 lauffähig: Python 3.14 kommt von uv (`uv python install 3.14`), das
System-Python bleibt 3.12. Einmalig `uv venv --python 3.14 && uv sync --extra dev --extra test`
(nicht `scripts/bootstrap`, das zieht zusätzlich apt-Pakete, Simulator, Docs und pre-commit).

```
uv run pytest tests/test_override.py -q     # Fork-Tests
uv run pytest -q                            # alles
uv run ruff format --check . && uv run ruff check .
```

## Actions im Fork

Push und Release lösen im Fork **nichts** aus. `test.yml` und `lint.yml` haben deshalb
`workflow_dispatch` und werden von Hand angestoßen, auch auf einem Branch:

```
gh workflow run test.yml --repo martinhoess/Area-Occupancy-Detection --ref <branch>
gh workflow run lint.yml --repo martinhoess/Area-Occupancy-Detection --ref <branch>
```

## Version

Steht an drei Stellen und muss gleich sein: `pyproject.toml`, `const.py` (`DEVICE_SW_VERSION`),
`manifest.json`; `uv.lock` zieht per `uv lock` nach. Schema `2026.<monat>.1xx`, PEP 440, kein
Suffix. Der Release-Tag muss exakt der Manifest-Version entsprechen.

## Ausrollen

Release mit Asset `area_occupancy.zip` (Inhalt von `custom_components/area_occupancy` ohne
`__pycache__`, `hacs.json` verlangt genau diesen Namen; `git archive` aus dem Tag liefert das
sauber). Installiert wird über HACS als Custom Repository
`martinhoess/Area-Occupancy-Detection`, danach HA-Neustart — nur nach Rückfrage. Danach die
Manifest-Version der installierten Integration gegenprüfen; Zugangswege stehen im Wiki, nicht hier.
