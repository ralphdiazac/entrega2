# Entrega 2 — Pipeline de extracción de entidades técnicas

Recibe un texto sin procesar (descripción de arquitectura, log de error, reporte de incidente) y devuelve un objeto **validado con Pydantic**: tecnologías mencionadas, nivel de criticidad y un resumen técnico.

La cadena es LCEL, con salida estructurada y reintentos automáticos:

```python
chain = (prompt | model.with_structured_output(EntidadesTecnicas)).with_retry(...)
```

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelo Pydantic `EntidadesTecnicas` y el enum `NivelCriticidad`. |
| `chain.py` | Cliente (`ChatAnthropic` / `ChatOpenAI`), `ChatPromptTemplate`, cadena LCEL, `.with_retry()`, logs y `process_text()` asíncrona. |
| `main.py` | Mini-script que ejecuta un ejemplo contra el LLM real. |
| `stress_test.py` | Prueba de estrés: reintentos con un LLM simulado y textos ambiguos contra el LLM real. |

## Requisitos

- Python **3.12**
- Una API key de Anthropic y/o de OpenAI

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Variables de entorno

Copia `.env.example` a `.env` (está en `.gitignore`; nunca lo subas a git) y completa los valores:

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `anthropic`) | `openai` o `anthropic`. Elige qué cliente se usa. |
| `LLM_MODEL` | Con `openai`; opcional con `anthropic` | Modelo a usar. Por defecto en Anthropic: `claude-opus-5-5`. |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | La lee el SDK. |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | La lee el SDK. |

La configuración del cliente reutiliza la lógica del Módulo 1 (`Entrega 1`): proveedor por `LLM_PROVIDER`, modelo por `LLM_MODEL`, modelo por defecto solo para Anthropic y credenciales leídas por los SDK.

## Uso

Ejecutar el ejemplo:

```bash
python main.py
```

Desde tu propio código:

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("Nuestra API en FastAPI usa Redis como caché y PostgreSQL..."))
print(resultado.model_dump_json(indent=2))
```

### Ejemplo de salida

Entrada:

> Nuestra API en FastAPI usa Redis como caché y PostgreSQL para persistencia; bajo carga, las consultas sin índice saturaron la base de datos y la API dejó de responder.

Salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API en FastAPI con Redis como caché y PostgreSQL para persistencia. Bajo carga, consultas sin índice saturaron la base de datos y la API dejó de responder, lo que indica una caída del servicio."
}
```

## El contrato (`schemas.py`)

| Campo | Tipo | Restricciones |
|---|---|---|
| `tecnologias` | `list[str]` | Al menos un elemento. Se limpian espacios y duplicados; se rechazan valores de relleno (`<UNKNOWN>`, `N/A`…). |
| `nivel_de_criticidad` | enum `baja \| media \| alta` | `baja`: informativo; `media`: degradación parcial; `alta`: caída, pérdida de datos o riesgo de seguridad. |
| `resumen_tecnico` | `str` | Entre 10 y 500 caracteres. |

## Resiliencia

- `.with_retry()` reintenta hasta **3 veces** cuando el LLM devuelve un JSON mal formado (`OutputParserException`) o que no cumple el esquema (`ValidationError`). Cada intento vuelve a llamar al modelo.
- Si se agotan los intentos, se propaga la última excepción: el pipeline nunca devuelve un objeto incompleto.
- `process_text()` registra con `logging` cada llamada al modelo, cada fallo de validación y si se reintenta o no:

```
INFO    Intento 1/3: llamando al modelo
WARNING Intento 1/3 falló (ValidationError): ... Reintentando…
INFO    Intento 2/3: llamando al modelo
ERROR   Intento 3/3 falló (ValidationError): ... Sin más reintentos.
```

Una respuesta cortada por falta de tokens llega como JSON incompleto, falla la validación y entra en este mismo ciclo de reintentos.

## Prueba de estrés

```bash
python stress_test.py          # parte simulada + 5 textos ambiguos contra el LLM real
python stress_test.py offline  # solo la parte simulada (sin API ni costo)
```

Con el LLM real, un texto sin ninguna tecnología ("Hoy todo anda raro, no sé qué pasó…") agota los 3 intentos y lanza `ValidationError`, en lugar de aceptar datos inventados.
