"""Mini-script de prueba: ejecuta un ejemplo contra el LLM real."""

import asyncio
import logging

from errors import PipelineError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EJEMPLO = (
    "Nuestra API en FastAPI usa Redis como caché y PostgreSQL para persistencia; "
    "bajo carga, las consultas sin índice saturaron la base de datos y la API dejó de responder."
)


def main() -> None:
    from chain import process_text  # se importa aquí: crear el cliente puede lanzar ConfigError

    print(asyncio.run(process_text(EJEMPLO)).model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except PipelineError as e:  # error controlado: mensaje claro y código de salida 1, sin traceback
        raise SystemExit(f"Error ({type(e).__name__}): {e}") from None
