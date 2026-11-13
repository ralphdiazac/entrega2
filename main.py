"""Mini-script de prueba: ejecuta un ejemplo contra el LLM real."""

import asyncio
import logging

from chain import process_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EJEMPLO = (
    "Nuestra API en FastAPI usa Redis como caché y PostgreSQL para persistencia; "
    "bajo carga, las consultas sin índice saturaron la base de datos y la API dejó de responder."
)

if __name__ == "__main__":
    print(asyncio.run(process_text(EJEMPLO)).model_dump_json(indent=2))
