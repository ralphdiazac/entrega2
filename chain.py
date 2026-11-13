"""Cadena LCEL: prompt | model.with_structured_output(Schema), con reintentos, logs y process_text() asíncrona."""

import logging
import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import EntidadesTecnicas

logger = logging.getLogger("chain")

MAX_ATTEMPTS = 3
MAX_TOKENS = 1024


# --------------------------------------------------------------------------- cliente (lógica del Módulo 1)
class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Modelo por defecto solo para Anthropic; con OpenAI hay que indicarlo en LLM_MODEL.
MODELO_POR_DEFECTO = {Provider.ANTHROPIC: "claude-opus-5-5"}


def crear_modelo() -> BaseChatModel:
    """Elige ChatAnthropic o ChatOpenAI según LLM_PROVIDER / LLM_MODEL. Las API keys las leen los SDKs."""
    load_dotenv(Path(__file__).with_name(".env"))  # ruta explícita: no depende del cwd
    valor = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    try:
        proveedor = Provider(valor)
    except ValueError:
        valid = ", ".join(p.value for p in Provider)
        raise ValueError(f"Proveedor desconocido {valor!r}. Válidos: {valid}.") from None

    modelo = os.environ.get("LLM_MODEL") or MODELO_POR_DEFECTO.get(proveedor)
    if not modelo:
        raise ValueError(f"Define LLM_MODEL para el proveedor {proveedor.value!r}.")

    # Sin temperature/top_p: el SDK actual de Anthropic los elimina (igual que en el Módulo 1).
    if proveedor is Provider.ANTHROPIC:
        return ChatAnthropic(model=modelo, max_tokens=MAX_TOKENS)
    return ChatOpenAI(model=modelo, max_tokens=MAX_TOKENS)


model = crear_modelo()

# --------------------------------------------------------------------------- prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un analista técnico. Extrae información técnica de descripciones de arquitectura, "
            "incidentes y logs de error.\n"
            "Reglas:\n"
            "- tecnologias: lista NO vacía, solo tecnologías mencionadas explícitamente; no inventes ninguna.\n"
            "- nivel_de_criticidad: exactamente uno de: baja, media, alta.\n"
            "- resumen_tecnico: una o dos frases, máximo 500 caracteres.\n"
            "Si el texto es ambiguo, elige la interpretación más conservadora y refleja la "
            "incertidumbre en el resumen; nunca inventes datos que no estén en el texto.",
        ),
        ("human", "Texto a analizar:\n\n{texto}"),
    ]
)


# --------------------------------------------------------------------------- cadena
def con_reintentos(runnable: Runnable) -> Runnable:
    """Reintenta ante JSON mal formado o incompleto (parser/validación Pydantic)."""
    return runnable.with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=MAX_ATTEMPTS,
    )


chain = con_reintentos(prompt | model.with_structured_output(EntidadesTecnicas))


# --------------------------------------------------------------------------- logs
class LogHandler(BaseCallbackHandler):
    """Registra cada intento al LLM y cada fallo de validación que provoca un reintento."""

    def __init__(self) -> None:
        self.intento = 0
        self._errores_vistos: list[BaseException] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.intento += 1
        logger.info("Intento %d/%d: llamando al modelo", self.intento, MAX_ATTEMPTS)

    def on_chain_error(self, error, **kwargs):
        # el mismo error se propaga por varias runs anidadas: solo se registra una vez
        if any(error is visto for visto in self._errores_vistos):
            return
        self._errores_vistos.append(error)
        if self.intento < MAX_ATTEMPTS:
            logger.warning("Intento %d/%d falló (%s): %s. Reintentando…", self.intento, MAX_ATTEMPTS, type(error).__name__, error)
        else:
            logger.error("Intento %d/%d falló (%s): %s. Sin más reintentos.", self.intento, MAX_ATTEMPTS, type(error).__name__, error)


async def process_text(text: str) -> EntidadesTecnicas:
    logger.info("Procesando texto (%d caracteres)", len(text))
    resultado = await chain.ainvoke({"texto": text}, config={"callbacks": [LogHandler()]})
    logger.info("Validación OK: %d tecnologías, criticidad=%s", len(resultado.tecnologias), resultado.nivel_de_criticidad.value)
    return resultado
