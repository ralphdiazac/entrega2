"""Estructura de salida del pipeline de extracción de entidades técnicas."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# valores de relleno con los que el LLM "cumple" min_length sin haber extraído nada
PLACEHOLDERS = {"unknown", "n/a", "na", "none", "null", "ninguna", "ninguno", "desconocido", "desconocida", "no especificado"}


class NivelCriticidad(str, Enum):
    baja = "baja"
    media = "media"
    alta = "alta"


class EntidadesTecnicas(BaseModel):
    tecnologias: list[str] = Field(
        min_length=1,
        description="Tecnologías, lenguajes, frameworks, servicios o herramientas mencionadas explícitamente en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="baja: informativo o sin impacto; media: degradación parcial; alta: caída, pérdida de datos o riesgo de seguridad."
    )
    resumen_tecnico: str = Field(
        min_length=10,
        max_length=500,
        description="Resumen técnico del texto en una o dos frases.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        vistos: dict[str, str] = {}
        for v in valores:
            limpio = v.strip()
            if limpio and limpio.strip("<>[]() ").lower() not in PLACEHOLDERS:
                vistos.setdefault(limpio.lower(), limpio)
        if not vistos:
            raise ValueError("tecnologias no puede quedar vacía ni contener valores de relleno")
        return list(vistos.values())
