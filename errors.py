"""Errores controlados del pipeline: se muestran como un mensaje claro, sin traceback."""


class PipelineError(Exception):
    """Base de los errores controlados."""


class ConfigError(PipelineError):
    """Configuración inválida o incompleta: proveedor, modelo o API key."""


class AuthError(PipelineError):
    """El proveedor rechazó la API key (inválida, revocada o sin permisos)."""
