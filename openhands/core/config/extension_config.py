"""Configuration for the extension system.

CCA F3: Configuration options for the extension system that provides
modular callbacks for agent behavior customization.
"""

from pydantic import BaseModel, ConfigDict, Field


class ExtensionConfig(BaseModel):
    """Configuration for the extension system (CCA F3).

    Controls which extensions are enabled and their behavior.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = Field(
        default=True,
        description='Whether the extension system is enabled',
    )

    active_extensions: list[str] = Field(
        default_factory=list,
        description='List of extension names to activate. Empty list means all registered extensions.',
    )

    disabled_extensions: list[str] = Field(
        default_factory=list,
        description='List of extension names to explicitly disable',
    )

    allow_custom_extensions: bool = Field(
        default=True,
        description='Whether to allow loading custom extensions from user directories',
    )

    custom_extensions_dir: str | None = Field(
        default=None,
        description='Directory to load custom extensions from. Defaults to ~/.openhands/extensions/',
    )

    log_extension_calls: bool = Field(
        default=False,
        description='Whether to log extension callback invocations for debugging',
    )

    fail_on_extension_error: bool = Field(
        default=False,
        description='Whether to fail the agent if an extension raises an exception',
    )

    max_extension_time_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description='Maximum time in milliseconds for extension callbacks before timeout',
    )
