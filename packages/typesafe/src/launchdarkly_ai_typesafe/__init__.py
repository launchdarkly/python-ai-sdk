"""LaunchDarkly AI SDK integration for TypeSafe Jev."""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import create_typesafe_handler
from .questions import (
    TypesafeQuestionError,
    build_typesafe_state,
    extract_typesafe_questions,
    reason_from_answer,
    score_from_answer,
)

__all__ = [
    "TypesafeQuestionError",
    "build_typesafe_state",
    "create_typesafe_handler",
    "extract_typesafe_questions",
    "reason_from_answer",
    "score_from_answer",
]

register_ai_sdk_package("launchdarkly-ai-typesafe", __version__)
