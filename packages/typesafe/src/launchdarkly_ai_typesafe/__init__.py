"""LaunchDarkly AI SDK integration for TypeSafe Jev."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
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

register_ai_sdk_package(PACKAGE_NAME, __version__)
