"""Functional evaluators and a chained workflow (prototype).

Create in the LaunchDarkly AI Library (Library > Tools > Create, choose the type):
  - functional evaluator ``pii-regex``  (lower is better)
  - a workflow ``support-quality`` with steps: judge -> pii-regex -> judge (reads step 1) -> aggregate

Run: LD_API_TOKEN=... LD_SDK_KEY=... LD_PROJECT_KEY=... python main.py
"""

import asyncio
import re

from launchdarkly_ai_server.evaluations import EvaluatorContext, init_evaluations

PII = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|[\w.]+@[\w.]+\.\w+")


def pii_regex(ctx: EvaluatorContext) -> dict:
    hits = PII.findall(str(ctx.output))
    # lower_is_better in LaunchDarkly: 0.0 means no PII found.
    return {"score": 1.0 if hits else 0.0, "reasoning": f"{len(hits)} PII match(es)"}


async def main() -> None:
    evals = init_evaluations()
    workflow = await evals.workflows.get(
        "support-quality", implementations={"pii-regex": pii_regex}
    )
    handlers: list = []  # your provider handlers, as for any judge
    result = await workflow.run(
        input="Where is my refund?",
        output="It ships Monday.",
        handlers=handlers,
        context={"kind": "user", "key": "demo"},
    )
    print(result.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
