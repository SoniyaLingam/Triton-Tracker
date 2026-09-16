# Day 8 Refactor Notes: Clean Code and SOLID

## Scope and preserved behavior

Day 8 refactors the existing Tracker codebase rather than replacing it. CSV iteration, Pydantic configuration, the `Step` abstraction, D5 decorators and resource management, D6 exceptions, and D7 JSONL logging remain available with their existing public entry points. The full test suite protects this behavior.

## Problems identified

1. `Pipeline.run()` coordinated steps, counted records, measured execution time, emitted structured logging events, and translated exceptions. Those are related but separate responsibilities.
2. `DataFilter` and `HighScoreFilter` duplicated identical numeric-threshold filtering code.
3. The D7 retry warning was unreachable because it appeared after the final-attempt `raise`.

## Before → after

| Before | After | Reason |
| --- | --- | --- |
| `Pipeline.run()` contained all structured logging details. | `Pipeline` orchestrates `Step` instances and delegates lifecycle notifications to `PipelineReporter`. | SRP and DIP: orchestration no longer owns logging formatting/details. |
| Pipeline logging was tied directly to the module logger. | `StructuredPipelineReporter` is the default implementation of the reporter contract. | Pipeline depends on an abstraction and remains usable with another reporter, including tests. |
| Two filters implemented the same row loop and numeric conversion. | `filter_rows_by_minimum()` contains the shared algorithm; both public filters retain their APIs. | DRY while avoiding a needless class hierarchy. |
| Retry warning followed `raise`. | The warning now runs only for attempts that will actually retry. | Correct control flow and useful D7 structured logging. |

## SOLID principles

### Single Responsibility Principle

`Pipeline` validates and sequences steps. `StructuredPipelineReporter` owns pipeline lifecycle reporting and record-count metadata. `filter_rows_by_minimum()` owns the reusable threshold-filter algorithm. The existing resource manager, configuration model, and logging configurator continue to own their original focused responsibilities.

### Open/Closed Principle

The pipeline still accepts any `Step`. Adding behavior means defining a new `Step` class; no `Pipeline` branch or concrete-step type check is needed:

```python
class AddSourceStep(Step):
    def process(self, rows):
        return [{**row, "source": "new"} for row in rows]

result = Pipeline([AddSourceStep()]).run(rows)
```

`tests/test_refactoring.py` verifies this acceptance criterion.

### Dependency Inversion Principle

The high-level `Pipeline` depends on `Step` and `PipelineReporter` contracts. `StructuredPipelineReporter` is composed as the default lower-level implementation, while a caller can inject another reporter without changing pipeline code. Pipeline orchestration remains independent of every concrete processing step.

## DRY, naming, and readability

`filter_rows_by_minimum(rows, field_name, minimum)` gives the shared logic a descriptive name and removes duplicated conversion/error-handling loops. `Pipeline._validate_steps()` and `Pipeline._run_step()` make the two orchestration responsibilities explicit. Reporter method names (`step_started`, `step_completed`, and `step_failed`) directly communicate lifecycle intent.

## Confirmation

The refactor preserves the D1-D7 APIs and behavior. Structured logging continues through `StructuredPipelineReporter`, including full tracebacks on failures; custom exceptions and their chains remain propagated by `Pipeline._run_step`; and the existing lazy iterators, configuration validation, retries, and runtime step swapping remain unchanged.
