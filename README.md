# Analytics Agent Evaluation & Prompt Optimization

A runnable Python toolkit for evaluating analytics agents and improving their
system prompts with execution-based, rule-based, and LLM-assisted graders.

It includes:

- Deterministic train/heldout dataset generation
- Text-to-SQL execution grading against a real database snapshot
- Rule-based chart selection grading
- Grounded insight grading with an independent LLM judge
- TextGrad-style prompt rewriting
- Fixed-budget optimization with keep/discard decisions and JSONL trial logs
- Offline mock engines for testing the complete workflow without API credentials

## Supported skills

| Skill | Grader | Pass condition |
|---|---|---|
| `text2sql` | SQL execution match | Candidate and reference SQL return equivalent results |
| `chart_selection` | Rule based | Chart type matches the shape of the data |
| `insight_generation` | LLM judge | Response is grounded in the supplied data and relevant |

Each skill owns its prompt, datasets, and grader while sharing the same optimization
loop.

## How it works

```text
Database + seed questions or directions
                  |
                  v
       Verified train/heldout tasks
                  |
                  v
Agent prompt -> agent response -> skill grader
                  |
                  v
        Failure feedback -> prompt rewrite
                  |
                  v
       Keep only heldout improvements
```

The agent under evaluation and the optimizer/judge are separate components. The
optimizer never replaces the agent during scoring.

## Installation

```bash
git clone <repository-url>
cd analytics_agent_evals_extended
python -m pip install -r requirements.txt
```

## Quick start

The default workflow uses a sample SQLite database and deterministic mock engines:

```bash
python bootstrap/generate_dataset.py
python optimize/autoresearch_loop.py
```

Results are written to:

- `prompts/*.txt` — best prompt retained for each skill
- `logs/trials_<skill>.jsonl` — score and decision for every trial
- `data/tasks_*.json` — generated train and heldout datasets

## Connect your agent

Edit `get_agent_engines()` in
`optimize/autoresearch_loop.py`.

One method for all skills:

```python
from engine.agent_adapter import ObjectAgentAdapter
from my_project.agent import AnalyticsAgent

agent = AnalyticsAgent(...)
engine = ObjectAgentAdapter(agent, method="run")
return {name: engine for name in skill_names}
```

Separate methods:

```python
return {
    "text2sql": ObjectAgentAdapter(agent, method="write_sql"),
    "chart_selection": ObjectAgentAdapter(agent, method="pick_chart"),
    "insight_generation": ObjectAgentAdapter(agent, method="summarize"),
}
```

An HTTP agent can use `EndpointAgentAdapter`:

```python
return {
    name: EndpointAgentAdapter(
        url="https://service.example/v1/agent",
        headers={"Authorization": f"Bearer {os.environ['AGENT_API_KEY']}"},
        response_parser=lambda body: body["result"]["output"],
    )
    for name in skill_names
}
```

Adapters only require a prompt input and final text output. The wrapped agent may
use tools, retries, or multiple internal steps.

## Configure the optimizer

Set an optimizer provider and model:

```bash
export OPTIMIZER_PROVIDER=anthropic
export OPTIMIZER_MODEL=<model-name>
export ANTHROPIC_API_KEY=<api-key>
```

Use `OPTIMIZER_PROVIDER=openai` and `OPENAI_API_KEY` for OpenAI. If no provider
credentials are configured, the optimization workflow uses the offline mock
optimizer.

The number of optimization attempts is controlled by `N_TRIALS` in
`optimize/autoresearch_loop.py`.

## Generate datasets

### Built-in seed dataset

```bash
python bootstrap/generate_dataset.py
```

This recreates the sample database, verifies all reference SQL, and writes
deterministic train/heldout splits.

### Generate from business directions

When no ground-truth questions exist, an LLM can propose candidate questions and
reference SQL from a SQLite snapshot and broad business directions:

```bash
export DATASET_PROVIDER=openai
export DATASET_MODEL=<model-name>
export OPENAI_API_KEY=<api-key>

python bootstrap/generate_dataset.py \
  --db-path db/readonly_snapshot.db \
  --schema-description-file db/business_definitions.md \
  --direction bootstrap/directions.example.csv \
  --values-per-direction 5 \
  --num-generated 30 \
  --skip-seeds
```

The generation pipeline:

1. Introspects the schema and expands database-backed placeholders.
2. Asks the LLM for question and reference-SQL candidates.
3. Executes each query through a read-only SQLite connection.
4. Rejects invalid, empty, oversized, duplicate, or semantically weak tasks.
5. Stores database-derived results and validation metadata.
6. Produces deterministic train/heldout splits.

The LLM proposes tasks; the database execution result establishes factual ground
truth.

### Direction CSV format

`--direction` accepts a literal string or a CSV path. CSV files require a
`direction` column:

```csv
direction
"Look at {{customers.customer_name}}, find their latest transaction, and recommend how we can grow business with this customer."
```

Placeholders use `{{table.column}}`. The generator samples distinct values from that
column and substitutes them into the direction. Multiple placeholders from the same
table are kept on the same source row.

`--values-per-direction` limits the number of instantiated values. `--direction`
is repeatable, so literal directions and multiple CSV files can be combined.

Advisory directions create two linked tasks:

- An execution-verified SQL task that retrieves the supporting evidence
- An insight task that grades the recommendation for grounding and relevance

This avoids pretending that an open-ended recommendation has one exact answer.

## Dataset outputs

| File | Purpose |
|---|---|
| `data/tasks_train.json` | Text-to-SQL optimization tasks |
| `data/tasks_heldout.json` | Text-to-SQL prompt-selection tasks |
| `data/tasks_charts_*.json` | Chart selection tasks |
| `data/tasks_insights_*.json` | Insight and advisory tasks |
| `data/dataset_manifest.json` | Snapshot path/hash and generation settings |

The optimizer verifies the recorded database hash before SQL grading. Set
`ANALYTICS_DB_PATH` only when an explicit snapshot override is required.

## Add a skill

1. Add the skill prompt under `prompts/`.
2. Implement a grader with the existing `grade_suite` result shape.
3. Register its prompt, datasets, and grader in `optimize/skills.py`.

The optimization loop itself does not require skill-specific changes.

## Project layout

```text
bootstrap/   Dataset generation and direction templates
data/        Generated task suites and dataset manifest
db/          Sample SQLite schema and database
engine/      LLM engines and agent adapters
graders/     Execution, rule-based, and LLM graders
optimize/    Skill registry and prompt-optimization loop
prompts/     Current system prompts
logs/        Append-only optimization trial records
tests/       Dataset-generation tests
```

## Operational guidance

- Use a frozen, read-only SQLite snapshot for repeatable SQL grading.
- Add business definitions that cannot be inferred from schema names.
- Treat synthetic tasks as technically grounded, not proof of business importance.
- Review a sample of generated tasks and LLM-judge decisions with domain experts.
- Include unanswerable questions to test refusal and uncertainty behavior.
- Prevent state leakage when repeatedly evaluating stateful agents.
- Do not place sensitive columns in direction placeholders unless the configured LLM
  provider is approved to receive those values.

The repository's heldout split is used to select prompt improvements. For an unbiased
final benchmark, maintain a separate test set that is never used during optimization.
