"""Bootstrap verified analytics eval tasks from seeds and/or broad directions.

The LLM proposes questions and reference SQL; it never supplies the final ground
truth.  Ground truth is produced by executing read-only SQL against the supplied
database snapshot, with a second LLM pass optionally checking semantic alignment.

Seed-only demo:  python bootstrap/generate_dataset.py
Synthetic tasks: python bootstrap/generate_dataset.py --direction "retention" \
                 --direction "monthly revenue trends" --num-generated 20
Own SQLite DB:   python bootstrap/generate_dataset.py --db-path snapshot.db \
                 --schema-description-file schema.md --direction "inventory risk" \
                 --num-generated 20 --skip-seeds
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.setup_db import build_db, SCHEMA_DESCRIPTION, DB_PATH  # noqa: E402
from engine.llm_engine import ClaudeEngine, OpenAIEngine  # noqa: E402
import sqlite3  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_MAX_ROWS = 200
DEFAULT_STEP_LIMIT = 1_000_000
PLACEHOLDER_RE = re.compile(
    r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*}}"
)

# Hand-written seed: real, unambiguous business questions with gold SQL.
# This is the part worth spending human time on -- 15-20 of these,
# written by someone who knows what the business actually asks,
# is worth more than 500 synthetic ones. :)
SEED_TASKS = [
    {
        "id": "seed_001",
        "question": "What is the total revenue by region?",
        "gold_sql": (
            "SELECT r.region_name, SUM(o.quantity * p.unit_price) AS revenue "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN products p ON o.product_id = p.product_id "
            "JOIN regions r ON c.region_id = r.region_id "
            "GROUP BY r.region_name;"
        ),
    },
    {
        "id": "seed_002",
        "question": "Which product had the highest total units sold?",
        "gold_sql": (
            "SELECT p.product_name, SUM(o.quantity) AS units "
            "FROM orders o JOIN products p ON o.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY units DESC LIMIT 1;"
        ),
    },
    {
        "id": "seed_003",
        "question": "List each customer's total spend, highest first.",
        "gold_sql": (
            "SELECT c.customer_name, SUM(o.quantity * p.unit_price) AS spend "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN products p ON o.product_id = p.product_id "
            "GROUP BY c.customer_name ORDER BY spend DESC;"
        ),
    },
    {
        "id": "seed_004",
        "question": "What is the total revenue from orders placed in May 2026?",
        "gold_sql": (
            "SELECT SUM(o.quantity * p.unit_price) AS revenue "
            "FROM orders o JOIN products p ON o.product_id = p.product_id "
            "WHERE o.order_date >= '2026-05-01' AND o.order_date < '2026-06-01';"
        ),
    },
    {
        "id": "seed_005",
        "question": "How many distinct customers ordered a 'Gadgets' category product?",
        "gold_sql": (
            "SELECT COUNT(DISTINCT o.customer_id) "
            "FROM orders o JOIN products p ON o.product_id = p.product_id "
            "WHERE p.category = 'Gadgets';"
        ),
    },
    {
        "id": "seed_006",
        "question": "What is the average order quantity per region?",
        "gold_sql": (
            "SELECT r.region_name, AVG(o.quantity) AS avg_qty "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN regions r ON c.region_id = r.region_id "
            "GROUP BY r.region_name;"
        ),
    },
]


CHART_SEED_TASKS = [
    {"id": "chart_001", "question": "Show revenue trend over the last 6 months.",
     "data": "[{'month':'Jan','revenue':1000},{'month':'Feb','revenue':1200},{'month':'Mar','revenue':1100}]",
     "data_shape": "time_series"},
    {"id": "chart_002", "question": "Show total revenue by region.",
     "data": "[{'region':'NA','revenue':5000},{'region':'EMEA','revenue':3000},{'region':'APAC','revenue':2000}]",
     "data_shape": "categorical_comparison"},
    {"id": "chart_003", "question": "What percentage of total revenue does each product category represent?",
     "data": "[{'category':'Widgets','pct':60},{'category':'Gadgets','pct':40}]",
     "data_shape": "part_of_whole"},
    {"id": "chart_004", "question": "Show monthly order counts over the year.",
     "data": "[{'month':'Jan','orders':12},{'month':'Feb','orders':15},{'month':'Mar','orders':9}]",
     "data_shape": "time_series"},
    {"id": "chart_005", "question": "Show customer count by region.",
     "data": "[{'region':'NA','customers':2},{'region':'EMEA','customers':2},{'region':'APAC','customers':1}]",
     "data_shape": "categorical_comparison"},
    {"id": "chart_006", "question": "Show the breakdown of orders by product category as a share of total.",
     "data": "[{'category':'Widgets','pct':70},{'category':'Gadgets','pct':30}]",
     "data_shape": "part_of_whole"},
]

INSIGHT_SEED_TASKS = [
    {"id": "insight_001", "question": "What does the regional revenue data tell us?",
     "data": "NA: $5000, EMEA: $3000, APAC: $2000"},
    {"id": "insight_002", "question": "What does the monthly revenue trend show?",
     "data": "Jan: $1000, Feb: $1200, Mar: $1100, Apr: $1400"},
    {"id": "insight_003", "question": "What does the product category breakdown tell us?",
     "data": "Widgets: 60%, Gadgets: 40%"},
    {"id": "insight_004", "question": "What can we conclude from customer spend by region?",
     "data": "NA: $2500, EMEA: $1800, APAC: $900"},
]


def verify_and_split_generic(tasks, required_fields, train_frac=0.6, seed=42):
    """For skills without an executable gold answer (chart, insight): just
    check the required fields are present and non-empty before splitting."""
    verified = []
    for t in tasks:
        if all(t.get(f) for f in required_fields):
            verified.append(t)
        else:
            print(f"[SKIP] {t['id']} missing required field(s) {required_fields}.")
    unique = {}
    for task in verified:
        key = _normalise_question(task["question"])
        if key and key not in unique:
            unique[key] = task
        elif key:
            print(f"[SKIP] {task['id']} duplicates question in {unique[key]['id']}.")
    verified = list(unique.values())
    random.Random(seed).shuffle(verified)
    n_train = max(1, int(len(verified) * train_frac))
    return verified[:n_train], verified[n_train:]


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _normalise_question(question):
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _is_read_only_sql(sql):
    """Conservative gate before SQLite's own read-only connection is used."""
    cleaned = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).strip()
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        return False
    # sqlite3.execute already rejects multiple statements. This catches writable
    # CTEs and dangerous pragmas early and gives a clearer validation reason.
    blocked = r"\b(insert|update|delete|replace|drop|alter|create|attach|detach|vacuum|pragma)\b"
    return re.search(blocked, cleaned, re.I) is None


def execute_gold_sql(db_path, sql, max_rows=DEFAULT_MAX_ROWS,
                     step_limit=DEFAULT_STEP_LIMIT):
    if not _is_read_only_sql(sql):
        raise ValueError("only a single read-only SELECT/WITH query is allowed")
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    steps = 0

    def stop_long_query():
        nonlocal steps
        steps += 1000
        return 1 if steps > step_limit else 0

    conn.set_progress_handler(stop_long_query, 1000)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"result exceeds the {max_rows}-row safety limit")
        return columns, rows
    finally:
        conn.close()


def verify_tasks(tasks, db_path, max_rows=DEFAULT_MAX_ROWS):
    verified = []
    for t in tasks:
        try:
            columns, rows = execute_gold_sql(db_path, t["gold_sql"], max_rows=max_rows)
        except Exception as e:
            print(f"[SKIP] {t['id']} gold SQL failed to execute: {e}")
            continue
        if not rows:
            print(f"[SKIP] {t['id']} gold SQL returned no rows (likely wrong).")
            continue
        if all(all(value is None for value in row) for row in rows):
            print(f"[SKIP] {t['id']} gold SQL returned only NULL values.")
            continue
        t = dict(t)
        t.setdefault("validation", {})
        t["validation"].update({
            "sql_executed": True,
            "row_count": len(rows),
            "columns": columns,
            "result_preview": [[_json_safe(v) for v in row] for row in rows[:10]],
            "result": [[_json_safe(v) for v in row] for row in rows],
        })
        verified.append(t)
    return verified


def split_tasks(tasks, train_frac=0.6, seed=42):
    """Deduplicate before a deterministic split to prevent train/heldout leakage."""
    unique = {}
    for task in tasks:
        key = _normalise_question(task["question"])
        if key and key not in unique:
            unique[key] = task
        elif key:
            print(f"[SKIP] {task['id']} duplicates question in {unique[key]['id']}.")
    verified = list(unique.values())

    random.Random(seed).shuffle(verified)
    n_train = max(1, int(len(verified) * train_frac))
    return verified[:n_train], verified[n_train:]


def verify_and_split(tasks, db_path, train_frac=0.6, seed=42,
                     max_rows=DEFAULT_MAX_ROWS):
    return split_tasks(verify_tasks(tasks, db_path, max_rows), train_frac, seed)


def introspect_sqlite_schema(db_path):
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return "\n\n".join(sql or f"-- {name} (definition unavailable)" for name, sql in tables)
    finally:
        conn.close()


def load_direction_inputs(values):
    """Load literal CLI directions or CSVs with a required `direction` column."""
    templates = []
    for value in values:
        if value.lower().endswith(".csv"):
            path = os.path.abspath(value)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"direction CSV does not exist: {path}")
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "direction" not in reader.fieldnames:
                    raise ValueError(f"direction CSV must contain a 'direction' column: {path}")
                for row_number, row in enumerate(reader, 2):
                    direction = (row.get("direction") or "").strip()
                    if direction:
                        templates.append({
                            "template": direction,
                            "source": path,
                            "csv_row": row_number,
                        })
        elif value.strip():
            templates.append({"template": value.strip(), "source": "cli"})
    if not templates:
        raise ValueError("no non-empty directions were supplied")
    return templates


def _quote_identifier(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _table_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return {row[1] for row in rows}


def expand_direction_templates(templates, db_path, values_per_direction=3, seed=42):
    """Bind {{table.column}} placeholders to real, relationally consistent DB values."""
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    rng = random.Random(seed)
    expanded = []
    try:
        known_tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        for template_number, spec in enumerate(templates, 1):
            placeholders = list(dict.fromkeys(PLACEHOLDER_RE.findall(spec["template"])))
            if not placeholders:
                expanded.append({
                    "id": f"direction_{template_number}_1",
                    **spec,
                    "direction": spec["template"],
                    "bindings": {},
                })
                continue

            by_table = {}
            for table, column in placeholders:
                if table not in known_tables:
                    raise ValueError(
                        f"unknown placeholder table {table!r} in direction: {spec['template']}"
                    )
                if column not in _table_columns(conn, table):
                    raise ValueError(
                        f"unknown placeholder column {table}.{column} in direction: "
                        f"{spec['template']}"
                    )
                by_table.setdefault(table, []).append(column)

            binding_groups = []
            for table, columns in by_table.items():
                select_list = ", ".join(_quote_identifier(c) for c in columns)
                non_null = " AND ".join(f"{_quote_identifier(c)} IS NOT NULL" for c in columns)
                sql = (
                    f"SELECT DISTINCT {select_list} FROM {_quote_identifier(table)} "
                    f"WHERE {non_null} LIMIT 1000"
                )
                rows = conn.execute(sql).fetchall()
                if not rows:
                    raise ValueError(f"placeholder source {table} has no usable values")
                bindings = [
                    {f"{table}.{column}": _json_safe(value)
                     for column, value in zip(columns, row)}
                    for row in rows
                ]
                rng.shuffle(bindings)
                binding_groups.append(bindings[:values_per_direction])

            combinations = itertools.islice(
                itertools.product(*binding_groups), values_per_direction
            )
            for expansion_number, groups in enumerate(combinations, 1):
                bindings = {key: value for group in groups for key, value in group.items()}
                direction = spec["template"]
                for key, value in bindings.items():
                    pattern = r"{{\s*" + re.escape(key) + r"\s*}}"
                    direction = re.sub(pattern, str(value), direction)
                expanded.append({
                    "id": f"direction_{template_number}_{expansion_number}",
                    **spec,
                    "direction": direction,
                    "bindings": bindings,
                })
    finally:
        conn.close()
    return expanded


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON array")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, list):
        raise ValueError("LLM response must be a JSON array")
    return value


def make_generation_engine(provider=None, model=None):
    provider = (provider or os.environ.get("DATASET_PROVIDER") or
                os.environ.get("OPTIMIZER_PROVIDER", "")).lower()
    model = model or os.environ.get("DATASET_MODEL") or os.environ.get("OPTIMIZER_MODEL")
    if not model:
        raise ValueError("set --model (or DATASET_MODEL) to an available model name")
    if provider == "openai":
        return OpenAIEngine(model)
    if provider == "anthropic":
        return ClaudeEngine(model)
    raise ValueError(
        "LLM generation needs --provider openai|anthropic (or DATASET_PROVIDER) "
        "and the provider API key"
    )


def generate_candidates(engine, schema, directions, count, batch_size=6):
    candidates = []
    while len(candidates) < count:
        requested = min(batch_size, count - len(candidates))
        prompt = f"""Create {requested} diverse text-to-SQL evaluation tasks.

Database schema and business definitions:
{schema}

Instantiated business directions:
{json.dumps(directions, indent=2)}

Return ONLY a JSON array. Each object must have exactly:
- question: a self-contained, objective data-retrieval question
- gold_sql: one SQLite SELECT/WITH query answering it
- direction_id: the id of the supplied direction it implements
- insight_question: the original decision/advisory question, rewritten to be
  self-contained; null if the direction has no interpretive/advisory component
- difficulty: easy, medium, or hard
- rationale: one sentence explaining why the SQL answers the question

Use only supplied schema facts. Prefer useful variation across filters, joins,
aggregation, time analysis, ranking, comparisons, and zero/absence cases where
supported. If a direction asks for advice (for example, how to grow a customer),
make question request the concrete database evidence needed for that advice and put
the advisory request in insight_question. Do not claim that advice has a single SQL
ground truth. Avoid forecasts, undefined business terms, nondeterministic relative
dates, SELECT *, and unavailable data. Preserve all instantiated entity values
exactly. Every SQL question must be answerable from this exact database snapshot.
Do not repeat any of these questions from earlier batches:
{json.dumps([c['question'] for c in candidates])}"""
        raw = engine.generate(
            prompt,
            system="You design rigorous analytics evaluations and emit strict JSON only.",
        )
        batch = _extract_json_array(raw)
        if not batch:
            raise ValueError("LLM returned an empty candidate batch")
        before = len(candidates)
        for i, item in enumerate(batch, 1):
            if len(candidates) >= count:
                break
            if not isinstance(item, dict) or not item.get("question") or not item.get("gold_sql"):
                print(f"[SKIP] generated item {i} is missing question or gold_sql.")
                continue
            digest = hashlib.sha256(
                f"{item['question']}\n{item['gold_sql']}".encode("utf-8")
            ).hexdigest()[:10]
            direction_id = item.get("direction_id")
            direction_spec = next(
                (direction for direction in directions if direction["id"] == direction_id),
                None,
            )
            if direction_spec is None:
                print(f"[SKIP] generated item {i} has unknown direction_id {direction_id!r}.")
                continue
            candidates.append({
                "id": f"synthetic_{digest}",
                "question": str(item["question"]).strip(),
                "gold_sql": str(item["gold_sql"]).strip(),
                "metadata": {
                    "source": "llm_generated",
                    "direction_id": direction_id,
                    "direction": direction_spec["direction"],
                    "direction_template": direction_spec["template"],
                    "placeholder_bindings": direction_spec["bindings"],
                    "insight_question": item.get("insight_question"),
                    "difficulty": item.get("difficulty"),
                    "generator_rationale": item.get("rationale"),
                },
            })
        if len(candidates) == before:
            raise ValueError("LLM batch contained no usable candidates")
    return candidates


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def critic_filter(engine, tasks, schema):
    """Use an independent prompt/pass for semantic checks, never for execution truth."""
    accepted = []
    for task in tasks:
        validation = task.get("validation", {})
        prompt = f"""Audit this proposed text-to-SQL evaluation task.
Schema:
{schema}

Question: {task['question']}
Paired advisory/insight question: {task.get('metadata', {}).get('insight_question')}
SQL: {task['gold_sql']}
Executed columns: {json.dumps(validation.get('columns'))}
Executed result preview: {json.dumps(validation.get('result_preview'))}

Return ONLY JSON: {{"accept": true|false, "reason": "...", "review_priority": "low|medium|high"}}.
Accept only if the SQL question is unambiguous, all business terms are defined by
the schema/context, and the SQL precisely answers it. If an advisory question is
present, the executed fields must provide useful evidence for a grounded response;
the recommendation itself does not need one unique answer. Do not invent facts."""
        try:
            raw = engine.generate(prompt, system="You are a strict dataset auditor.")
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
            verdict = json.loads(raw)
        except Exception as exc:
            print(f"[SKIP] {task['id']} critic response was invalid: {exc}")
            continue
        task["validation"]["critic"] = verdict
        if verdict.get("accept") is True:
            accepted.append(task)
        else:
            print(f"[SKIP] {task['id']} rejected by critic: {verdict.get('reason', 'no reason')}")
    return accepted


def make_paired_insight_tasks(text2sql_tasks):
    """Turn advisory directions into grounded insight tasks using executed SQL data."""
    insight_tasks = []
    for task in text2sql_tasks:
        metadata = task.get("metadata", {})
        question = metadata.get("insight_question")
        validation = task.get("validation", {})
        if not question:
            continue
        data = {
            "columns": validation.get("columns", []),
            "rows": validation.get("result", []),
        }
        insight_tasks.append({
            "id": f"insight_{task['id']}",
            "question": question,
            "data": json.dumps(data, ensure_ascii=False),
            "metadata": {
                "source": "llm_generated_from_verified_sql",
                "source_task_id": task["id"],
                "direction": metadata.get("direction"),
                "direction_template": metadata.get("direction_template"),
                "placeholder_bindings": metadata.get("placeholder_bindings", {}),
            },
        })
    return insight_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build verified analytics eval tasks from seeds and broad directions."
    )
    parser.add_argument("--db-path", help="Existing SQLite snapshot; never modified.")
    parser.add_argument(
        "--schema-description-file",
        help="Optional business definitions/schema notes appended to DB introspection.",
    )
    parser.add_argument(
        "--direction", action="append", default=[],
        help=(
            "Literal direction or CSV path with a 'direction' column; repeatable. "
            "Templates may use {{table.column}} placeholders."
        ),
    )
    parser.add_argument(
        "--values-per-direction", type=int, default=3,
        help="Maximum DB-value instantiations for each placeholder direction.",
    )
    parser.add_argument("--num-generated", type=int, default=0)
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--model", help="Dataset-generation model name.")
    parser.add_argument("--skip-seeds", action="store_true")
    parser.add_argument(
        "--no-critic", action="store_true",
        help="Skip the semantic LLM audit; SQL execution checks still run.",
    )
    parser.add_argument("--max-result-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_generated < 0:
        parser.error("--num-generated must be non-negative")
    if args.num_generated and not args.direction:
        parser.error("--num-generated requires at least one --direction")
    if args.direction and not args.num_generated:
        parser.error("--direction requires a positive --num-generated")
    if not 0 < args.train_frac < 1:
        parser.error("--train-frac must be between 0 and 1")
    if args.max_result_rows < 1:
        parser.error("--max-result-rows must be positive")
    if args.values_per_direction < 1:
        parser.error("--values-per-direction must be positive")
    return args


def _write_json(name, value):
    with open(os.path.join(DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    if args.db_path:
        db_path = os.path.abspath(args.db_path)
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"database snapshot does not exist: {db_path}")
        schema = introspect_sqlite_schema(db_path)
    else:
        db_path = build_db(overwrite=True)
        schema = SCHEMA_DESCRIPTION + "\n\nDDL:\n" + introspect_sqlite_schema(db_path)

    if args.schema_description_file:
        with open(args.schema_description_file, encoding="utf-8") as f:
            schema += "\n\nBusiness definitions:\n" + f.read().strip()

    text2sql_tasks = [] if args.skip_seeds else list(SEED_TASKS)
    generated_count = 0
    generated_insight_tasks = []
    resolved_directions = []
    if args.num_generated:
        direction_templates = load_direction_inputs(args.direction)
        resolved_directions = expand_direction_templates(
            direction_templates,
            db_path,
            values_per_direction=args.values_per_direction,
            seed=args.split_seed,
        )
        engine = make_generation_engine(args.provider, args.model)
        candidates = generate_candidates(
            engine, schema, resolved_directions, args.num_generated
        )
        executed = verify_tasks(candidates, db_path, args.max_result_rows)
        generated = executed if args.no_critic else critic_filter(engine, executed, schema)
        text2sql_tasks.extend(generated)
        generated_insight_tasks = make_paired_insight_tasks(generated)
        generated_count = len(generated)

    if not text2sql_tasks:
        raise ValueError("no tasks to write; enable seeds or request generated tasks")

    train, heldout = verify_and_split(
        text2sql_tasks, db_path, args.train_frac, args.split_seed, args.max_result_rows
    )
    _write_json("tasks_train.json", train)
    _write_json("tasks_heldout.json", heldout)
    print(f"[text2sql] verified {len(train) + len(heldout)}/{len(text2sql_tasks)} tasks "
          f"({generated_count} generated accepted). train={len(train)} heldout={len(heldout)}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_path": db_path,
        "db_sha256": file_sha256(db_path),
        "direction_inputs": args.direction,
        "directions": resolved_directions,
        "values_per_direction": args.values_per_direction,
        "generated_requested": args.num_generated,
        "generated_accepted": generated_count,
        "critic_enabled": bool(args.num_generated and not args.no_critic),
        "split_seed": args.split_seed,
        "train_fraction": args.train_frac,
    }
    _write_json("dataset_manifest.json", manifest)

    chart_train, chart_heldout = verify_and_split_generic(
        CHART_SEED_TASKS, required_fields=["question", "data", "data_shape"],
        train_frac=args.train_frac, seed=args.split_seed)
    _write_json("tasks_charts_train.json", chart_train)
    _write_json("tasks_charts_heldout.json", chart_heldout)
    print(f"[chart_selection] verified {len(chart_train) + len(chart_heldout)}/{len(CHART_SEED_TASKS)} seed tasks. "
          f"train={len(chart_train)} heldout={len(chart_heldout)}")

    insight_tasks = list(INSIGHT_SEED_TASKS) + generated_insight_tasks
    insight_train, insight_heldout = verify_and_split_generic(
        insight_tasks, required_fields=["question", "data"],
        train_frac=args.train_frac, seed=args.split_seed)
    _write_json("tasks_insights_train.json", insight_train)
    _write_json("tasks_insights_heldout.json", insight_heldout)
    print(f"[insight_generation] verified {len(insight_train) + len(insight_heldout)}/{len(insight_tasks)} "
          f"tasks ({len(generated_insight_tasks)} generated). "
          f"train={len(insight_train)} heldout={len(insight_heldout)}")


if __name__ == "__main__":
    main()
