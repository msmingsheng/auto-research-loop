import json
import os
import sqlite3
import tempfile
import unittest

from bootstrap.generate_dataset import (
    _extract_json_array,
    expand_direction_templates,
    execute_gold_sql,
    generate_candidates,
    load_direction_inputs,
    make_paired_insight_tasks,
    split_tasks,
    verify_tasks,
)


class GenerateDatasetTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("CREATE TABLE sales(id INTEGER, amount REAL); INSERT INTO sales VALUES (1, 10.5);")
        conn.close()

    def tearDown(self):
        os.remove(self.db_path)

    def test_executes_select_read_only(self):
        columns, rows = execute_gold_sql(self.db_path, "SELECT amount FROM sales")
        self.assertEqual(["amount"], columns)
        self.assertEqual([(10.5,)], rows)
        with self.assertRaisesRegex(ValueError, "read-only"):
            execute_gold_sql(self.db_path, "DELETE FROM sales")

    def test_verify_adds_database_ground_truth_metadata(self):
        tasks = [{"id": "x", "question": "Total?", "gold_sql": "SELECT SUM(amount) AS total FROM sales"}]
        verified = verify_tasks(tasks, self.db_path)
        self.assertEqual([[10.5]], verified[0]["validation"]["result_preview"])
        self.assertTrue(verified[0]["validation"]["sql_executed"])

    def test_split_removes_duplicate_questions(self):
        tasks = [
            {"id": "a", "question": "Total revenue?"},
            {"id": "b", "question": "Total revenue"},
        ]
        train, heldout = split_tasks(tasks)
        self.assertEqual(1, len(train) + len(heldout))

    def test_extracts_fenced_json(self):
        value = _extract_json_array("```json\n" + json.dumps([{"question": "Q"}]) + "\n```")
        self.assertEqual("Q", value[0]["question"])


    def test_csv_direction_expands_db_column_placeholder(self):
        handle, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write('direction\n"Review customer {{sales.id}} with spend {{sales.amount}}"\n')
            templates = load_direction_inputs([csv_path])
            expanded = expand_direction_templates(
                templates, self.db_path, values_per_direction=1
            )
            self.assertEqual(
                "Review customer 1 with spend 10.5", expanded[0]["direction"]
            )
            self.assertEqual(1, expanded[0]["bindings"]["sales.id"])
        finally:
            os.remove(csv_path)

    def test_advisory_direction_creates_paired_insight_task(self):
        class FakeEngine:
            def generate(self, prompt, system=""):
                return json.dumps([{
                    "question": "What is customer 1's transaction amount?",
                    "gold_sql": "SELECT amount FROM sales WHERE id = 1",
                    "direction_id": "direction_1_1",
                    "insight_question": "How could we grow business with customer 1?",
                    "difficulty": "easy",
                    "rationale": "The query retrieves the customer's transaction.",
                }])

        directions = [{
            "id": "direction_1_1",
            "template": "Grow {{sales.id}}",
            "direction": "Grow 1",
            "bindings": {"sales.id": 1},
            "source": "test",
        }]
        candidates = generate_candidates(FakeEngine(), "sales(id, amount)", directions, 1)
        verified = verify_tasks(candidates, self.db_path)
        insights = make_paired_insight_tasks(verified)
        self.assertEqual("How could we grow business with customer 1?", insights[0]["question"])
        self.assertIn("10.5", insights[0]["data"])


if __name__ == "__main__":
    unittest.main()
