#!/usr/bin/env python3
"""Lifecycle and failure tests for the isolated teacher/student launcher."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("teacher_student.py")
spec = importlib.util.spec_from_file_location("teacher_student", SOURCE)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class TeacherStudentTests(unittest.TestCase):
    def args(self, dataset, *extra):
        return pipeline.parse_args([
            "--teacher-model", "teacher.gguf", "--student-model", "student.gguf",
            "--server", "llama-server", "--finetune", "llama-finetune",
            "--dataset", str(dataset), *extra,
        ])

    def test_teacher_finishes_before_student_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "data.jsonl"
            commands = []

            def execute(command, check):
                self.assertTrue(check)
                commands.append(command)
                if len(commands) == 1:
                    self.assertIn("teacher_dataset.py", command[1])
                    self.assertNotIn("llama-finetune", command)
                    dataset.write_text(json.dumps({"messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                    ]}) + "\n", encoding="utf-8")
                else:
                    self.assertEqual(command[0], "llama-finetune")
                    self.assertEqual(command[1:5], ["--model", "student.gguf", "--teacher-jsonl", str(dataset)])
                    self.assertEqual(pipeline.validate_corpus(dataset), 1)

            args = self.args(dataset, "--tasks", "tasks.jsonl", "--tools", "tools.json",
                             "--teacher-server-arg=-ngl", "--teacher-server-arg=12",
                             "--finetune-arg=-c", "--finetune-arg=512")
            self.assertEqual(pipeline.run(args, execute=execute), 1)
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][-3:], ["--", "-ngl", "12"])
            self.assertEqual(commands[1][-2:], ["-c", "512"])

    def test_teacher_failure_never_starts_student(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "data.jsonl"
            commands = []

            def execute(command, check):
                commands.append(command)
                raise RuntimeError("teacher failed")

            with self.assertRaisesRegex(RuntimeError, "teacher failed"):
                pipeline.run(self.args(dataset), execute=execute)
            self.assertEqual(len(commands), 1)

    def test_partial_or_missing_dataset_never_starts_student(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "data.jsonl"
            commands = []
            def execute(command, check):
                commands.append(command)
            with self.assertRaises(FileNotFoundError):
                pipeline.run(self.args(dataset), execute=execute)
            self.assertEqual(len(commands), 1)
            dataset.write_text('{"messages": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no assistant"):
                pipeline.run(self.args(dataset, "--overwrite"), execute=execute)
            self.assertEqual(len(commands), 2)

    def test_refuses_to_overwrite_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "data.jsonl"
            dataset.write_text("existing data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pipeline.run(self.args(dataset), execute=lambda *_args, **_kwargs: None)


if __name__ == "__main__":
    unittest.main()
