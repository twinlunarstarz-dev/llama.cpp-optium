#!/usr/bin/env python3
"""Tests for the isolated teacher-data generator; no model is required."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("teacher_dataset.py")
spec = importlib.util.spec_from_file_location("teacher_dataset", SOURCE)
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)


class TeacherDatasetTests(unittest.TestCase):
    def test_default_categories(self):
        self.assertEqual(len(teacher.load_tasks(None)), 6)

    def test_invalid_tasks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            tasks = Path(folder) / "tasks.jsonl"
            tasks.write_text('{"category":"coding"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt"):
                teacher.load_tasks(tasks)

    def test_tool_transcript_requires_fixture(self):
        first = {"choices": [{"message": {"tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]}}]}
        second = {"choices": [{"message": {"content": "The result is 42."}}]}
        with patch.object(teacher, "request_json", side_effect=[first, second]):
            record = teacher.generate_record("http://localhost", {
                "prompt": "Search", "tool_results": {"search": {"answer": 42}}
            }, "teacher", [], 64)
        self.assertEqual([m["role"] for m in record["messages"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(record["messages"][2]["tool_call_id"], "call-1")
        with patch.object(teacher, "request_json", return_value=first):
            with self.assertRaisesRegex(ValueError, "missing fixture"):
                teacher.generate_record("http://localhost", {"prompt": "Search"}, "teacher", [], 64)

    def test_multiple_tool_rounds_and_repeated_names(self):
        def call(call_id):
            return {"choices": [{"message": {"tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]}}]}
        final = {"choices": [{"message": {"content": "Combined both results."}}]}
        with patch.object(teacher, "request_json", side_effect=[call("c1"), call("c2"), final]):
            record = teacher.generate_record("http://localhost", {
                "prompt": "Search twice", "tool_results": {"search": ["first result", "second result"]}
            }, "teacher", [{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}], 64)
        self.assertEqual([m["role"] for m in record["messages"]],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual(record["messages"][2]["content"], "first result")
        self.assertEqual(record["messages"][4]["content"], "second result")

    def test_server_lifecycle_and_atomic_output(self):
        fake_server = '''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
 def do_POST(self):
  body = json.dumps({"choices": [{"message": {"content": "Teacher response"}}]}).encode()
  self.send_response(200); self.end_headers(); self.wfile.write(body)
port = int(sys.argv[sys.argv.index('--port') + 1])
open(sys.argv[sys.argv.index('-m') + 1], 'w').write(str(os.getpid()))
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            server = folder / "fake-server"
            server.write_text(fake_server, encoding="utf-8")
            server.chmod(0o755)
            pid_file = folder / "server.pid"
            output = folder / "teacher.jsonl"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            subprocess.run([sys.executable, str(SOURCE), "--server", str(server),
                            "--model", str(pid_file), "--output", str(output),
                            "--port", str(port), "--startup-timeout", "10"],
                           check=True, capture_output=True, text=True, timeout=20)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[0]["messages"][1]["content"], "Teacher response")
            self.assertFalse(output.with_name(output.name + ".partial").exists())
            if sys.platform.startswith("linux"):
                self.assertFalse(Path("/proc", pid_file.read_text()).exists())


if __name__ == "__main__":
    unittest.main()
