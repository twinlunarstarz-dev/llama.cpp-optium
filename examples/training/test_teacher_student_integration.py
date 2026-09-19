#!/usr/bin/env python3
"""Process-level teacher/student lifecycle test with synthetic executables."""
from pathlib import Path
import json
import socket
import subprocess
import sys
import tempfile
import unittest

PIPELINE = Path(__file__).with_name("teacher_student.py")

FAKE_SERVER = '''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
 def do_POST(self):
  body = json.dumps({"choices": [{"message": {"content": "Answer from teacher"}}]}).encode()
  self.send_response(200); self.end_headers(); self.wfile.write(body)
port = int(sys.argv[sys.argv.index('--port') + 1])
open(sys.argv[sys.argv.index('-m') + 1], 'w').write(str(os.getpid()))
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''

FAKE_FINETUNE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv
pidfile = pathlib.Path(args[args.index('--model') + 1])
pid = pidfile.read_text().strip()
if sys.platform.startswith('linux') and pathlib.Path('/proc', pid).exists():
 sys.exit('teacher server is still alive when training starts')
dataset = pathlib.Path(args[args.index('--teacher-jsonl') + 1])
records = [json.loads(line) for line in dataset.read_text().splitlines()]
if len(records) != 6 or records[0]['messages'][-1]['content'] != 'Answer from teacher':
 sys.exit('invalid teacher dataset')
pathlib.Path(args[args.index('--marker') + 1]).write_text('trained')
'''


class TeacherStudentIntegrationTests(unittest.TestCase):
    def test_real_process_barrier_and_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            server = folder / "fake-server"
            student = folder / "fake-finetune"
            server.write_text(FAKE_SERVER, encoding="utf-8")
            student.write_text(FAKE_FINETUNE, encoding="utf-8")
            server.chmod(0o755)
            student.chmod(0o755)
            pidfile = folder / "teacher.pid"
            corpus = folder / "teacher.jsonl"
            marker = folder / "trained"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            result = subprocess.run([
                sys.executable, str(PIPELINE),
                "--teacher-model", str(pidfile), "--student-model", str(pidfile),
                "--server", str(server), "--finetune", str(student),
                "--dataset", str(corpus), "--port", str(port),
                "--finetune-arg=--marker", "--finetune-arg=" + str(marker),
            ], capture_output=True, text=True, timeout=35)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "trained")
            self.assertIn("teacher stopped", result.stdout)


if __name__ == "__main__":
    unittest.main()
