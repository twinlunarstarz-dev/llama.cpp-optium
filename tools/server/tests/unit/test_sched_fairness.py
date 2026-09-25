import pytest
import time
from utils import *
from concurrent.futures import ThreadPoolExecutor

server = ServerPreset.tinyllama2()

# long enough that the server must chunk it across several steps
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 28
SHORT_PROMPT = "Once upon a time"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 4
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_ubatch = 256
    # needed by the slot-hygiene assertions below
    server.server_slots = True
    server.start()


def test_cancel_during_decode_releases_slot():
    """Aborting mid-generation must release the slot, not leak it."""
    import json
    import urllib.request

    payload = {
        "prompt": SHORT_PROMPT,
        "n_predict": 400,
        "seed": 1,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=10)
    # read a little so the server is really generating, then drop the connection
    resp.readline()
    resp.close()

    time.sleep(2)

    # every slot must be idle again
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    busy = [s for s in res.body if s.get("is_processing")]
    assert busy == [], f"slots still processing after abort: {busy}"

    # and the server must still serve a normal request with identical output
    res = server.make_request(
        "POST", "/completion", {"prompt": SHORT_PROMPT, "n_predict": 8, "temperature": 0.0, "seed": 1234}
    )
    assert res.status_code == 200
    assert res.body["content"] == ", there was a little girl"


def test_cancel_during_prefill_releases_slot():
    """Aborting while a long prompt is being chunked must also release the slot."""
    import json
    import threading
    import urllib.request

    result = {}

    def run():
        payload = {
            "prompt": LONG_PROMPT,
            "n_predict": 4,
            "seed": 2,
            "temperature": 0.0,
            "stream": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            resp.readline()
            resp.close()
        except Exception as e:  # aborted mid-prefill, which is fine
            result["exc"] = str(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "prefill request did not return after abort"

    time.sleep(2)

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    busy = [s for s in res.body if s.get("is_processing")]
    assert busy == [], f"slots still processing after prefill abort: {busy}"


def test_determinism_under_concurrent_load():
    """A fixed seed must give the same output no matter the concurrent load.

    This is the accuracy invariant for a scheduler change: scheduling may move
    *when* a request runs, never *what* it generates.
    """
    def gen(seed, n_predict=16):
        res = server.make_request(
            "POST",
            "/completion",
            {"prompt": SHORT_PROMPT, "n_predict": n_predict, "temperature": 0.0, "seed": seed, "top_k": 1},
        )
        assert res.status_code == 200
        return res.body["content"]

    reference = gen(1234)

    # same request, same seed, under heavy mixed load
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=4) as ex:
            noise = [
                ex.submit(
                    server.make_request,
                    "POST",
                    "/completion",
                    {"prompt": LONG_PROMPT, "n_predict": 4, "temperature": 0.0, "seed": 9000 + i, "cache_prompt": False},
                )
                for i in range(4)
            ]
            under_load = gen(1234)
            for f in noise:
                f.result()

        assert under_load == reference, f"output changed under load: {under_load!r} != {reference!r}"


def test_decode_not_starved_by_long_prefill():
    """A decode request must not wait for long prefills to drain the batch.

    Time-to-first-token is the quantity a scheduler change can move without
    touching token accuracy, so it is the one asserted here.
    """
    import json
    import urllib.request

    def ttft(prompt, n_predict, seed, cache_prompt=False):
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "seed": seed,
            "temperature": 0.0,
            "top_k": 1,
            "stream": True,
            "cache_prompt": cache_prompt,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        first = None
        content = []
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("content"):
                    if first is None:
                        first = time.time() - t0
                    content.append(obj["content"])
        return (first or 0.0) * 1000.0, "".join(content)

    # warm up
    ttft(SHORT_PROMPT, 8, 1)
    ttft(LONG_PROMPT, 4, 2, cache_prompt=False)

    n = 7
    idle = sorted(ttft(SHORT_PROMPT, 32, 1000 + r)[0] for r in range(n))

    loaded = []
    for r in range(n):
        with ThreadPoolExecutor(max_workers=4) as ex:
            noise = [
                ex.submit(ttft, LONG_PROMPT, 4, 2000 + r * 10 + i, False)
                for i in range(4)
            ]
            t, _ = ttft(SHORT_PROMPT, 32, 1000 + r)
            for f in noise:
                f.result()
        loaded.append(t)
    loaded.sort()

    idle_med = idle[len(idle) // 2]
    loaded_med = loaded[len(loaded) // 2]

    # Measured on this fixture: slot-iteration order gave 7.36x, the shared
    # budget gives 1.13x. 4.0 sits between the two with margin on both sides,
    # and is loose enough not to trip on ordinary host load variation.
    ratio = loaded_med / max(idle_med, 1.0)
    assert ratio < 4.0, (
        f"decode TTFT inflated under prefill load: idle {idle_med:.1f} ms, loaded {loaded_med:.1f} ms, "
        f"ratio {ratio:.2f} (idle={idle}, loaded={loaded})"
    )



def test_many_concurrent_agents():
    """Many simultaneous requests must all complete, with no leaked slot."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(
                server.make_request,
                "POST",
                "/completion",
                {"prompt": SHORT_PROMPT, "n_predict": 8, "temperature": 0.0, "seed": 100 + i},
            )
            for i in range(8)
        ]
        results = [f.result() for f in futs]

    assert all(r.status_code == 200 for r in results)
    assert all(r.body["content"] for r in results)

    time.sleep(1)
    res = server.make_request("GET", "/slots")
    busy = [s for s in res.body if s.get("is_processing")]
    assert busy == [], f"slots leaked after 8 concurrent agents: {busy}"
