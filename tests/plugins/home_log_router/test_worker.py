"""HomeLogWorker — drains the queue, applies Throttle, sends under the guard."""
import queue
import threading

from plugins.observability.home_log_router.guard import ReentrancyGuard
from plugins.observability.home_log_router.throttle import Throttle
from plugins.observability.home_log_router.worker import HomeLogWorker


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _worker(rate=100, window=60.0, dedup_window=300.0, sender=None, guard=None):
    clock = FakeClock()
    q: queue.Queue[str] = queue.Queue(maxsize=100)
    throttle = Throttle(rate=rate, window=window, dedup_window=dedup_window, clock=clock)
    guard = guard or ReentrancyGuard()
    w = HomeLogWorker(out_queue=q, throttle=throttle, sender=sender or (lambda m: None), guard=guard)
    return w, q, clock, guard


def test_process_sends_admitted_message():
    sent = []
    w, *_ = _worker(sender=sent.append)
    w.process("hello")
    assert sent == ["hello"]


def test_process_applies_throttle_dedup():
    sent = []
    w, _, clock, _ = _worker(sender=sent.append)
    w.process("dup")
    clock.t = 1.0
    w.process("dup")  # within dedup window -> suppressed
    assert sent == ["dup"]


def test_send_runs_under_guard():
    seen = []
    guard = ReentrancyGuard()
    w, *_ = _worker(sender=lambda m: seen.append(guard.active), guard=guard)
    w.process("x")
    assert seen == [True]            # guard active during the send
    assert guard.active is False     # released afterward


def test_sender_exception_is_swallowed():
    def boom(_):
        raise RuntimeError("send failed")

    w, *_ = _worker(sender=boom)
    w.process("x")  # must not raise


def test_thread_drains_queue():
    received = []
    done = threading.Event()

    def sender(m):
        received.append(m)
        done.set()

    w, q, *_ = _worker(sender=sender)
    w.start()
    try:
        q.put("from-thread")
        assert done.wait(timeout=2.0), "worker thread did not drain the queue"
    finally:
        w.stop()
    assert received == ["from-thread"]


def test_stop_is_idempotent_and_joins():
    w, *_ = _worker()
    w.start()
    w.stop()
    w.stop()  # second stop must not raise


def test_idle_flush_sends_pending_summary():
    sent = []
    w, _, _clock, _ = _worker(rate=1, window=10.0, sender=sent.append)
    w.process("A")          # admitted
    w.process("B")          # suppressed (rate)
    sent.clear()
    w.idle_flush()          # should drain the pending suppression summary
    assert len(sent) == 1
    assert "suppress" in sent[0].lower()


def test_idle_flush_sends_under_guard():
    seen = []
    guard = ReentrancyGuard()
    w, *_ = _worker(rate=1, window=10.0, sender=lambda m: seen.append(guard.active), guard=guard)
    w.process("A")
    w.process("B")  # suppressed
    seen.clear()
    w.idle_flush()
    assert seen == [True]
