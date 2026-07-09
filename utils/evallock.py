"""Cross-process lock that serialises scored evaluations.

WHY THIS EXISTS (the local form of the "serial evaluation" rule)
---------------------------------------------------------------
Our composite has NO runtime term, so CPU contention cannot corrupt the *metric*
directly. What it corrupts is the BUDGET: every run trains for a fixed wall-clock
`TIME_BUDGET` (1 hour), so two runs sharing the single GPU each fit fewer epochs
than either would alone. Epoch count is the hidden confound -- a config compared
against a config that ran alone is compared unfairly. Hence: one scored run at a time.

This is a lock, not a scheduler. No daemon, no queue, no service. It is a single
advisory `flock` on `out/.eval.lock`, held for the duration of train()/test().

Robustness (a stale lock must never deadlock the researcher):
  - flock is released BY THE KERNEL when the holding process dies, however it dies
    (exception, SIGKILL, OOM, power loss). There is nothing to garbage-collect.
  - The lock file's *content* (pid, experiment, start time) is advisory metadata for
    the human, not the locking mechanism.
  - Released on success, on exception, and on interrupt, via the context manager.

Usage:
    from utils.evallock import eval_lock
    with eval_lock('raydpt_e4'):
        ...  # scored training / evaluation

Opt out (screening / throughput tooling only, runtime + epoch count NON-AUTHORITATIVE):
    AADE_NO_EVAL_LOCK=1
"""

import contextlib
import errno
import fcntl
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, 'out', '.eval.lock')


@contextlib.contextmanager
def eval_lock(experiment_name='?', poll_seconds=30.0):
    """Block until this process holds the exclusive scored-evaluation lock.

    Prints who holds it while waiting, so a stuck run is diagnosable from the log
    rather than silent. Set AADE_NO_EVAL_LOCK=1 to bypass (non-authoritative runs).
    """
    if os.environ.get('AADE_NO_EVAL_LOCK'):
        print('[evallock] BYPASSED (AADE_NO_EVAL_LOCK=1) -- '
              'epoch count / runtime from this run are NON-AUTHORITATIVE', flush=True)
        yield None
        return

    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        waited = 0.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if waited == 0.0:
                    try:
                        holder = os.pread(fd, 4096, 0).decode('utf-8', 'replace').strip()
                    except Exception:
                        holder = '(unreadable)'
                    print(f'[evallock] waiting for the scored-evaluation lock; held by: '
                          f'{holder or "(unknown)"}', flush=True)
                time.sleep(poll_seconds)
                waited += poll_seconds
                if waited % 600 < poll_seconds:
                    print(f'[evallock] still waiting ({waited/60:.0f} min)...', flush=True)
        # Held. Record advisory metadata for a human reading the lock file.
        os.ftruncate(fd, 0)
        os.pwrite(fd, f'pid={os.getpid()} exp={experiment_name} '
                      f'since={time.strftime("%Y-%m-%dT%H:%M:%S")}\n'.encode(), 0)
        print(f'[evallock] acquired (exp={experiment_name}, pid={os.getpid()})', flush=True)
        try:
            yield fd
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
            print('[evallock] released', flush=True)
    finally:
        os.close(fd)
