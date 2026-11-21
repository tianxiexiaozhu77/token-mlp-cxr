"""Regression: a forked CPU worker must not retain its parent's GPU flock."""
import fcntl
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import eval_public


def main():
    with tempfile.TemporaryDirectory() as tmp:
        def query(command, **kwargs):
            return "GPU-unit-test, test-card, 15\n" if "-i" in command else ""
        with patch.object(eval_public, "REPO", Path(tmp)), \
             patch.object(eval_public.subprocess, "check_output", side_effect=query):
            lease, _ = eval_public.reserve_gpu(2)
            read_fd, write_fd = os.pipe()
            ready_read, ready_write = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(write_fd)
                os.close(ready_read)
                os.write(ready_write, b"r")
                os.close(ready_write)
                os.read(read_fd, 1)
                os._exit(0)
            os.close(read_fd)
            os.close(ready_write)
            os.read(ready_read, 1)
            os.close(ready_read)
            try:
                lease.close()
                with (Path(tmp) / "runs/gpu_locks/eval_gpu_2.lock").open("a+") as second:
                    # Must succeed while the child is alive, proving it closed
                    # the inherited fd rather than extending the GPU lease.
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("PASS: forked child does not retain GPU lease")
            finally:
                os.write(write_fd, b"x")
                os.close(write_fd)
                os.waitpid(child, 0)


if __name__ == "__main__":
    main()
