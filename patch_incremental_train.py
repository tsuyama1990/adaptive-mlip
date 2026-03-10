with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

import re

search_inc = """    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: LoopStrategyConfig,
        initial_potential: str | Path | None = None,
    ) -> Any:
        \"\"\"
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        \"\"\"
        # In a real implementation this would merge replay buffer with the new dataset
        # Here we just delegate to train
        _replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size)
        return self.train(new_data_path, initial_potential)"""

replace_inc = """    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: LoopStrategyConfig,
        initial_potential: str | Path | None = None,
    ) -> Any:
        \"\"\"
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        Streams data to pace_train using a named pipe to avoid O(N) memory/disk overhead.
        \"\"\"
        import os
        import tempfile
        import itertools
        from ase.io import iread, write
        import threading

        replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size)

        try:
            new_data_iter = iread(new_data_path, format="extxyz")
        except Exception:
            new_data_iter = iter([])

        combined_iter = itertools.chain(replay_buffer, new_data_iter)

        # Create a named pipe
        tmpdir = tempfile.mkdtemp()
        fifo_path = Path(tmpdir) / "stream.extxyz"
        os.mkfifo(fifo_path)

        # Write to the pipe in a background thread so we don't block
        def _writer():
            try:
                chunk_size = 100
                while True:
                    chunk = list(itertools.islice(combined_iter, chunk_size))
                    if not chunk:
                        break
                    write(str(fifo_path), chunk, format="extxyz", append=True)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error writing to pipe: {e}")
            finally:
                # Open pipe for writing and close it to send EOF
                try:
                    with open(fifo_path, 'w') as _:
                        pass
                except Exception:
                    pass

        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()

        try:
            return self.train(fifo_path, initial_potential)
        finally:
            # Cleanup pipe and temp dir
            try:
                fifo_path.unlink(missing_ok=True)
                Path(tmpdir).rmdir()
            except Exception:
                pass"""

content = content.replace(search_inc, replace_inc)

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
