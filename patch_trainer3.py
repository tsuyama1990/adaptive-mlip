with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

search = """        # Write to the pipe in a background thread so we don't block
        def _writer() -> None:
            try:
                chunk_size = 100
                while True:
                    chunk = list(itertools.islice(combined_iter, chunk_size))
                    if not chunk:
                        break
                    write(str(fifo_path), chunk, format="extxyz", append=True)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Error writing to pipe")
            finally:
                # Open pipe for writing and close it to send EOF
                try:
                    with fifo_path.open("w") as _:
                        pass
                except OSError:
                    pass"""

replace = """        # Write to the pipe in a background thread so we don't block
        def _writer() -> None:
            try:
                chunk_size = 100
                with fifo_path.open("w") as f_out:
                    while True:
                        chunk = list(itertools.islice(combined_iter, chunk_size))
                        if not chunk:
                            break
                        # Write the chunk directly to the open file object
                        write(f_out, chunk, format="extxyz")
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Error writing to pipe")
            # When the with block exits, f_out is closed, sending EOF to pace_train."""

content = content.replace(search, replace)

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
