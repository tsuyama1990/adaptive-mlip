with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

content = content.replace(
"""        def _writer():
            try:""",
"""        def _writer() -> None:
            try:""")

content = content.replace(
"""            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"Error writing to pipe: {e}")
            finally:
                # Open pipe for writing and close it to send EOF
                try:
                    with open(fifo_path, "w") as _:
                        pass
                except Exception:
                    pass""",
"""            except Exception:
                import logging

                logging.getLogger(__name__).exception("Error writing to pipe")
            finally:
                # Open pipe for writing and close it to send EOF
                try:
                    with fifo_path.open("w") as _:
                        pass
                except OSError:
                    pass""")

content = content.replace(
"""        try:
            return self.train(fifo_path, initial_potential)
        finally:
            # Cleanup pipe and temp dir
            try:
                fifo_path.unlink(missing_ok=True)
                Path(tmpdir).rmdir()
            except Exception:
                pass""",
"""        try:
            return self.train(fifo_path, initial_potential)
        finally:
            # Cleanup pipe and temp dir
            try:
                fifo_path.unlink(missing_ok=True)
                Path(tmpdir).rmdir()
            except OSError:
                pass""")

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
