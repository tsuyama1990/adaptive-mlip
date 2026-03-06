with open("src/pyacemaker/core/oracle.py") as f:
    content = f.read()

# Add warning logic to DFTManager.compute
content = content.replace("""    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        \"\"\"Internal generator for streaming computations with batching.\"\"\"
        # Use batched processing (chunking) to reuse temporary directories
        # without materializing the whole batch in memory list.
        # However, islice consumes the iterator.

        while True:""", """    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        \"\"\"Internal generator for streaming computations with batching.\"\"\"
        # Use batched processing (chunking) to reuse temporary directories
        # without materializing the whole batch in memory list.
        # However, islice consumes the iterator.

        first_batch = True
        while True:""")

content = content.replace("""            batch = list(islice(structures, batch_size))
            if not batch:
                break""", """            batch = list(islice(structures, batch_size))
            if not batch:
                if first_batch:
                    import warnings
                    warnings.warn("Oracle received empty iterator. No calculations performed.", UserWarning, stacklevel=2)
                break
            first_batch = False""")

with open("src/pyacemaker/core/oracle.py", "w") as f:
    f.write(content)
