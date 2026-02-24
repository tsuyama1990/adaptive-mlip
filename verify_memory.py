import tracemalloc
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import StructureConfig, ExplorationPolicy

def verify_memory_usage():
    tracemalloc.start()

    config = StructureConfig(
        elements=["H"],
        supercell_size=[1,1,1],
        active_policies=[ExplorationPolicy.RANDOM_RATTLE]
    )
    generator = StructureGenerator(config)

    # Generate large number of structures
    n = 10000
    count = 0
    snapshot1 = tracemalloc.take_snapshot()

    for _ in generator.generate(n_candidates=n):
        count += 1
        if count == 100:
             # Take snapshot after some items
             snapshot2 = tracemalloc.take_snapshot()

    snapshot3 = tracemalloc.take_snapshot()

    # Compare
    stats = snapshot3.compare_to(snapshot1, 'lineno')

    print(f"Generated {count} structures.")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage:    {peak / 1024 / 1024:.2f} MB")

    tracemalloc.stop()

if __name__ == "__main__":
    verify_memory_usage()
