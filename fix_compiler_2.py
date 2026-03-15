import re

with open('src/pyacemaker/domain_models/compiler.py', 'r') as f:
    data = f.read()

import signal

data = data.replace('import networkx as nx\nimport signal', 'import networkx as nx\nimport concurrent.futures')
data = data.replace('import networkx as nx', 'import networkx as nx\nimport concurrent.futures')

timeout_old = """        try:
            signal.alarm(30)
            sorted_ids = list(nx.topological_sort(graph))
            return [nodes_dict[nid] for nid in sorted_ids]
        except TimeoutError as err:
            msg = "DAG processing timeout"
            raise CompilerError(msg) from err
        finally:
            signal.alarm(0)"""

timeout_new = """        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda g: list(nx.topological_sort(g)), graph)
                sorted_ids = future.result(timeout=10.0)
            return [nodes_dict[nid] for nid in sorted_ids]
        except concurrent.futures.TimeoutError as err:
            msg = "DAG processing timeout"
            raise CompilerError(msg) from err"""

data = data.replace(timeout_old, timeout_new)

with open('src/pyacemaker/domain_models/compiler.py', 'w') as f:
    f.write(data)
