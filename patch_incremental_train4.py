with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

# wait, the threading approach:
# we started the thread but didn't wait for it. The pace_train reads the named pipe.
# this is correct. the process block pace_train will read from the pipe, and the thread will write to it.
# when pace_train finishes, we don't need to join the thread explicitly but we should probably avoid hanging if pace_train crashes.
# daemon=True makes it okay.

# let's review if the import is available at the module level.
import_search = """        import os
        import tempfile
        import itertools
        from ase.io import iread, write
        import threading"""
import_replace = """        import os
        import tempfile
        import itertools
        import threading
        from ase.io import iread, write"""
content = content.replace(import_search, import_replace)
with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
