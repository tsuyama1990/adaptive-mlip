#!/bin/bash
sed -i 's/exclude = \[\]/exclude = \["tests\/"\]/g' pyproject.toml
if ! grep -q "exclude = \[\"tests\/\"\]" pyproject.toml; then
  sed -i 's/ignore_missing_imports = true/ignore_missing_imports = true\nexclude = \["tests\/", "tutorials\/"\]/g' pyproject.toml
fi
