with open("src/pyacemaker/interfaces/pace_driver_template.py", "r") as f:
    content = f.read()

search = """class PaceDriverConfig(BaseModel):
    potential_path: str = Field(..., pattern=r"^[a-zA-Z0-9_\-\.\/]+$", description="Path to the potential file")"""

# A broader pattern that allows commas, parentheses, brackets, quotes, etc.,
# but forbids shell metacharacters: ;, &, |, `, $, <, >
# We can just forbid dangerous characters using a regex with negative lookahead or character class.
# Using a positive lookahead: only characters NOT in the dangerous set.
replace = """class PaceDriverConfig(BaseModel):
    potential_path: str = Field(..., pattern=r"^[^;&\|`$<>]+$", description="Path to the potential file")"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/pace_driver_template.py", "w") as f:
    f.write(content)
