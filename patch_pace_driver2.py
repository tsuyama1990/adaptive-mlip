with open("src/pyacemaker/interfaces/pace_driver_template.py", "r") as f:
    content = f.read()

search = """class PaceDriverConfig(BaseModel):
    potential_path: str = Field(..., description="Path to the potential file")"""

replace = """class PaceDriverConfig(BaseModel):
    potential_path: str = Field(..., pattern=r"^[a-zA-Z0-9_\-\.\/]+$", description="Path to the potential file")"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/pace_driver_template.py", "w") as f:
    f.write(content)
