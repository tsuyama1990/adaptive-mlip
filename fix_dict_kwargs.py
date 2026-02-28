
with open('src/pyacemaker/core/policy.py') as f:
    content = f.read()

content = content.replace('**kwargs: dict,', '**kwargs: Any,')
content = content.replace('**kwargs: dict)', '**kwargs: Any)')

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)
