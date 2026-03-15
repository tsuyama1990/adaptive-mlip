from pathlib import Path

content = Path("src/pyacemaker/domain_models/heuristics.py").read_text()

# Replace hardcoded smearing fallback logic
old_fallback = """    # Contextual fallbacks
    smearing_type = "gaussian"
    smearing_width = 0.1
    if "Pt" in element_context:
        smearing_type = "mv"
        smearing_width = 0.02"""

new_fallback = """    from pyacemaker.domain_models.defaults import (
        DEFAULT_SMEARING_TYPE,
        DEFAULT_SMEARING_WIDTH,
        ELEMENT_SMEARING_FALLBACKS,
    )

    smearing_type = DEFAULT_SMEARING_TYPE
    smearing_width = DEFAULT_SMEARING_WIDTH
    for el in element_context:
        if el in ELEMENT_SMEARING_FALLBACKS:
            smearing_type = ELEMENT_SMEARING_FALLBACKS[el]["smearing_type"]  # type: ignore[assignment]
            smearing_width = ELEMENT_SMEARING_FALLBACKS[el]["smearing_width"]  # type: ignore[assignment]
            break"""

content = content.replace(old_fallback, new_fallback)
Path("src/pyacemaker/domain_models/heuristics.py").write_text(content)
