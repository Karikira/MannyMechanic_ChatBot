"""Utility to help scaffold knowledge base entries.

Run this script manually to append placeholder sections for a list of models.
Modify `models` list with actual make/model/year ranges and fill in the details later.

Example usage:
    python generate_car_datasets.py > knowledge_base/car_repair_guide.txt

This tool is not run automatically by the app; it's just a helper for the developer.
"""

models = [
    # add make/model entries like "TOYOTA ALTIS (2001-2023)"
    # you can load from an external source or maintain manually
    "TOYOTA ALTIS (2001-2023)",
    "HONDA BRIO (2012-2023)",
    # ... populate up to 150+ entries ...
]

template = """
{title} — COMMON PROBLEMS
------------------------------------------
Brief description goes here.

1. Example issue one
Cause: ...
Fix: ...
Severity: GREEN

2. Example issue two
Cause: ...
Fix: ...
Severity: YELLOW

3. Example issue three
Cause: ...
Fix: ...
Severity: RED

---
"""

for m in models:
    print(template.format(title=m))
