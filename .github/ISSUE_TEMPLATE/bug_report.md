---
name: Bug Report
about: Report a bug to help us improve Forge
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Steps to Reproduce

1. Install forge with `pip install forge-features`
2. Run the following code:
   ```python
   # Your code here
   ```
3. See error

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

A clear description of what actually happened.

## Error Message

```
Paste the full error message/traceback here
```

## Environment

- OS: [e.g., Ubuntu 22.04, macOS 14, Windows 11]
- Python version: [e.g., 3.11.5]
- Forge version: [e.g., 0.1.0]
- Relevant dependencies:
  ```
  pip freeze | grep -E "(numpy|pandas|scikit-learn|scipy)"
  ```

## Minimal Reproducible Example

```python
# A minimal code example that reproduces the issue
import pandas as pd
from forge import AutoFeatureTransformer

# Your minimal example here
```

## Additional Context

Add any other context about the problem here.
