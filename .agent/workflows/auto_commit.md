---
description: Auto-snapshot the repository before making changes
---
// turbo-all

1. Snapshot current state
   ```pwsh
   git add -A
   git commit -m "snapshot prior to change" --quiet || $true
   ```
