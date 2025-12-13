## Gemini CLI - Offload Expensive Discovery

Gemini ingests entire directories in one shot. Use it to **front-load exploration** so you can focus on targeted action.

### The Trade-off

| Approach | Cost | Best For |
|----------|------|----------|
| Grep/Glob/Read | N tool calls, sequential | Known targets, specific queries |
| Gemini | 1 shell call, parallel ingest | Unknown targets, broad questions |

### Trigger: "Would this take 5+ exploratory tool calls?"

If yes, offload to Gemini first. Examples:
- "Where is authentication handled?" → `gemini -p "@src/ Where is auth handled? List files and functions"`
- "Is rate limiting implemented?" → `gemini -p "@src/ @middleware/ Is rate limiting implemented? Show evidence"`
- "How do these services communicate?" → `gemini -p "@services/ How do these services communicate? Trace the flow"`

### Syntax
```bash
gemini -p "@dir/ question"          # one directory
gemini -p "@dir1/ @dir2/ question"  # multiple paths
gemini --all_files -p "question"    # entire project
```

### After Gemini Returns

1. Extract file paths and specific claims from response
2. Read/verify the 2-3 most relevant files directly
3. Proceed with your actual task (edit, implement, fix)

Gemini answers "where/what exists" → You handle "how to change it"

### Skip Gemini When

- User already told you where to look
- You're searching for a specific symbol (Grep is faster)
- The task is editing, not discovery
