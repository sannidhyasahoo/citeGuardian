# CiteGuardian Submission Checklist

## ✅ Docker Build Creation
- [x] `Dockerfile` exists in repo root
- [x] Builds successfully: `docker build -t openenv-citeguardian .`
- [x] Base image: `ghcr.io/meta-pytorch/openenv-base:latest`
- [x] Exposes port 8000
- [x] Health check configured
- [x] CMD runs FastAPI server

**Test:** `docker build -t openenv-citeguardian .`

---

## ✅ inference.py Execution
- [x] Emits `[START]` before any operations
- [x] Emits `[STEP]` for each action
- [x] Emits `[END]` always (even on failure)
- [x] Handles `ENV_URL` (direct server connection)
- [x] Handles `LOCAL_IMAGE_NAME` (Docker image fallback)
- [x] Catches all exceptions and prints traceback
- [x] Never exits without printing `[END]`

**Test:** `ENV_URL=http://localhost:8000 uv run python inference.py`

---

## ✅ Output Parsing
- [x] `[START]` format: `[START] task=<task> env=<env> model=<model>`
- [x] `[STEP]` format: `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- [x] `[END]` format: `[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>`
- [x] Rewards formatted to 2 decimal places
- [x] Score formatted to 3 decimal places
- [x] `done` and `success` are lowercase booleans
- [x] No newlines within a single log line

**Verify:** Check inference.py log functions match exact format

---

## ✅ Task Validation
- [x] Environment implements 3 task levels (A, B, C)
- [x] Task A: Structural error (missing Results section)
- [x] Task B: Citation orphans (2 errors)
- [x] Task C: Logical inconsistency (100 vs 85 subjects)
- [x] Rewards are cumulative and clamped to [0, 1]
- [x] Perfect run (100% recall, 0 false positives) = 1.0
- [x] `reset()` randomly selects a task
- [x] `observation.task_level` returns 'A', 'B', or 'C'

**Test:** Run environment locally and verify all 3 tasks work

---

## ✅ LLM Criteria Check
- [x] System prompt includes all 5 tools
- [x] System prompt has task-specific strategies
- [x] Agent uses `GO_TO` to navigate sections
- [x] Agent uses `SCAN_CITATIONS` to find markers
- [x] Agent uses `COMPARE_VALUES` for numeric checks
- [x] Agent uses `FLAG_ERROR` with correct error_type
- [x] Agent uses `SUBMIT` to end audit
- [x] LLM responses are parsed as JSON
- [x] Fallback action on parse failure (SUBMIT)

**Test:** Run full inference loop and verify agent completes audit

---

## 🔧 Pre-Submission Commands

```bash
# 1. Build Docker image
docker build -t openenv-citeguardian .

# 2. Run prevalidation (local checks)
uv run python prevalidation.py https://zrypton-citeguardian.hf.space .

# 3. Test inference locally
ENV_URL=https://zrypton-citeguardian.hf.space uv run python inference.py

# 4. Validate with openenv CLI
uv run openenv validate

# 5. Push to HF Space
openenv push --repo-id zrypton/citeGuardian
```

---

## 📋 Required Files

- [x] `openenv.yaml` - OpenEnv manifest
- [x] `pyproject.toml` - Project metadata
- [x] `Dockerfile` - Container definition
- [x] `models.py` - Action & Observation models
- [x] `client.py` - CiteguardianEnv client
- [x] `inference.py` - LLM agent loop
- [x] `server/app.py` - FastAPI application
- [x] `server/citeGuardian_environment.py` - RL environment
- [x] `README.md` - Documentation
- [x] `.gitignore` - Excludes .env, __pycache__, etc.

---

## 🚨 Common Failure Points

### Docker Build
- ❌ Missing dependencies in pyproject.toml
- ✅ All deps listed: openenv-core[core], openai, python-dotenv

### inference.py
- ❌ Crashes before `[START]` is printed
- ✅ `log_start()` called immediately in main()
- ❌ Missing `[END]` on exception
- ✅ `log_end()` in finally block

### Output Format
- ❌ Extra newlines in action JSON
- ✅ `action.model_dump_json(exclude_none=True)` on single line
- ❌ Wrong decimal precision
- ✅ `reward:.2f`, `score:.3f`

### Environment
- ❌ Observation missing required fields
- ✅ All fields: current_view, metadata, audit_log, tool_result, message, task_level, reward, done
- ❌ Reward not in [0, 1]
- ✅ Clamped: `min(max(score, 0.0), 1.0)`

---

## ✅ Final Verification

Run this command to verify everything:

```bash
# Full validation pipeline
docker build -t openenv-citeguardian . && \
uv run openenv validate && \
ENV_URL=https://zrypton-citeguardian.hf.space uv run python inference.py
```

If all three succeed, you're ready to submit.
