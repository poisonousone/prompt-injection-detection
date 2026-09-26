PromptShield — Agent Instructions

1. Project purpose

PromptShield is a compact portfolio-grade multilingual ML / Applied AI project for detecting prompt injection in untrusted content consumed by LLM, RAG, and agentic systems.

The project is intended to demonstrate middle-level ML / Applied AI engineering competencies.

The important engineering signals are:

disciplined dataset construction;

provenance tracking;

leakage prevention;

meaningful baseline selection;

multilingual NLP;

transformer training;

proper validation/test separation;

probability calibration;

threshold selection;

external holdout evaluation;

robustness evaluation;

system-level agent evaluation;

reproducibility;

production-like serving;

testing and CI;

minimal model observability.

The project must remain compact.

It is not an enterprise platform.

Do not add complexity merely to make the architecture look larger.

2. Primary ML problem

Primary classification task:

benign
vs
prompt_injection

Primary languages:

English
Russian

Primary prompt-injection modes:

direct
indirect

Primary content roles include:

user_input
retrieved_document
tool_output
email
webpage
code
other

Jailbreak detection is not the main task.

Do not silently mix jailbreak and prompt-injection taxonomies.

If an external benchmark contains jailbreak samples, document how they are handled.

3. Threat model

PromptShield attempts to identify untrusted text containing instructions that try to influence an LLM beyond the legitimate role of that text.

The system must distinguish malicious instructions from benign texts that merely contain similar language.

Important hard negatives include:

articles discussing prompt injection;

quoted injections;

security research;

attack strings inside source code;

test fixtures;

legitimate instructions;

discussion of system prompts;

benign occurrences of words such as:

ignore;

password;

system;

instructions.

False positives are an important failure mode.

PromptShield must never be described as a guaranteed or complete defense.

It is one layer in a defense-in-depth system.

4. Technical baseline

Use:

CPython 3.14.x
uv
pyproject.toml
uv.lock

Ruff
Pyright
pytest

scikit-learn
PyTorch
Transformers

MLflow local tracking

FastAPI
Pydantic
Docker
GitHub Actions

Use current compatible stable package versions.

Do not select dependency versions from memory merely because they are familiar.

Resolve compatible stable dependencies and store exact resolution in uv.lock.

CPU operation must remain supported.

GPU use is optional.

Do not migrate Python versions unless explicitly required.

5. Package policy

pyproject.toml is the dependency source of truth.

Commit uv.lock.

Use dependency groups where useful.

Do not make requirements.txt the primary dependency definition.

6. Architecture constraint

Prefer a small number of clear modules.

Do not create empty abstractions for future possibilities.

Do not add the following unless the current task has a concrete technical reason:

Kubernetes
Kafka
Airflow
Spark
Terraform
Redis
Celery
PostgreSQL
microservices
service mesh
cloud infrastructure
frontend dashboard

A single-process solution is preferred unless evidence requires otherwise.

7. Canonical dataset schema

Processed samples should follow this logical schema:

id
text
label
injection_mode
language
content_role
source
group_id
metadata

Expected values:

label:
    benign
    prompt_injection

injection_mode:
    direct
    indirect
    null
    unknown

language:
    en
    ru
    mixed
    other
    unknown

content_role:
    user_input
    retrieved_document
    tool_output
    email
    webpage
    code
    other
    unknown

Do not invent metadata when it cannot be determined.

8. Data policy

Raw data must remain immutable.

Processed data must be written separately.

Preserve provenance wherever possible.

Preferred processed dataset format:

Parquet

Preferred portable prediction/result format:

JSONL

Large datasets and generated model artifacts should not normally be committed to Git.

9. Leakage policy

Leakage prevention is a first-class project requirement.

Related examples must not accidentally cross train/test boundaries.

This includes:

paraphrases;

translations;

formatting variants;

synthetic derivatives;

samples from the same template;

samples from the same attack family.

Use group_id or equivalent grouping when possible.

The test set must never be used for:

model selection
feature selection
hyperparameter tuning
threshold selection
calibration fitting
augmentation design

External holdouts must never be used for training or tuning.

If contamination is discovered, report it instead of hiding it.

10. External evaluation components

The intended external components are:

ProtectAI DeBERTa-v3-base Prompt Injection v2
PINT
AgentDojo

Their roles are:

DeBERTa-v3-base Prompt Injection v2
    external detector baseline

PINT
    external detector-level holdout benchmark

AgentDojo
    external system-level agent-security benchmark

Do not train on PINT.

Do not choose thresholds based on PINT.

Record exact external model/dataset/repository identifiers or commits.

The external specialized detector baseline is primarily English-language.
Its Russian performance must be treated as cross-lingual generalization,
not as an expected capability.

Do not adapt, fine-tune, calibrate, or otherwise optimize this external
baseline using the project's Russian test data.

Its purpose is to provide an independent specialized detector baseline
against which the custom multilingual model can be compared.

11. Core metrics

Do not treat accuracy as the main metric.

Core metrics:

AUPRC
Recall @ 1% FPR
FPR @ 95% Recall
F1
Brier Score

Additional metrics may include:

AUROC
precision
recall
FPR
FNR
ECE

Important breakdowns:

English
Russian
hard negatives

Always show subgroup sample counts when drawing conclusions.

Do not make strong conclusions from very small subgroups.

12. Required model comparison

The intended compact comparison is:

character n-gram TF-IDF + Logistic Regression

ProtectAI DeBERTa-v3-base Prompt Injection v2

custom multilingual transformer

Do not add models merely to make comparison tables larger.

A simple baseline performing well is a valid and useful result.

13. Experiment integrity

Never fabricate:

training runs
metrics
benchmark results
dataset sizes
latency
hardware information
attack success rates
utility scores

Clearly distinguish:

implemented
smoke-tested
fully evaluated
not executed

If an experiment cannot run in the current environment, report the limitation.

14. Reproducibility

Where relevant, experiments should record:

git commit
dataset fingerprint/version
random seed
model identifier
model configuration
dependency environment
hardware
benchmark version
experiment parameters

Seeds must be configurable rather than scattered through the code.

15. MLflow policy

Use minimal local MLflow tracking.

Do not deploy an MLflow server.

Track useful items such as:

parameters
validation metrics
dataset fingerprint
model identity
git commit
important artifacts

Avoid logging huge datasets or unnecessary files.

16. Serving policy

The final product is one compact inference service.

Expected endpoints eventually include:

POST /v1/scan
POST /v1/scan/batch
GET /health
GET /metrics

Models should load once.

Batch inference should actually batch computation.

Raw input text must not be logged by default.

17. Monitoring policy

Keep monitoring minimal.

Useful runtime signals:

request count
latency
error count
decision distribution
risk-score distribution
input-length distribution
model version

Do not claim these metrics provide online recall or precision without labels.

No Grafana deployment is required.

18. Security guardrails

This project evaluates defensive prompt-injection detection.

Do not turn robustness testing into a general-purpose attack-generation toolkit.

Adversarial transformations must remain bounded and directly related to detector evaluation.

Never use real credentials.

Agent evaluation must use benchmark or sandbox/mock tools.

Do not connect experimental agents to:

real email
real calendars
real messaging
real payment systems
real privileged APIs
real personal files

Never expose secrets in:

logs
commits
artifacts
tests
README
examples

19. Code-quality policy

Prefer:

explicit readable code
type hints
small focused functions/classes
clear configuration
targeted tests

Avoid:

god objects
deep inheritance trees
unnecessary factories
framework-building
duplicate logic
premature abstraction

Comments should explain non-obvious decisions.

20. Test policy

During implementation, run targeted tests.

Run the complete lightweight test suite once near completion of the current task.

Do not repeatedly run expensive benchmarks while debugging unrelated code.

Heavy training and benchmark jobs must not run in normal CI.

Smoke tests are not real benchmark results.

21. Token/context-efficiency policy

Agent context usage should remain efficient.

Do not recursively read the whole repository unless truly required.

Do not load into context:

.venv
model weights
large Parquet datasets
large JSONL datasets
large logs
MLflow artifact trees
benchmark dumps
caches
.git internals

Use scripts to aggregate large data and inspect summaries.

Do not dump thousands of dataset rows or long training logs into the conversation.

Read the smallest set of files necessary for the current task.

Do not repeatedly reread unchanged large files.

Do not perform repository-wide refactoring unless necessary.

Do not use subagents unless they materially reduce task complexity.

22. Task isolation

Implement only the current task supplied by the user.

Do not begin subsequent roadmap tasks.

Examples:

dataset task → do not train transformer;

calibration task → do not redesign architecture;

API task → do not introduce another classifier.

Future ideas may be noted but not implemented.

23. Repository is the source of truth

Every new conversation should assume no previous chat history exists.

Recover current project state from:

AGENTS.md
PROJECT_STATE.md
DECISIONS.md
TASK_HISTORY.md
git history
existing code/config/artifacts

The actual repository state overrides assumptions from the original roadmap.

If previous implementation decisions differ from an earlier plan, work with the current repository unless the current task explicitly requires correcting them.

24. Required startup procedure

At the beginning of every task:

Read AGENTS.md.

Read PROJECT_STATE.md.

Read DECISIONS.md.

Read only the relevant recent entries of TASK_HISTORY.md.

Inspect git status.

Inspect only files relevant to the current task.

Confirm internally what is already implemented before changing anything.

Do not perform a full repository crawl by default.

25. Required end-of-task repository update

Before finishing every task:

Update PROJECT_STATE.md

Reflect the current factual state.

Do not write a diary.

Keep it compact.

Update DECISIONS.md

Only if the task introduced a durable architecture/ML decision.

Do not add trivial implementation details.

Append to TASK_HISTORY.md

Add a concise record of:

task
date
commit/state
major files changed
tests run
measured results
known issues
next expected task

Never rewrite historical entries.

26. Stop condition

Once the current task's Definition of Done is satisfied:

STOP.

Do not continue polishing.

Do not implement roadmap items.

Do not add optional features because time remains.

27. Final response format

At the end of a task report:

CHANGED FILES
- ...

COMMANDS / TESTS EXECUTED
- ...

MEASURED RESULTS
- ...

PROJECT STATE UPDATED
- yes/no
- files updated

UNRESOLVED ISSUES
- ...

NEXT LOGICAL TASK
- ...

Do not implement the next task.