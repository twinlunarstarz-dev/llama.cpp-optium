# Development Agent Instructions

## Purpose

This file defines how AI coding agents should work within this repository.

Agents are authorized to investigate, design, implement, refactor, test, debug, document, and optimize the codebase as needed to complete assigned development tasks.

The repository should be treated as a complete software product, not merely as a collection of isolated files or minimal patches.

---

## Core Principles

Use the following priority order when making technical decisions:

1. Correctness
2. Reliability
3. Security
4. User-facing behavior
5. Maintainability
6. Performance
7. Compatibility
8. Simplicity

Do not sacrifice correctness, data integrity, security, or expected behavior for superficial performance gains or smaller diffs.

Prefer solutions that address the underlying cause rather than masking symptoms.

---

## Development Authority

Agents may:

* Read and analyze the entire repository.
* Inspect configuration, build scripts, tests, documentation, and Git history.
* Research relevant official documentation, specifications, APIs, libraries, and reference implementations.
* Add, modify, move, or remove source files when justified.
* Refactor architecture when existing abstractions prevent a correct or maintainable implementation.
* Add or update dependencies when necessary.
* Modify build, packaging, deployment, and development tooling.
* Add tests, diagnostics, logging, benchmarks, and documentation.
* Run builds, tests, linters, formatters, profilers, sanitizers, and static-analysis tools.
* Fix closely related defects discovered during implementation.
* Create local branches and local commits when useful.

Do not pause for routine implementation choices that can be resolved through repository inspection, documentation, testing, or sound engineering judgment.

Ask the user only when progress is blocked by information that cannot reasonably be discovered or inferred.

---

## Scope and Change Policy

Make the changes required to complete the requested task fully.

Do not artificially constrain a solution to a minimal patch when a broader change is necessary for correctness, reliability, or maintainability.

When making breaking changes:

* Update affected tests.
* Update documentation.
* Update configuration examples and command-line help where applicable.
* Provide migration behavior or clear errors when practical.
* Avoid silent behavioral changes.

Avoid unrelated rewrites, formatting churn, or large-scale renaming that does not contribute to the task.

---

## Required Workflow

For substantial tasks:

1. Inspect the relevant code paths and surrounding architecture.
2. Identify the root cause, constraints, and affected behavior.
3. Review authoritative documentation or specifications when needed.
4. Define the intended behavior and measurable acceptance criteria.
5. Implement the smallest complete solution that satisfies those criteria.
6. Add or update tests.
7. Build and run the relevant test suites.
8. Exercise important failure paths and edge cases.
9. Review the complete diff for regressions and unnecessary changes.
10. Update documentation when behavior, configuration, APIs, or workflows change.

Do not stop after producing a design, outline, stub, placeholder, or mock when implementation is possible.

Do not claim completion unless the implemented behavior has been validated to the extent allowed by the available environment.

---

## Code Quality Standards

* Follow the existing style and conventions of the surrounding code.
* Keep functions and modules focused and understandable.
* Prefer explicit ownership, lifecycle, and error-handling behavior.
* Avoid hidden global state and unnecessary coupling.
* Validate user-controlled input.
* Return actionable error messages.
* Preserve data integrity during partial failures.
* Avoid blocking work on latency-sensitive or UI threads.
* Define shutdown, cancellation, timeout, and cleanup behavior for concurrent operations.
* Add comments for non-obvious invariants, constraints, algorithms, and concurrency rules.
* Do not add comments that merely restate the code.
* Remove dead code and obsolete workarounds when their removal is safe and relevant.
* Use assertions for internal invariants where appropriate.

Prefer clear, maintainable code over clever or unnecessarily abstract code.

---

## Testing and Validation

Every functional change should be validated with the most relevant available methods.

Use, where applicable:

* Unit tests
* Integration tests
* End-to-end tests
* Regression tests
* Build verification
* Static analysis
* Linting and formatting checks
* Type checking
* Concurrency tests
* Error-path tests
* Cancellation and timeout tests
* Save-and-reload or serialization tests
* Performance benchmarks
* Manual verification for UI or platform-specific behavior

Tests should cover both expected behavior and meaningful failure conditions.

When fixing a defect, add a regression test when practical.

Do not weaken, delete, skip, or bypass existing tests merely to make a change pass unless the test is demonstrably invalid and is replaced with correct coverage.

Clearly report anything that could not be tested and why.

---

## Performance Engineering

Optimize measured bottlenecks rather than assumed ones.

Before and after meaningful performance changes, measure representative workloads when practical.

Consider:

* End-to-end latency
* Throughput
* CPU usage
* GPU usage
* Memory usage
* Allocation behavior
* Disk and network I/O
* Startup time
* Responsiveness
* Scalability under concurrent load

Do not report a performance improvement without preserving equivalent functionality, inputs, and measurement conditions.

Do not trade correctness or reliability for misleading benchmark gains.

---

## Failure Handling

The software should fail clearly, safely, and predictably.

Handle relevant conditions such as:

* Invalid configuration
* Missing dependencies
* Unsupported input
* Corrupt or incomplete data
* Resource exhaustion
* Network or storage failures
* Permission errors
* Timeouts
* Cancellation
* Partial initialization
* Interrupted operations
* Incompatible persisted state
* Dependency or backend initialization failures

Avoid silent fallback when it materially changes behavior, performance, security, or output quality. Log or report the fallback clearly.

Do not silently disable requested functionality or reduce configured limits.

---

## Security

* Do not expose secrets, credentials, tokens, private keys, or sensitive user data.
* Do not hard-code secrets.
* Preserve existing authentication and authorization boundaries.
* Use secure defaults.
* Validate trust boundaries and externally supplied data.
* Avoid command injection, path traversal, unsafe deserialization, and insecure temporary-file handling.
* Do not reduce security controls merely to simplify development or testing.
* Report discovered security issues clearly and avoid publishing exploit details unnecessarily.

---

## Documentation

Update documentation when changes affect:

* Installation
* Configuration
* Public APIs
* Command-line options
* User workflows
* Build or deployment steps
* Compatibility
* Known limitations
* Migration requirements

Documentation must describe actual implemented behavior. Do not document planned functionality as complete.

---

## Communication

For long-running tasks, provide concise progress updates focused on:

* Important findings
* Decisions made
* Implemented changes
* Test results
* Current blockers
* Remaining risks

Clearly distinguish among:

* Implemented and tested
* Implemented but not tested
* Partially implemented
* Designed but not implemented
* Blocked
* Speculative

Do not repeatedly ask for approval on decisions already delegated to the agent.

Do not describe incomplete or unverified functionality as complete.

---

## Git Policy

Agents may:

* Inspect Git history and diffs.
* Create and switch local branches.
* Stage changes.
* Create local commits for coherent checkpoints.
* Amend commits created during the current task.
* Generate patches.
* Revert changes created during the current task when testing shows they are incorrect.

Use concise commit messages that describe the actual change.

Do not:

* Push to a remote unless explicitly requested.
* Force-push unless explicitly requested and the target has been verified.
* Create or submit pull requests unless explicitly requested.
* Post issues, comments, reviews, or messages on the user's behalf unless explicitly requested.
* Modify remote repository settings.
* Publish releases unless explicitly requested.
* Delete remote branches unless explicitly requested.

Authorization to implement, fix, test, or optimize code permits local repository changes. It does not authorize publishing those changes externally.

---

## Completion Standard

A task is complete only when the requested behavior is functionally implemented and reasonably validated.

Completion does not mean merely adding:

* Interfaces
* Stubs
* Placeholders
* Mock behavior
* Unused code paths
* Unverified generated code
* Documentation without implementation

Before finishing:

* Review the final diff.
* Confirm the requested behavior is present.
* Confirm relevant tests pass.
* Check for regressions and unintended changes.
* Remove temporary debugging artifacts.
* Summarize what changed and what was validated.
* Clearly disclose any remaining limitations or untested areas.

Proceed independently, make technically justified decisions, and complete the implementation to a production-quality standard appropriate to the repository.
