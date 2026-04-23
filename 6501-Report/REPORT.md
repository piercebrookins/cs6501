# AI Talks at UVA — A Self-Learning Agentic Calendar Aggregator

**CS 6501 — Agentic AI · Final Project Report**
**Author:** Pierce
**Live deployment:** <https://aiatuva.cs.virginia.edu/events> · <https://aiatuva.cs.virginia.edu/news>
**Demo video:** <https://www.loom.com/share/a7b42a80fb1544539a46ab5430d1c032>

> **Security note.** The production system runs on UVA CS infrastructure and
> ingests a live email inbox under an institutional account. For this reason
> the source code is **not** included in this report or the companion
> repository. This document is written to be **self-contained**: every
> architectural claim, algorithm, and evaluation result below can be
> understood without access to the underlying source.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation](#2-motivation)
3. [Problem Statement](#3-problem-statement)
4. [System Overview](#4-system-overview)
5. [Agent Components](#5-agent-components)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Canonical Data Model](#7-canonical-data-model)
8. [Deployment & Operations](#8-deployment--operations)
9. [Evaluation](#9-evaluation)
10. [Synthesis with Course Topics](#10-synthesis-with-course-topics)
11. [Limitations](#11-limitations)
12. [Future Work](#12-future-work)
13. [Conclusion](#13-conclusion)
14. [Appendix A — Extraction Recipe Schema](#appendix-a--extraction-recipe-schema)
15. [Appendix B — Sample I/O Walkthrough](#appendix-b--sample-io-walkthrough)

---

## 1. Executive Summary

**AI Talks at UVA** is a production-deployed agentic system that unifies
the fragmented landscape of AI-related events and news at the University
of Virginia into a single, searchable, subscribable feed. It ingests from
fourteen heterogeneous sources (custom HTML listings, iCalendar feeds,
WordPress Events plugins, institutional Saffire calendars, and a live
IMAP inbox) and serves a consolidated web dashboard, a JSON API, and RSS
feeds.

The system is **agentic** in a precise sense: the difficult work — figuring
out *how* to extract events from an unknown calendar page, deciding
whether an event is AI-relevant, routing an inbound email to the correct
pipeline, and recovering when a scraper breaks — is delegated to an LLM
operating as a bounded, tool-using agent, while routine operations run
deterministically against cached artifacts called **extraction recipes**.

This "parse once, crawl many" pattern is the single most important
design decision in the system. It reduces the LLM cost of keeping the
calendar fresh by roughly an order of magnitude and makes the system's
behavior predictable, auditable, and cheap enough to run indefinitely
from a shared UVA CS host with no GPU.

At the snapshot taken for this report the system holds **905 events across
14 sources** in a single SQLite database, refreshes every six hours via
cron, and serves the live site behind nginx with a five-minute watchdog
loop.

---

## 2. Motivation

Universities are, organizationally, a confederation of fiefdoms that each
run their own calendar. UVA is typical: the School of Data Science uses
one platform, Engineering uses another, the Library publishes iCalendar,
McIntire runs on Saffire, Karsh and the Miller Center roll their own
WordPress, and dozens of individual labs post events only to a mailing
list or a Google Doc. A graduate student or junior faculty member who
wants to simply *find the AI talks this week* must monitor five to ten
separate websites, none of which share a schema, a taxonomy, or a
refresh cadence.

The concrete trigger for this project was a small but telling failure
mode observed on an earlier prototype that used keyword heuristics for
AI relevance: an event titled **"Agentic Workflows in Practice"** was
scored 0.0 and ranked next to a talk titled **"Taxes"**, also scored 0.0.
The keyword scorer could not see that "agentic" is *the* AI buzzword of
the moment because the dictionary did not contain it. This is a
quintessential case where the right tool is an LLM-as-judge, not a
regex.

A second trigger came from a real stakeholder. Prof. Kautz asked whether
the same ingestion pipeline could handle **AI news** in addition to
events, because the UVA AI community has no consolidated news feed
either. Rather than stand up a second system, the email-ingestion
classifier was extended with a four-way label — `event` / `news` /
`both` / `neither` — and a parallel `news_items` table and `/news`
surface were added. Real users asking for real features are the best
indication that a system is solving an actual problem.

---

## 3. Problem Statement

**Given** a set of seed URLs pointing to university event pages of
unknown and diverse structure,

**produce** a single, deduplicated, AI-relevance-filtered calendar that
is:

- **Correct** — events map to a canonical schema with validated dates,
  times, timezones, and locations.
- **Fresh** — new events appear within one crawl cycle (six hours).
- **Cheap** — the incremental cost of keeping the calendar current is
  dominated by HTTP traffic, not LLM inference.
- **Self-maintaining** — when a source changes its HTML, the system
  detects the break, attempts recovery, and only escalates to a human
  when automatic healing fails.
- **Ethical** — respects `robots.txt`, rate-limits per domain, supports
  opt-out, and attributes every event back to its canonical source.

The research question this framing raises is: **where, in a production
pipeline, should an LLM actually sit?** The answer this project argues
for is: **at the edges, not in the middle.** LLMs infer extraction
recipes once, judge borderline content, and recover from drift — but
they do not touch the routine crawl.

---

## 4. System Overview

### 4.1 Physical Architecture

```
┌─────────────┐   cron every 6h    ┌───────────┐   OpenAI API    ┌───────────┐
│  14 sources │ ─────────────────▶ │  Crawler   │ ─────────────▶ │ gpt-4o    │
│  + IMAP box │                    │ + Dedup    │  (recipe gen   │  -mini    │
└─────────────┘                    └─────┬─────┘   + classify)   └───────────┘
                                         │
                                         ▼
                                   ┌───────────┐
                                   │  SQLite    │
                                   │ 905 events │
                                   └─────┬─────┘
                                         │
                                         ▼
                                   ┌───────────┐   nginx proxy   ┌──────────────┐
                                   │  uvicorn   │ ◀────────────── │ aiatuva.cs.  │
                                   │  :8080     │                 │ virginia.edu │
                                   └───────────┘                 └──────────────┘
```

- **Host:** `portal.cs.virginia.edu` — AMD EPYC, 2 vCPUs, 15 GB RAM, no
  GPU. Deliberately modest; the whole point of the design is that
  production does not need accelerators.
- **Storage:** single SQLite database (`data/ai_talks.db`), nightly
  backup copy.
- **Process model:** one uvicorn web worker, four cron jobs
  (events crawl, news crawl, watchdog, relevance rescorer, email poll).
- **Secrets:** `.env` file outside the repo, loaded at startup; never
  committed.

### 4.2 Logical Layers

- **Ingestion** — seed-URL receiver, calendar discoverer, scheduled
  crawler, IMAP inbox poller.
- **Processing** — Schema Inference Agent (LLM), Extraction Engine
  (no LLM), Change Detection + Self-Healing, Dedup pipeline, BARK
  relevance classifier (LLM), email router + extractor (LLM).
- **Persistence** — a single SQLite file with tables for `events`,
  `news_items`, `sources`, `recipes`, `ai_relevance`,
  `extraction_logs`, and a processed-emails cache.
- **Outputs** — `/events` HTML dashboard, `/news` HTML dashboard,
  JSON API (`/api/v1/*`), RSS feeds (`/feeds/news.rss`,
  `/feeds/news-ai.rss`), ICS export is a near-term roadmap item.

---

## 5. Agent Components

Each numbered component below is implemented as a bounded agent with an
explicit input/output contract. Boundedness matters: none of these
agents are given open-ended tool access, and each has a single job with
a measurable success criterion.

### 5.1 Schema Inference Agent

**Trigger:** A new source URL is added.
**Model:** `gpt-4o-mini` at `temperature=0.1`.
**Input:** the fetched HTML of the listing page, response headers, and
(when available) `robots.txt`.
**Output:** a structured **Extractor Recipe** (see Appendix A) that
specifies source type, CSS selectors, date/time formats, timezone,
pagination strategy, and a confidence score.

The agent is prompted to look first for *structured* alternatives — ICS
feeds, JSON endpoints, embedded widget data sources — and only fall back
to HTML scraping when none are found. This priority ordering is not an
optimization; it is a correctness concern, because ICS/JSON sources
carry timezone and recurrence metadata that HTML cards discard.

If the agent's self-reported confidence is below 0.7, the recipe is
flagged for human review rather than auto-deployed.

### 5.2 Extraction Engine

**Trigger:** Every cron-scheduled crawl.
**Model:** **None.** This component is intentionally LLM-free.
**Input:** a URL and its stored recipe.
**Output:** a list of raw events.

The engine is a thin dispatcher over four handlers (`HTML_LISTING`,
`ICS_FEED`, `JSON_API`, `EMBEDDED_WIDGET`). Each handler is pure
Python, deterministic, and fast. This is the "crawl many" half of
parse-once-crawl-many; the number of LLM calls at steady state is
*zero*.

### 5.3 Change Detection Agent

**Trigger:** After each crawl of a source.
**Inputs:** current page hash, current extraction result, rolling
historical metrics (event count mean/stddev, selector hit rates, DOM
structure fingerprint).

It emits one of four states — `HEALTHY`, `DEGRADED`, `BROKEN`,
`SELF_HEALING` — based on thresholds:

- `HEALTHY` — zero consecutive failures, extracted count within 2σ of
  rolling average, all critical selectors above 0.9 hit rate.
- `DEGRADED` — 1–2 consecutive failures or 2–3σ deviation.
- `BROKEN` — 3+ consecutive failures, extracted count below 10% of the
  mean, or critical selectors below 0.5 hit rate.
- `SELF_HEALING` — recovery in progress.

### 5.4 Self-Healing Agent

**Trigger:** Change Detection returns `BROKEN` or `SELF_HEAL`.
**Strategies, in priority order:**

1. **Selector migration.** Take sample titles from the last known-good
   extraction, find them in the current HTML, and infer new container
   selectors by walking back up the DOM.
2. **ICS / API fallback.** Re-scan the page for any structured feed
   that the original inference missed or that was newly exposed.
3. **Full re-analysis.** Re-run Schema Inference from scratch and diff
   the new recipe against the old.

A healed recipe auto-deploys only if the agent's confidence exceeds
0.8; otherwise it is queued for human review. Every healing attempt is
logged with the strategy used and the outcome, so patterns are visible
over time.

### 5.5 LLM Relevance Classifier (AI Relevance Scoring)

**Model:** `gpt-4o-mini` (JSON mode).
**Why it exists:** A prior keyword-heuristic scorer gave
*"Agentic Workflows in Practice"* a score of 0.0, tying it with a talk
literally named *"Taxes"*. Any scorer that cannot tell those apart is
doing something worse than useless — it is actively masking the signal.

Every newly-ingested event (and every historical event, during a 30-second
full-DB rescore) is sent to the LLM with the title, description, speaker
affiliation, and host unit. The LLM returns a float score in `[0, 1]`
and a short justification, which is stored alongside the event.

**Filtering rule used by the web layer:**

- Events pass if `ai_relevance.score ≥ 0.3`, **or**
- Score is in `[0.2, 0.3)` **and** the source has `ai_focus_score ≥ 0.8`
  (a "trusted source gets a reduced floor" rule that keeps borderline
  talks from AI-centric venues like Karsh, AI Upskilling, and DSI).

### 5.6 BARK — Broadcasting AI-Relevant Knowledge (Email Ingestion)

BARK polls an institutional Gmail inbox every fifteen minutes over IMAP.
Each new message is sent to `gpt-4o-mini` for a two-stage classification:

1. **Route label:** `event` / `news` / `both` / `neither`.
2. **Structured extraction:** if `event`, pull title / start / end /
   venue / speaker / description; if `news`, pull title / summary /
   publish date / author / URL.

Routed items land in the appropriate table (`events` or `news_items`)
under the synthetic source `source_id = email-inbox`. Processed
`Message-ID`s are cached in `data/cache/processed_emails.json` so
replays are idempotent. The fifteen-minute cadence overlaps the
thirty-minute lookback window, giving a built-in retry safety net
without extra code.

### 5.7 Calendar Discovery Agent

Runs weekly. Given a seed URL and a university domain, it chains four
discovery strategies:

1. **Link crawling** (depth ≤ 3, same-domain) for anchors whose text
   contains `events`, `calendar`, `seminars`, `colloquia`, `talks`,
   `workshops`.
2. **Sitemap analysis** via `robots.txt`.
3. **Targeted search** — site-scoped queries like
   `site:virginia.edu "machine learning" "seminar"`.
4. **Department enumeration** — probe known URL conventions
   (`/events`, `/calendar`, Localist subdomains) per department.

Each candidate URL is scored on AI relevance of its host department,
event density, freshness, and public accessibility, then placed in an
onboarding queue for Schema Inference.

### 5.8 Deduplication Pipeline

Cross-posting is the dominant source of noise. A single LLM talk may
appear on the CS calendar, the DSI calendar, the Engineering school
calendar, and in the inbox. The pipeline has three layers:

1. **Exact match.** SHA-256 over `(normalized_title, rounded_start_time,
   normalized_location)`. Normalization strips prefixes like
   `Seminar:` / `Talk:` / `Lecture:`, lowercases, removes punctuation,
   and rounds the start time to the nearest fifteen minutes.
2. **Fuzzy match.** Weighted average of title similarity (0.4), time
   proximity (0.3), speaker similarity (0.2), and location similarity
   (0.1). A threshold of 0.85 flags a pair as duplicates.
3. **Semantic match.** For pairs that pass the fuzzy threshold but look
   suspicious (e.g., very different titles, same speaker + time), a
   sentence-embedding cosine is computed and, if still ambiguous, an
   LLM confirmation call is made.

The **canonical selection rule** prefers the source with the most
complete information, then the most authoritative source, then the
most recent update. Non-canonical events are kept in the database
(for provenance) but hidden from the public view and linked back to
the canonical record.

---

## 6. Key Design Decisions

### 6.1 Parse Once, Crawl Many

The single highest-leverage decision. A naive approach — "call the LLM
every crawl to re-extract events" — would cost roughly **$150–$300 per
month** at this source count and cadence. Caching the inferred recipe
and only invoking the LLM on (a) initial onboarding and (b) detected
breakage collapses that to **~$5–$15 per month**, almost all of which
is the relevance classifier rather than extraction.

The tradeoff is that recipes can go stale silently. The
Change-Detection → Self-Healing loop exists specifically to pay that
risk down.

### 6.2 LLM-as-Judge over Keyword Heuristics

See §5.5. The keyword scorer was fast, cheap, and wrong in exactly the
way that matters (it missed the emerging-terminology case). LLM
scoring is slower (≈0.03 s per event) and non-free (≈$0.01 per full DB
rescore of ~1000 events) but is correct on the cases where correctness
matters most.

### 6.3 Single SQLite Database

Postgres would be nicer in theory and worse in practice. The whole
workload is one writer and a handful of readers on a single host with a
few thousand rows. SQLite is faster to operate, faster to back up
(`cp` on a single file), faster to inspect (`sqlite3` one-liners), and
has zero external dependencies to go wrong. This is an application of
*"simple is better than complex"*: pick the smallest primitive that
solves the problem.

### 6.4 News as a Sibling, Not a Field

When Prof. Kautz asked for news ingestion, the easy path would have
been to add a `type` column to `events` and special-case it everywhere.
Instead, `news_items` is a first-class table with its own schema,
because news has no venue, no speaker, no start/end time, and a very
different relevance notion. Schemas should describe the real shape of
the data, not be warped to fit reuse.

### 6.5 Trusted Source Floor for Borderline Relevance

The AI-relevance filter is `score ≥ 0.3`, *except* that sources with
`ai_focus_score ≥ 0.8` (AI Upskilling, DSI, Karsh, Faculty Affairs, the
email inbox) get a reduced floor of 0.2. This is a small but deliberate
Bayesian adjustment — if a talk shows up on the AI-specific newsletter,
the prior that it is AI-relevant is elevated even when the title is
ambiguous.

---

## 7. Canonical Data Model

Every event in the system, regardless of origin, is projected into a
single normalized schema with the following logical groupings:

- **Identification** — `id` (UUID), `source_id`, content `fingerprint`.
- **Core** — `title`, `description`, optional LLM-generated `summary`.
- **Speaker** — `name`, `affiliation`, `title`, `bio`, `photo_url`.
- **AI relevance** — `score ∈ [0,1]`, `tags`, `topic_category`,
  `is_primary_topic`.
- **Time** — `start_time`, `end_time`, IANA `timezone`, `is_all_day`.
- **Location** — typed as `in-person` / `virtual` / `hybrid`, with
  `venue`, `address`, `building_code`, `room`, and/or `virtual_url` +
  `virtual_platform`.
- **Registration** — `required`, `url`, `deadline`, `capacity`, `cost`.
- **Host** — `unit_name`, `unit_code`, `contact_email`, `contact_name`.
- **Series** — optional grouping (e.g., "AI Seminar Series").
- **Provenance** — `source.url`, `calendar_name`, `platform`, and the
  triplet `first_seen_at` / `last_seen_at` / `scraped_at`.
- **Deduplication** — `is_canonical`, `canonical_id`,
  `alternate_sources`.
- **Metadata** — `status`, `visibility`, `created_at`, `updated_at`.

Validation is enforced both at write time (ISO-8601 timestamps, valid
timezone strings, `score` clamped) and at read time by the JSON API
response layer. Invalid rows are logged to `extraction_logs` and do not
crash the pipeline.

The `news_items` schema is deliberately narrower: `title`, `summary`,
`publish_at`, `author`, `publisher`, `ai_relevance`, and `source_url`.

---

## 8. Deployment & Operations

### 8.1 Scheduled Jobs

| Schedule | Job | Purpose |
|---|---|---|
| `15 0,6,12,18 * * *` | Events crawl | Fetch + dedup across 14 event sources |
| `20 0,6,12,18 * * *` | News crawl | Fetch across 5 news sources (5 min offset to avoid overlap) |
| `30 0,6,12,18 * * *` | Relevance rescore | LLM re-scores newly ingested events |
| `*/15 * * * *` | BARK email poll | IMAP → classify → route |
| `*/5 * * * *` | Watchdog | HTTP probe, restart uvicorn if dead |

### 8.2 Output Surfaces

```
GET /events                  HTML dashboard (paginated, AI-only toggle)
GET /events/{id}             Event detail
GET /news                    HTML dashboard for news
GET /news/{id}               News detail
GET /api/v1/events           JSON (page, per_page, days, ai_only, q)
GET /api/v1/events/{id}      Single event JSON
GET /api/v1/news             JSON news feed
GET /feeds/news.rss          RSS 2.0 — all news
GET /feeds/news-ai.rss       RSS 2.0 — AI-relevant only (score ≥ 0.3)
```

### 8.3 Observability

All long-running components write to dedicated rotating log files:
`cron_crawl.log`, `cron_news_crawl.log`, `cron_rescore.log`,
`cron_email.log`, `web.log`, `watchdog.log`. A daily integrity check
runs `PRAGMA integrity_check` against the SQLite file; failures page a
human.

---

## 9. Evaluation

Because the live system has a real audience and real operational
constraints, the evaluation is a mix of **quantitative** system
measurements and **qualitative** stakeholder signals. Both are
reported honestly, including where numbers are snapshot-based rather
than longitudinal.

### 9.1 Quantitative

**Cost per month.** The dominant cost axis is LLM inference. Measured
at gpt-4o-mini list prices, at four crawls per day over 14 sources plus
the email pipeline:

| Approach | LLM calls / month | Estimated cost / month |
|---|---:|---:|
| Naive (LLM extracts every crawl) | ~3,000 | **$150–$300** |
| **Parse-once-crawl-many (this system)** | ~100 (onboarding + heals + classify) | **$5–$15** |

**Data volume at snapshot.** 905 events across 14 sources; ~1,000
historical events rescored in approximately 30 seconds end-to-end
(≈0.03 s / event, ≈$0.01 total).

**AI relevance classifier.** A small held-out evaluation set of 40
events was hand-labeled as AI-relevant or not and compared to (a) the
prior keyword heuristic and (b) the LLM-as-judge. The LLM scorer
corrected every adversarial case where modern terminology
(*"agentic"*, *"foundation models"*, *"in-context learning"*) was
absent from the keyword list, while preserving negative judgments on
clearly-off-topic talks (administrative, athletic, purely social). The
dollar cost of this correctness improvement is roughly one cent per
full database rescore.

**Self-healing.** The design target is that selector migration succeeds
on a majority of breakage cases without human intervention; over the
operational period observed, the healing path has triggered and
succeeded on every case tried. This sample is small and future work
includes deliberate adversarial testing (see §12).

**Latency.** A full crawl of all fourteen sources completes inside the
six-hour window with substantial margin; the web dashboard responds in
tens of milliseconds because all reads hit the single local SQLite
file with FTS-style indexes on title and description.

**Uptime.** The five-minute watchdog has restarted the uvicorn process
a small number of times (captured in `watchdog.log`); user-visible
downtime has been under five minutes in each case.

### 9.2 Qualitative

**Stakeholder adoption.** The most credible qualitative signal is that
a faculty stakeholder (Prof. Kautz) requested a new vertical — AI news —
and that vertical was delivered, deployed, and is live at
`/news` with its own RSS feed. Real users asking for real features is
the gold standard of "this thing is used."

**Extensibility in practice.** Adding the fourteenth source (the DTD
Lab) required no code changes: a single CLI invocation fetched the
page, ran Schema Inference, persisted the recipe, and added the source
to the scheduler. This is the end state the "parse once" decision was
aiming at.

**Debugging the scorer.** The *"Agentic Workflows = 0.0, Taxes = 0.0"*
incident, recounted in §2, is the clearest qualitative argument for
the LLM-as-judge pattern. Scorers that cannot distinguish the signal
from the absence of signal need to be replaced, not tuned.

**Known rough edges.** News HTML extraction relies on generic
article-card heuristics rather than per-site recipes, so some
event-styled cards slip in on sites that mix events and news on the
same index page. The AI-only filter hides most of them from users but
the underlying data has known imperfections. These are called out
honestly in §11.

---

## 10. Synthesis with Course Topics

This project was built as a deliberate capstone of the semester's
material. Each topic contributed a reusable pattern:

| Topic | Contribution to the Final |
|---|---|
| **T1 — Running an LLM** | Local Qwen 0.8B is wired in as a third LLM provider (`openai` / `gemini` / `local`) for offline or privacy-sensitive operation. |
| **T2 — Frameworks (LangGraph)** | Graph-structured orchestration of the Schema Inference → Validate → Persist → Schedule flow. |
| **T3 — Agent Tool Use** | The Change-Detection → Self-Healing loop is a ReAct-style agent with bounded tools (fetcher, DOM inspector, recipe diff). |
| **T4 — Exploring Tools** | The discovery agent's targeted-search + link-crawl pattern reuses T4's ReAct + external-API exercises. |
| **T5 — RAG** | Embedding-based semantic dedup for cross-posted events is the third layer of §5.8's pipeline. |
| **T6 — VLM** | Listed in future work: multimodal parsing of event-flyer images, reusing T6's VLM setup. |
| **T7 — MCP / A2A** | The extractor-recipe store is conceptually an MCP-style tool registry — tools (per-source extractors) are discovered once and invoked many times. |
| **T8 — Fine-tuning** | Roadmap item: train a small LoRA on logged `(page → recipe)` pairs to replace `gpt-4o-mini` for extraction, reusing T8's SQL-LoRA pipeline. |

---

## 11. Limitations

Honest accounting of what the system does *not* yet do:

1. **News extraction is heuristic, not recipe-driven.** Events get
   per-source recipes; news still uses generic article-card patterns.
   The AI-only filter masks the downstream impact but the underlying
   signal is noisier than it should be.
2. **No cross-source news deduplication.** One story republished by
   two publishers appears twice. Events have a three-layer dedup
   pipeline; news currently has none.
3. **Self-healing test coverage is real but small.** The healing path
   has worked on observed breakages but has not yet been stressed with
   deliberately adversarial platform migrations.
4. **Single-tenant.** The database is scoped to UVA. Supporting
   multiple universities would require tenant partitioning and source
   attribution disambiguation.
5. **No user accounts.** Saved searches, email digests, and
   personalized alerts are designed (see §12) but not implemented.
6. **LLM vendor lock-in at the edges.** The classifier and email
   router currently assume `gpt-4o-mini`-class JSON-mode support.
   Swapping in the local Qwen model works but at lower quality on
   borderline relevance calls.
7. **No automated regression tests for extraction recipes.** Recipe
   validity is checked at inference time and monitored via health
   metrics, but there is no "golden page → golden event list" test
   harness yet.

---

## 12. Future Work

Near-term, in roughly priority order:

- **Per-source news recipes** — close the parity gap with events
  (§11.1).
- **News dedup pass** — at minimum, title-and-publish-date fuzzy match
  across publishers.
- **LoRA-fine-tuned recipe generator** — trained on the accumulated log
  of `(HTML → recipe)` pairs the system has produced, using the
  pipeline from Topic 8. Target: replace `gpt-4o-mini` for extraction
  at 1/10 the cost and comparable quality.
- **VLM flyer ingestion** — many UVA labs post only a PDF or PNG flyer.
  Wiring in a VLM (per Topic 6) would cover sources that currently
  have no textual calendar at all.
- **User accounts + personalized digests** — saved searches, email
  alerts on match, and an "events like this" recommender built on the
  dedup embeddings.
- **Federation** — the framework is not UVA-specific. A multi-tenant
  deployment that served MIT, Stanford, and Berkeley instances with a
  shared recipe library and optional cross-university index is
  described at length in the internal framework document and is a
  natural next step.
- **Adversarial healing tests** — deliberately mutate stored HTML
  fixtures and measure the self-healer's true recovery rate.

---

## 13. Conclusion

The practical lesson of this project is that **LLMs belong at the edges
of production pipelines, not in the middle.** Putting a language model
inside every crawl cycle is slow, expensive, and non-reproducible;
putting one only at the moments that require judgment — inferring how
to read a new calendar, deciding whether a talk is AI-relevant,
routing an ambiguous email, recovering from drift — gets almost all of
the capability at a small fraction of the cost and with full
auditability.

The architectural lesson is that **caching *the plan* is more valuable
than caching the data.** HTTP responses go stale in hours; a good
extraction recipe is good for months. "Parse once, crawl many" is a
specific instance of a more general principle: use expensive,
general-purpose tools to produce cheap, specific-purpose tools, and
run the cheap ones in the hot path.

The system is live at <https://aiatuva.cs.virginia.edu>, used by a
real stakeholder, and cheap enough to keep running. That, more than
any benchmark number, is the evaluation that matters.

---

## Appendix A — Extraction Recipe Schema

Abbreviated to show the essential shape; full JSON Schema lives in the
operational runbook.

```jsonc
{
  "id": "uuid",
  "version": 3,
  "source_url": "https://example.virginia.edu/events",
  "source_type": "HTML_LISTING",          // or ICS_FEED | JSON_API | WIDGET
  "platform": "Localist",                 // optional, for recognition
  "html_config": {
    "selectors": {
      "event_container": "article.event-card",
      "title":           ".event-title a",
      "date":            "time[datetime]",
      "location":        ".event-location",
      "description":     ".event-description",
      "link":            ".event-title a@href",
      "speaker":         ".event-speaker"
    },
    "date_format": "YYYY-MM-DDTHH:mm:ssZ",
    "timezone_source": "explicit",
    "assumed_timezone": "America/New_York",
    "pagination": { "type": "url_param", "param_name": "page", "max_pages": 5 }
  },
  "transformations": [
    { "field": "description", "type": "html_strip" },
    { "field": "start_time",  "type": "timezone_convert",
      "config": { "to": "America/New_York" } }
  ],
  "health": {
    "last_successful_extraction": "2025-04-23T06:15:00Z",
    "consecutive_failures": 0,
    "average_event_count": 18.4,
    "extraction_time_ms_avg": 340
  }
}
```

---

## Appendix B — Sample I/O Walkthrough

A concrete end-to-end trace of a single event flowing through the
system.

**Step 1 — Source onboarding (one-time, LLM).**
Administrator runs `ai-talks add-source https://datascience.virginia.edu/events`.
The Schema Inference Agent fetches the page, identifies a custom HTML
listing, and returns the recipe in Appendix A (confidence 0.92).
Recipe is persisted; source enters the schedule.

**Step 2 — Crawl (every 6h, no LLM).**
Extraction Engine replays the recipe, yielding raw event rows such as:

```jsonc
{
  "title": "Seminar: Large Language Models in Practice",
  "date_text": "2025-05-12T15:00:00-04:00",
  "location_text": "Rice Hall 340",
  "link": "https://datascience.virginia.edu/events/llm-practice",
  "speaker": "Dr. Jane Smith"
}
```

**Step 3 — Normalization.**
The raw row is mapped onto the canonical schema (§7): title is
prefix-stripped for dedup use but preserved for display; `start_time`
is parsed and tagged `America/New_York`; `location.type` is set to
`in-person`; `host.unit_name` is inferred from `source.calendar_name`.

**Step 4 — Deduplication.**
Fingerprint `sha256("large language models in practice|2025-05-12
15:00|rice hall 340")` is computed. A fuzzy pass finds a candidate
duplicate posted to the CS calendar ("LLMs in Practice - AI Seminar"),
weighted-similarity 0.89 ≥ 0.85. The DSI version wins canonical
(fuller description, includes speaker bio); the CS record is linked as
`alternate_sources` and hidden.

**Step 5 — AI Relevance scoring.**
At 30 minutes past the hour, the rescorer sends the event to
gpt-4o-mini, which returns `{ "score": 0.95, "tags": ["nlp", "llm"],
"topic_category": "Natural Language Processing" }`. The event now
passes the `≥ 0.3` filter.

**Step 6 — Surfacing.**
The event is now visible at `/events`, retrievable at
`/api/v1/events/{id}`, and included in the `/feeds/news-ai.rss`-analog
event feed. Total LLM spend for this event's journey: one classify
call, approximately a tenth of a cent.

**Step 7 — Failure mode (hypothetical).**
Six months later, DSI rebuilds their events page. The next crawl
returns zero events. Change Detection flags `BROKEN`. Self-Healing
takes sample titles from the last-good run, finds them under a new
`section.event-row` container, synthesizes new selectors with
confidence 0.87, validates against the current page, and auto-deploys
the new recipe. Total human intervention required: zero.

---

*End of report.*
