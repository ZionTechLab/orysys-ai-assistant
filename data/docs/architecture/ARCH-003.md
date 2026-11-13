---
doc_id: ARCH-003
title: Core banking integration layer
department: core-banking
document_type: architecture
access_level: internal
created_date: 2026-01-18
tags: [core-banking, integration, esb, api]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-003 – Core banking integration

## Overview
Channels and the payment hub talk to core banking through an integration layer (ESB) with REST and message queues.

## Patterns
- Postings are sent through a queue with idempotency keys.
- Balance enquiries are synchronous with a 2-second timeout.
- During EOD, postings are queued and applied after EOD completes.

## Constraints
- Core banking allows a maximum of 400 concurrent sessions.
