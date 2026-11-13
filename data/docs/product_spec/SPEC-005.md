---
doc_id: SPEC-005
title: Real-time fraud alert notifications
department: cards
document_type: product_spec
access_level: internal
created_date: 2026-08-10
tags: [fraud, notifications, cards, product]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# SPEC-005 – Real-time fraud alerts

## Goal
Notify customers within 5 seconds of a suspicious card transaction and let them confirm or block the card.

## Requirements
- Push notification first, SMS fallback after 30 seconds.
- "Not me" blocks the card immediately and opens a dispute case.
