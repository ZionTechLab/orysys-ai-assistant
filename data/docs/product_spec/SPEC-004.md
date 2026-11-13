---
doc_id: SPEC-004
title: Corporate salary bulk payments
department: payments
document_type: product_spec
access_level: internal
created_date: 2025-12-05
tags: [bulk-payments, salary, corporate, product, month-end]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# SPEC-004 – Corporate salary bulk payments

## Goal
Corporate customers upload a salary file; employees are credited on the same day.

## Requirements
- Files up to 20,000 lines.
- Files uploaded before 10:00 are credited by 13:00.
- Status report per line available to the corporate customer.

## Peak load
Month-end volume is 4–5x a normal day (see INC-2026-007, INC-2026-034, INC-2026-078).
