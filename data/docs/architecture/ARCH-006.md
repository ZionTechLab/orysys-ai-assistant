---
doc_id: ARCH-006
title: Observability and monitoring stack
department: core-banking
document_type: architecture
access_level: internal
created_date: 2026-05-15
tags: [observability, monitoring, alerts, logging]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-006 – Observability stack

## Components
- Metrics and dashboards for all payment services.
- Central log platform with 90-day retention.
- Synthetic monitoring for login, transfer and card authorisation every minute.

## Alert gaps found in incidents
- No alert on connection pool usage (fixed after INC-2026-078).
- No certificate expiry alert for some endpoints (fixed after INC-2026-049).
