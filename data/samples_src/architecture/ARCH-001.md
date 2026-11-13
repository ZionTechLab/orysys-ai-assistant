---
doc_id: ARCH-001
title: Payments platform architecture overview
department: payments
document_type: architecture
access_level: internal
created_date: 2025-11-20
tags: [payments, payment-hub, architecture, paybridge, card-switch]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-001 – Payments platform overview

## Components
- **Payment hub** – Java services on 2 nodes; handles internal transfers, interbank transfers, bill payments and bulk (salary) payments.
- **Payments database** – primary + read replica.
- **PayBridge client** – connects to the PayBridge interbank gateway (third party).
- **Card switch** – authorises card transactions (see ARCH-002).
- **Channels** – mobile app, internet banking, branch teller, corporate bulk upload.

## Flow – interbank transfer
Channel → API gateway → payment hub → fraud check → PayBridge → core banking posting.

## Known risks
- Single third-party dependency for interbank transfers (PayBridge).
- Month-end load peaks at 4–5x normal volume.
