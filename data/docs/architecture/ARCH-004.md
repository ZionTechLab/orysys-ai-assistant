---
doc_id: ARCH-004
title: Mobile banking and wallet architecture
department: payments
document_type: architecture
access_level: internal
created_date: 2026-02-25
tags: [mobile, wallet, top-up, paybridge, api-gateway]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-004 – Mobile banking and wallet

## Components
- Mobile app (iOS, Android) → API gateway → mobile backend services.
- **Wallet service** – stored-value wallet; top-ups from bank accounts and other banks.
- Wallet top-up from other banks uses PayBridge through the **legacy PayBridge client** (not the shared resilient client).

## Notes
- Migration of the wallet service to the shared resilient client was planned for Q4 2026 (see INC-2026-095).
