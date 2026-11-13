---
doc_id: ARCH-002
title: Card switch and merchant acquiring architecture
department: cards
document_type: architecture
access_level: internal
created_date: 2025-12-15
tags: [cards, card-switch, acquiring, tls, certificate]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-002 – Card switch and acquiring

## Components
- **Card switch** – routes card authorisations between POS/e-commerce, the card networks and core banking.
- **Acquiring gateway** – connects our merchant POS terminals.
- **HSM** – holds PIN and card keys.

## Security
- All network links use mutual TLS. Each link has its own client certificate.
- Certificates are listed in the central certificate inventory (added after INC-2025-118).

## Availability
- Active-active in two data centres. Stand-in processing approves low-value transactions if core banking is down.
