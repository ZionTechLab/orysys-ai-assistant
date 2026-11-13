---
doc_id: SPEC-001
title: Instant payments via QR code (QuickPay)
department: payments
document_type: product_spec
access_level: internal
created_date: 2026-02-14
tags: [payments, qr, instant-payments, product]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# SPEC-001 – QuickPay instant QR payments

## Goal
Customers pay merchants by scanning a QR code in the mobile app. Money reaches the merchant within 10 seconds.

## Requirements
- Limit: LKR 200,000 per transaction, LKR 500,000 per day.
- Works 24/7, including during EOD (postings queued).
- Merchant receives an instant notification.

## Non-functional
- 99.9% availability; response time under 3 seconds for 95% of payments.
