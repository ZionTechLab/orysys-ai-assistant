---
doc_id: SPEC-003
title: Mobile wallet top-up from other banks
department: payments
document_type: product_spec
access_level: internal
created_date: 2026-04-20
tags: [mobile-wallet, top-up, paybridge, product]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# SPEC-003 – Wallet top-up from other banks

## Goal
Customers top up their wallet from an account at another bank.

## Requirements
- Uses the PayBridge interbank gateway.
- Top-up limit LKR 100,000 per day.
- If PayBridge is unavailable, show a clear message and do not debit the customer.
