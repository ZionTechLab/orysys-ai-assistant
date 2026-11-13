---
doc_id: ARCH-005
title: Identity and access management architecture
department: security
document_type: architecture
access_level: internal
created_date: 2026-03-08
tags: [identity, sso, mfa, access]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-005 – Identity and access management

## Components
- Staff directory with single sign-on for internal applications.
- MFA using an authenticator app; required for remote access and privileged accounts.
- Privileged access management (PAM) vault for production admin accounts.

## Roles
Access is role based. Roles are reviewed every quarter by line managers.
