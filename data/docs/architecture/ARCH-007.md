---
doc_id: ARCH-007
title: Network segmentation and disaster recovery site design
department: security
document_type: architecture
access_level: confidential
created_date: 2026-04-02
tags: [network, segmentation, dr, firewall]
disclaimer: "Fictional sample data - not real Commercial Bank of Ceylon information."
---
# ARCH-007 – Network segmentation and DR site

## Zones
- Internet DMZ, channel zone, application zone, core banking zone, card (PCI) zone.
- Traffic between zones only through firewalls with explicit rules.

## DR site
- Second data centre 120 km away. Asynchronous database replication.
- Card switch is active-active; payment hub is active-passive (see RB-008).
