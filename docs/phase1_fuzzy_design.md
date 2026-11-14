# CyberGuard - Phase 1 (Member 2: Fuzzy Logic + Genetic Algorithm)

**Phase:** 1 - Data Preparation (fuzzy side)
**Task:** Choose fuzzy inputs. Draft membership shapes and rule table.

## 1. Fuzzy inputs and output

| Variable | Source (CICIDS2017 column) | Range used | Why it is used |
|---|---|---|---|
| Anomaly score | Output of Member 1's autoencoder | 0 - 1 | Main signal; interface between ANN and fuzzy |
| Packet rate | `Flow Packets/s` | log10(1 + x), 0 - 6 | DoS / DDoS floods give very high packet rates |
| Failed-connection proxy | `SYN Flag Count` + `RST Flag Count` | 0 - 10 | SYN floods and port scans give many SYN and RST flags |
| **Risk (output)** | computed by fuzzy engine | 0 - 100 | Final risk score |

Packet rate is log-scaled because raw values range from near 0 to over a million.

## 2. Membership functions (trapezoid [a, b, c, d])

| Variable | Low | Medium | High |
|---|---|---|---|
| Anomaly (0-1) | [0.00, 0.00, 0.15, 0.40] | [0.20, 0.45, 0.55, 0.80] | [0.60, 0.85, 1.00, 1.00] |
| log10(1 + Packets/s) | [0.0, 0.0, 1.5, 3.0] | [2.0, 3.0, 3.5, 4.5] | [3.8, 5.0, 6.0, 6.0] |
| SYN + RST count | [0.0, 0.0, 1.0, 3.0] | [2.0, 4.0, 5.0, 7.0] | [5.0, 7.5, 10.0, 10.0] |

| Output set | Parameters [a, b, c, d] |
|---|---|
| Low | [0, 0, 15, 35] |
| Medium | [25, 40, 50, 65] |
| High | [55, 70, 80, 90] |
| Critical | [80, 92, 100, 100] |

See `membership_functions.png` for the plots.

## 3. Rule table (AND = min, aggregation = max)

| # | Anomaly | Packet rate | Flags | Risk |
|---|---|---|---|---|
| 1 | High | any | High | Critical |
| 2 | High | High | any | Critical |
| 3 | High | any | any | High |
| 4 | Medium | High | any | High |
| 5 | Medium | any | High | High |
| 6 | Medium | Low or Medium | any | Medium |
| 7 | Low | High | High | High |
| 8 | Low | High (or flags High) | | Medium |
| 9 | Low | Low | Low | Low |
| 10 | Low | Medium | Low | Low |

Rules 7 and 8 let the fuzzy layer flag flood-like traffic even when the ANN gives a low anomaly score.

## 4. Points to agree with Member 1 (shared)

- Feature list: `Flow Packets/s`, `SYN Flag Count` and `RST Flag Count` must be kept in data prep (column names have leading spaces in the raw CSVs, so strip them).
- Unseen-attack categories: decide which attack types are held out. They are never used in training or in the GA.
- Anomaly score must be scaled to 0-1.

## 5. To verify once data is loaded

In CICIDS2017, SYN and RST flag counts are often only 0 or 1 per flow. Plot the real distribution of both flag columns and `Flow Packets/s` on the training data. If needed, shrink the flag range and shift the packet-rate sets. The GA in Phase 4 will refine the shapes later, but the starting ranges should match the real data.
