This project open sources some of the ML models used at Twitter.

Currently these are:

1. The "For You" Heavy Ranker (projects/home/recap).

2. TwHIN embeddings (projects/twhin) https://arxiv.org/abs/2202.05387

## 📊 Algorithm Analysis & Virality Guide

Want to understand how Twitter ranks content and how to optimize your account for maximum reach?

- **[Twitter Virality Report](TWITTER_VIRALITY_REPORT.md)** - Comprehensive 25KB analysis of Twitter's ranking algorithm with detailed strategies for going viral
- **[Quick Reference Guide](QUICK_REFERENCE_GUIDE.md)** - TL;DR version with cheat sheets, checklists, and actionable tips

These documents analyze the Heavy Ranker model and feature engineering to provide practical insights on:
- How tweets are scored (exact engagement weights)
- What features influence ranking (1000+ features analyzed)
- Actionable strategies for maximizing reach
- What to do and what to avoid

## Running the Code

This project can be run inside a python virtualenv. We have only tried this on Linux machines and because we use torchrec it works best with an Nvidia GPU. To setup run

`./images/init_venv.sh` (Linux only).

The READMEs of each project contain instructions about how to run each project.
