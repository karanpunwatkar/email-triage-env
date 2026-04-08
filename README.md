---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: inference.py
pinned: false
---

# 📧 Email Triage OpenEnv Environment

## 🚀 Overview
This project simulates a real-world **email triage workflow** where an AI agent:
- Classifies email priority
- Decides appropriate action
- Generates response

## 🧠 Tasks
- Easy → Meeting email
- Medium → Server down alert
- Hard → Project update response

## ⚙️ Action Space
- acknowledge
- escalate
- reply

## 📥 Observation Space
- sender
- subject
- body

## 🎯 Reward System
- Step-wise reward
- Partial scoring
- Final normalized score (0–1)

## ▶️ Run Locally
```bash
python inference.py