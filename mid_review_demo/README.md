# Mid-Review Presentation & Demo Guide (2 Robots + Bidding)

This folder contains a clean, minimal, self-contained implementation of the multi-robot warehouse simulation with **2 heterogeneous robots** specifically designed for your **Mid-Term Review**.

---

## 🚀 How to Run the Visual Demo

Run the following single command from your terminal:

```bash
py -3.12 mid_review_demo/run_demo.py
```

### Controls (In Visual Mode):
* `SPACE`: Pause / Resume simulation
* `RIGHT ARROW`: Advance frame-by-frame (when paused)
* `Q` or `ESC`: Exit simulation

---

## 📁 What Each File Does (File Structure)

* [`simple_env.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_env.py): Wraps the warehouse environment to support 2 heterogeneous robots (`Robot 0: FAST_LIGHT`, `Robot 1: HEAVY_LOAD`) with battery tracking and 80-dim observation vectors.
* [`simple_bidding.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_bidding.py): PyTorch Neural Bidding Network (`SimpleBiddingNetwork`) that outputs continuous bid values (0.0 to 1.0) to evaluate decentralized task auctions.
* [`simple_planner.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_planner.py): Standard grid-based A\* path planning for shortest path navigation around shelf obstacles.
* [`run_demo.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/run_demo.py): Standalone Pygame visual renderer that brings all components together with live HUD stats and event logs.

---

## 🧠 4-Sentence Mid-Review Cheat Sheet for Evaluators

1. **Environment:** *"We built a multi-robot warehouse grid with heterogeneous robots (Fast Light vs Heavy Load) that track battery drain."*
2. **Bidding:** *"Robots evaluate their battery, type, and distance using a PyTorch neural network to output bid scores (0 to 1) for task auctions."*
3. **Navigation:** *"Robots use A\* pathfinding to calculate shortest paths to shelves and goal delivery slots while avoiding shelf obstacles."*
4. **Decentralization:** *"Decision-making is decentralized—each robot calculates its own bid onboard and communicates locally with nearby peers."*

---

## 🛡️ What to say if asked about advanced features:
* **"What about fault tolerance / 12-robot scaling / baseline comparisons?"**
  * **Answer:** *"Those advanced evaluation benchmarks, fault injection tests, and swarm scalability studies are scheduled for our Final Review phase."*
