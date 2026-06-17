# Project Breakdown: AI-Powered Traffic Prediction & Emission Reduction

This document provides a structured academic and technical overview of the project, suitable for a thesis defense or technical report.

---

### 1. Problem Recap
**The Challenge:** Urbanization has led to chronic traffic congestion. Congested "stop-and-go" traffic produces up to **3-4x more emissions** than free-flowing traffic due to constant idling, braking, and acceleration.
**The Gap:** Most current Traffic Management Systems (TMS) are reactive—they address congestion only after it has formed.
**The Solution:** This project proposes a **proactive** system that predicts congestion 15 minutes in advance, allowing for preemptive traffic flow adjustments to minimize environmental impact.

---

### 2. Literature Insights
-   **One-Stage Detection:** Research (Redmon et al.) proves that YOLO-style models offer the sub-millisecond latency required for real-time ITS (Intelligent Transportation Systems).
-   **Temporal Dependency:** Literature on traffic flow (Hochreiter & Schmidhuber) suggests that congestion is non-linear and seasonal. **LSTM** is the gold standard for capturing these temporal "long-term" dependencies.
-   **Vision-to-Emission:** Recent studies (EPA MOVES) highlight that emission inventories can be accurately estimated by combining vehicle classification data with average speed profiles.

---

### 3. AI/ML Methodology
The system employs a multi-staged AI pipeline:
1.  **Computer Vision (Perception):** YOLOv8 identifies vehicle types and locations.
2.  **State Estimation (Analytics):** Centroid tracking and ROI-polygon analysis calculate instantaneous density and speed.
3.  **Time-Series Forecasting (Prediction):** LSTM takes historical density sequences to predict future states.
4.  **Regression Modeling (Impact):** XGBoost correlates traffic features with 11 different gaseous pollutants.

---

### 4. Dataset and Preprocessing
-   **Primary Dataset:** COCO (Common Objects in Context) for foundational vehicle weights.
-   **Specialized Dataset:** Custom-labeled "Emergency Vehicle" dataset for transfer learning.
-   **Preprocessing Steps:**
    -   **Frame Sampling:** Reducing video input to a manageable 15 FPS to optimize CPU usage.
    -   **Coordinate Normalization:** Mapping bounding box pixels to 1280x720 reference planes for consistent ROI analysis.
    -   **Data Augmentation:** Using brightness/rotation shifts during training to ensure the model works in rain or low light.

---

### 5. Feature Engineering
-   **Weighted Density:** Assigning spatial weights (Bus=2.5, Car=1.0) to represent true road occupancy.
-   **Temporal Sliding Window:** Creating a 20-minute "memory" window of traffic snapshots for the LSTM.
-   **Cyclic Time Encoding:** Converting "Time of Day" into Sine and Cosine waves so the model understands that 11:59 PM and 12:01 AM are close.
-   **Speed Profiling:** Estimating average flow speed as an inverse function of density and vehicle composition.

---

### 6. Model Selection
-   **YOLOv8n (Nano):** Selected for its extreme speed, allowing the system to run on standard laptops without a dedicated GPU.
-   **LSTM:** Chosen over standard RNNs because it avoids the "vanishing gradient" problem, allowing it to remember traffic patterns from much earlier in the day.
-   **XGBoost:** Selected for emission prediction because it handles small, tabular datasets with high non-linearity much better than deep neural networks.

---

### 7. System Architecture
-   **Data Layer:** SQLite for lightweight, high-speed relational storage of traffic logs.
-   **Service Layer:** Flask (Python) as the orchestrator for detection, prediction, and API management.
-   **Processing Layer:** OpenCV for the vision pipeline; PyTorch for neural network inference.
-   **Presentation Layer:** A glassmorphism-themed Dashboard using Chart.js for real-time visualization.

---

### 8. Evaluation Metrics
-   **Detection:** mAP (mean Average Precision) at 0.5 IoU to verify vehicle classification accuracy.
-   **Forecasting:** RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) to measure the gap between predicted and actual congestion.
-   **Inference Speed:** Measured in Milliseconds per Frame (ms/f) to ensure real-time viability.

---

### 9. Responsible AI
-   **Privacy by Design:** The system does not store faces or license plates. All video is processed in-memory or anonymized into "counts" and "stats."
-   **Transparency:** Emission calculations are based on open EPA/EEA factor tables, allowing the math to be audited.
-   **Bias Mitigation:** The model is trained on diverse traffic scenes (day/night, urban/highway) to avoid "environmental bias."

---

### 10. Expected Outcomes
-   **Reduced Emissions:** Proactive management can theoretically reduce junction-level CO2 by **15-25%**.
-   **Emergency Priority:** Automated alerts for ambulances can reduce response times by identifying path-blockages early.
-   **Evidence-Based Urban Planning:** Cities gain a detailed heatmap of where and when the most toxic gases (NOx/PM2.5) are released, allowing for better public health policies.
