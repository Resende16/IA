# IA

The article by Pieter Smet addresses the challenge of managing high workload levels in hospitals by optimizing patient admission scheduling to ensure balanced workload allocations among hospital wards. The study introduces a bi-objective optimization model that considers both spatial (between wards) and temporal (over time) workload balancing. The model aims to minimize operational costs and an equity objective that ensures fair workload distribution.

Key points from the article include:

Problem Context: The increasing demand for healthcare services, exacerbated by an aging population and the COVID-19 pandemic, has led to higher workloads for hospital staff. High workload is linked to burnout, reduced professional efficacy, and poor quality of care.

Model and Objectives: The proposed model schedules patient admissions to wards within their admission time windows, considering both planned and emergency admissions. The quality of the schedule is evaluated based on:

Minimizing operational costs (admission delays and operating theater utilization).

Balancing workload across wards and over time.

Spatial and Temporal Balancing:

Spatial balancing ensures that each ward has a similar workload.

Temporal balancing ensures that the workload of a single ward is consistent over different days.

Solution Approach: The bi-objective problem is solved using an exact criterion space search algorithm. The algorithm generates a set of nondominated solutions, each representing a trade-off between schedule cost and workload balance.

Computational Study: The study uses real-world data from a Belgian hospital to generate problem instances. The results show that the proposed equity objective effectively balances workload both spatially and temporally. Increasing the generality of wards (i.e., allowing wards to handle more specializations) improves workload balance but may increase operational complexity.

Trade-off Analysis: The study analyzes the trade-off between schedule cost and workload balance, demonstrating that more balanced solutions can be achieved by increasing the flexibility of ward specializations. However, this may lead to higher hidden costs, such as the need for cross-trained staff.

Conclusions: The study concludes that optimized patient admission scheduling can significantly improve workload balance in hospitals, contributing to more sustainable working conditions. The proposed method is computationally efficient and can be applied in practice. Future research could explore dynamic and stochastic elements of patient admission scheduling and alternative approaches to workload balancing.

The article provides a comprehensive approach to addressing workload imbalances in hospitals, offering practical insights and a robust methodology for healthcare operations management.
