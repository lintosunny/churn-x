# ChurnX: AI-Powered Telco Customer Retention
We build an ML-powered churn prediction system for a telecom company that not only identifies at-risk customers and key churn drivers but also uses Agno AI Agent and Rasa Chatbot to deliver personalized offers, automate retention conversations, and boost customer loyalty.

**Use case**: Customer support rep enters customer ID in CRM → backend calls agent API → suggests best retention offer


## Why Churn Matters

* Acquiring new customers costs 5–25x more than keeping existing ones.
* A 5% boost in retention can raise profits by 25–95%.
* High churn signals deeper issues in product, service, or customer experience.

## ChurnX Architecture
![churn-x-architecture](assets/churn-x-main.png)
> ML pipeline trains in EC2 and pushes the latest model to S3. The agent microservice pulls that model and exposes it as an API. Any company system (web, CRM, apps) can call the agent API to get predictions + recommended sales offers. This makes the ML system usable by business teams in real-time, not just a research model.

## ML Architecture
![ml-architecture](assets/ml-architecture.png)
## ML Deployment Architecture
![deployment-architecture](assets/deployment-architecture.png)