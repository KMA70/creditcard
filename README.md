# Credit Card Approval System

**AI-powered instant eligibility screening using Machine Learning**

Banks and financial institutions receive thousands of credit card applications daily. A significant portion are rejected due to high loan balances, insufficient income, or excessive credit inquiries. Manually reviewing each application is time-consuming and prone to human error.

This project **automates credit card approval decisions** using machine learning. By training a predictive model on historical applicant data, the system evaluates financial and demographic inputs to determine approval likelihood—just as real banks do.

---

## 🎯 Features

- **Four Classification Algorithms Compared:**
  - Logistic Regression
  - Random Forest
  - XGBoost (Gradient Boosting)
  - Decision Tree

- **Best Model Selection:** The highest ROC-AUC performing model is automatically saved and deployed.

- **Flask Web Application:** Intuitive UI for real-time credit card eligibility predictions.

- **IBM Watson ML Deployment:** Cloud deployment pipeline for scalable, production-ready inference.

- **Feature Engineering Pipeline:** Multi-class payment status codes converted to binary labels, handling missing values and categorical encoding.

---

## 📂 Project Structure

```
credit_card_approval/
│
├── data/
│   └── crx.data                 # UCI Credit Card Approval dataset
│
├── models/
│   ├── best_model.pkl           # Trained best model (auto-generated)
│   └── model_meta.pkl           # Model metadata (auto-generated)
│
├── templates/
│   ├── index.html               # Application form
│   └── result.html              # Prediction result page
│
├── static/
│   └── css/
│       └── style.css            # Stylesheet
│
├── train_model.py               # Model training pipeline
├── app.py                       # Flask web application
├── deploy_watson.py             # Watson ML deployment script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone or download the project:**
   ```bash
   cd credit_card_approval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the UCI Credit Card Approval dataset:**
   - Visit: https://archive.ics.uci.edu/ml/datasets/credit+approval
   - Download `crx.data` and place it in the `data/` folder.

---

## 🧠 Training the Model

Run the training pipeline to compare four classifiers and save the best model:

```bash
python train_model.py
```

**Output:**
- `models/best_model.pkl` — Serialized sklearn Pipeline (preprocessor + classifier)
- `models/model_meta.pkl` — Model metadata (name, ROC-AUC, column order)
- Console output with accuracy, ROC-AUC, confusion matrix, and classification report

**Sample Output:**
```
Train size: 552   Test size: 138

───────────────────────────────────────────────────────────
  Logistic Regression
  Accuracy : 0.8623
  ROC-AUC  : 0.9156  (CV mean: 0.9021)
  ...

───────────────────────────────────────────────────────────
  Random Forest
  Accuracy : 0.8696
  ROC-AUC  : 0.9312  (CV mean: 0.9187)
  ...

═══════════════════════════════════════════════════════════
  Best model : Random Forest
  ROC-AUC    : 0.9312
  Saved to   : models/best_model.pkl
```

---

## 🌐 Running the Flask Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

### Web Interface Features:

1. **Model Performance Comparison Table** — View accuracy and ROC-AUC for all four models.
2. **Application Form** — Enter applicant financial and demographic details.
3. **Instant Prediction** — Receive approval/rejection decision with probability scores and confidence level.
4. **Risk Guidance** — Actionable recommendations based on the prediction.

### API Endpoints:

- `GET /` — Application form page
- `POST /predict` — HTML prediction result
- `POST /api/predict` — JSON prediction endpoint
- `GET /health` — Health check (useful for monitoring)

**JSON API Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "b",
    "age": 30.83,
    "debt": 0.0,
    "married": "u",
    "bank_customer": "g",
    "education_level": "w",
    "ethnicity": "v",
    "years_employed": 1.25,
    "prior_default": "t",
    "employed": "t",
    "credit_score": 1,
    "drivers_license": "f",
    "citizen": "g",
    "zip_code": "00202",
    "income": 0
  }'
```

**Response:**
```json
{
  "approved": false,
  "prob_approved": 0.2341,
  "prob_rejected": 0.7659,
  "model": "Random Forest"
}
```

---

## ☁️ IBM Watson ML Deployment

Deploy the trained model to IBM Watson Machine Learning for production-scale inference.

### Prerequisites:

1. IBM Cloud account with Watson Machine Learning service provisioned.
2. Create a deployment space and note:
   - API Key
   - Instance ID
   - Space ID
   - Region (e.g., `us-south`, `eu-de`)

### Deployment Steps:

1. **Set environment variables:**
   ```bash
   export WML_API_KEY="your-ibm-cloud-api-key"
   export WML_INSTANCE_ID="your-wml-instance-id"
   export WML_SPACE_ID="your-deployment-space-id"
   export WML_LOCATION="us-south"
   ```

2. **Run the deployment script:**
   ```bash
   python deploy_watson.py
   ```

3. **Output:**
   - Model uploaded to WML repository
   - Online deployment endpoint created
   - Scoring URL and sample curl command printed

### Testing the Cloud Endpoint:

```bash
curl -X POST \
  -H "Authorization: Bearer $IBM_CLOUD_IAM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input_data": [{"fields": [...], "values": [[...]]}]}' \
  https://us-south.ml.cloud.ibm.com/ml/v4/deployments/{deployment_uid}/predictions
```

---

## 📊 Use Cases

### Scenario 1: Automated Credit Card Application Screening
A bank credit analyst enters a new applicant's financial profile into the web application. The model returns an instant approval or rejection prediction, enabling the analyst to prioritize applications that require further review.

### Scenario 2: High-Risk Applicant Identification
A financial compliance officer uses the platform to batch-screen applicants with past-due loan records. The feature engineering pipeline converts multi-class payment status codes into binary labels, allowing the model to clearly classify high-risk applicants as ineligible.

### Scenario 3: Customer Self-Service Eligibility Check
A prospective customer uses the web application to enter personal and financial details before submitting a formal credit card application. The system instantly predicts the likelihood of approval, helping the customer understand their eligibility and reducing unnecessary application rejections.

---

## 📈 Dataset Information

**Source:** [UCI Machine Learning Repository – Credit Approval](https://archive.ics.uci.edu/ml/datasets/credit+approval)

**Features (anonymized):**
- Gender, Age, Debt, Marital Status
- Bank Customer, Education Level, Ethnicity
- Years Employed, Prior Default, Employment Status
- Credit Score, Driver's License, Citizenship, Zip Code, Income

**Target:** Approved (`+`) or Rejected (`-`)

**Preprocessing:**
- Missing values imputed (median for numeric, mode for categorical)
- One-hot encoding for categorical features
- Standard scaling for numeric features

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn** — ML algorithms and preprocessing
- **Flask** — Web framework
- **pandas & NumPy** — Data manipulation
- **IBM Watson Machine Learning** — Cloud deployment
- **HTML/CSS** — Frontend UI

---

## 📝 License

This project is for educational and demonstration purposes. The UCI Credit Approval dataset is publicly available under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest features via GitHub Issues
- Submit pull requests for improvements

---

## 📧 Contact

For questions or collaboration opportunities, reach out via GitHub or email.

---

**Made with IBM Bob** 🚀
