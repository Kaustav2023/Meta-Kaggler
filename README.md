# 🤖 Autonomous Kaggle Competition Companion

**Multi-Agent ML Automation System Powered by Google Gemini**

Transform your Kaggle workflow from **5 hours to 5 minutes** with intelligent agent orchestration.

---

## 🚀 Quick Start

1. Clone repository
git clone https://github.com/Kaustav2023/Meta-Kaggler
cd Meta-Kaggler

2. Install dependencies
pip install -r requirements.txt

3. Configure API keys
cp .env.example .env

Edit .env and add your GEMINI_API_KEY and Kaggle credentials
4. Run on Iris dataset
python main.py --dataset "uciml/iris" --use-llm

5. See results
cat artifacts/results_*.json

text

**Output:**
🏆 Best Model: RandomForest | Score: 0.97 | Time: 18s
🤖 Agent Evaluation: 0.93/1.0 (Excellent)

text

---

## 💡 What Problem Does This Solve?

Data scientists waste **5-8 hours per Kaggle competition** on repetitive tasks:
- ⏰ Manual data exploration and cleaning
- 🤔 Trial-and-error model selection
- 🐌 Sequential training (one model at a time)
- 🧠 No memory of what worked before

**This system automates the entire pipeline using 5 specialized AI agents.**

---

## 🏗️ Architecture

### Multi-Agent System

User Input: python main.py --dataset "uciml/iris" --use-llm
↓
┌─────────────────────┐
│ CoordinatorAgent │ ← Orchestrates pipeline
└──────────┬──────────┘
│
┌──────────────────────┼──────────────────────┐
▼ ▼ ▼
┌───────────────┐ ┌──────────────────┐ ┌──────────────┐
│ DataAgent │───▶│PreprocessingAgent│───▶│ ModelAgent │
│ │ │ (Gemini 2.5) │ │ (Gemini 2.5) │
│ - KaggleTool │ │ - Analyzes data │ │ - Proposes │
│ - Downloads │ │ - Cleans │ │ models │
│ datasets │ │ - Transforms │ │ - Tunes │
└───────────────┘ └──────────────────┘ └───────┬──────┘
│
▼
┌──────────────────┐
│ TrainingAgent │
│ - Parallel pool │
│ - Trains 6 models│
│ - Selects best │
└─────────┬────────┘
│
▼
┌──────────────────┐
│ EvaluationAgent │
│ - Scores agents │
│ - Recommendations│
└──────────────────┘
│
▼
┌──────────────────┐
│ MemoryBank │
│ - Stores models │
│ - Learns patterns│
└──────────────────┘

text

### Agent Responsibilities

| Agent | Role | Intelligence | Output |
|-------|------|--------------|--------|
| **DataAgent** | Downloads datasets | Kaggle API integration | Raw DataFrame |
| **PreprocessingAgent** | Cleans data | **Gemini 2.5** reasoning | Clean DataFrame |
| **ModelAgent** | Proposes models | **Gemini 2.5** strategy | 6 model configs |
| **TrainingAgent** | Trains models | Parallel execution | Best model |
| **EvaluationAgent** | Evaluates agents | Meta-assessment | Performance scores |

---

## ✨ Features

### 🧠 Gemini-Powered Intelligence
- **Smart Preprocessing:** Gemini analyzes data structure and suggests transformations
- **Adaptive Model Selection:** Proposes models based on dataset characteristics
- **Context-Aware:** Understands semantics (e.g., "Species" is target, not "Id")

### ⚡ Parallel Execution
- **6 Models Simultaneously:** ThreadPoolExecutor trains in parallel
- **6x Faster:** 3 minutes → 30 seconds

### 🧬 Persistent Memory
- **MemoryBank:** Stores best models per dataset
- **Learning System:** Recalls successful approaches

### 📊 Agent Evaluation
- **Meta-Agent:** Evaluates each agent's performance
- **Observability:** Know when preprocessing or training fails
- **Actionable Recommendations:** "PreprocessingAgent scored 0.6 - review logic"

### 🛡️ Production-Ready
- **Error Handling:** Graceful fallbacks for API failures
- **Timeout Protection:** Models don't hang indefinitely
- **Large Dataset Handling:** Samples 50K rows if >250K
- **Structured Logging:** Full audit trail in artifacts/

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Kaggle account
- Google AI Studio API key

### Step-by-Step Setup

#### 1. Clone Repository
git clone https://github.com/Kaustav2023/Meta-Kaggler
cd kaggle-companion

text

#### 2. Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate

Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

text

#### 3. Install Dependencies
pip install -r requirements.txt

text

#### 4. Configure Kaggle API

**Get Kaggle Credentials:**
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`

**Place Credentials:**
Windows
mkdir %USERPROFILE%.kaggle
copy kaggle.json %USERPROFILE%.kaggle\

Mac/Linux
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

text

#### 5. Configure Gemini API

**Get Gemini API Key:**
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

**Create `.env` file:**
cp .env.example .env

text

**Edit `.env`:**
GEMINI_API_KEY=your-gemini-api-key-here

text

---

## 🎮 Usage

### Basic Usage

Run with Gemini intelligence
python main.py --dataset "uciml/iris" --use-llm

Run without Gemini (heuristic mode)
python main.py --dataset "uciml/iris"

Mock mode (no API calls, demo purposes)
python main.py --dataset "demo" --mock --use-llm

text

### Example Datasets

Classification - Iris (150 rows, 4 features)
python main.py --dataset "uciml/iris" --use-llm

Classification - Titanic (891 rows, 11 features)
python main.py --dataset "yasserh/titanic-dataset" --use-llm

Classification - Diabetes (100K rows, 8 features)
python main.py --dataset "iammustafatz/diabetes-prediction-dataset" --use-llm

Classification - Heart Disease (303 rows, 13 features)
python main.py --dataset "johnsmith88/heart-disease-dataset" --use-llm

text

### Output Example

================================================================================
🤖 Autonomous Kaggle Competition Companion
Dataset: uciml/iris
Mode: REAL
Model Proposals: Gemini-powered
🌐 REAL MODE: Downloading dataset 'uciml/iris' from Kaggle...
✅ Loaded 150 rows, 6 columns

🔍 Preprocessing dataset with AI analysis...
🧠 Gemini analyzing dataset structure...
✅ Gemini recommendations:
Target: Species
Features: 4 columns
Task: classification

📋 Applying preprocessing steps...
🏷️ Encoding target column 'Species'
🗑️ Dropping columns: ['Id']
❌ Dropping rows with missing values

✅ Preprocessing complete: (150, 5)

📊 TRAINING RESULTS
🏆 Best Model: rf_deep
Type: RandomForestClassifier
Score: 0.97
Training Time: 0.45s
Artifact: artifacts/models/rf_deep.joblib

📈 All Models:
rf_deep | Type: RandomForestClassifier | Score: 0.9700 | Time: 0.45s

xgb_tuned | Type: XGBClassifier | Score: 0.9667 | Time: 0.38s

lgbm_default | Type: LGBMClassifier | Score: 0.9667 | Time: 1.12s

rf_default | Type: RandomForestClassifier | Score: 0.9600 | Time: 0.32s

logreg | Type: LogisticRegression | Score: 0.9533 | Time: 0.01s

xgb_default | Type: XGBClassifier | Score: 0.9467 | Time: 0.28s

================================================================================

🤖 AGENT EVALUATION
✅ DataAgent:
Score: 1.00 [PASS]

data_loaded: True

dataset_id: uciml/iris

rows: 150

✅ PreprocessingAgent:
Score: 0.85 [PASS]

original_features: 6

final_features: 5

dimensionality_reduction: 0.17

rows_retained: 1.00

✅ ModelAgent:
Score: 0.98 [PASS]

proposals_count: 6

model_types: ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier', 'LogisticRegression']

diversity: 4

best_model_score: 0.97

✅ TrainingAgent:
Score: 0.98 [PASS]

models_trained: 6

success_rate: 1.00

best_model: rf_deep

best_score: 0.97

avg_training_time: 0.43

🏆 Overall Agent Score: 0.95/1.0

💡 Recommendations:
✅ DataAgent performing excellently (1.0)
✅ PreprocessingAgent performing excellently (0.85)
✅ ModelAgent performing excellently (0.98)
✅ TrainingAgent performing excellently (0.98)
🏆 Overall agent system performing well - ready for production

================================================================================

📄 Results saved to: artifacts/results_abc123.json

📊 Dataset Info:
Dataset ID: uciml/iris
Status: completed
Original Shape: (150, 6)
Final Shape: (150, 5)

text

---

## 📁 Project Structure

kaggle-companion/
├── main.py # Entry point
├── config.py # Configuration
├── requirements.txt # Dependencies
├── .env.example # Environment template
├── README.md # This file
├── WRITEUP.md # Competition writeup
│
├── src/
│ ├── agents/
│ │ ├── coordinator_agent.py # Orchestrator
│ │ ├── data_agent.py # Data loading
│ │ ├── preprocessing_agent.py # Gemini-powered cleaning
│ │ ├── model_agent.py # Gemini-powered model selection
│ │ └── training_agent.py # Parallel training
│ │
│ ├── services/
│ │ ├── memory_bank.py # Persistent storage
│ │ ├── session_service.py # State management
│ │ └── agent_evaluator.py # Meta-evaluation
│ │
│ ├── tools/
│ │ └── kaggle_tool.py # Kaggle API integration
│ │
│ └── utils/
│ └── logger.py # Structured logging
│
└── artifacts/ # Generated outputs
├── models/ # Trained models (.joblib)
├── logs/ # Execution logs
├── evaluations/ # Agent scores
└── downloads/ # Cached datasets

text

---

## 🧪 Testing

Quick test with small dataset
python main.py --dataset "uciml/iris" --use-llm

Test without Gemini (fallback mode)
python main.py --dataset "uciml/iris"

Mock mode (no API calls)
python main.py --dataset "demo" --mock

Large dataset test
python main.py --dataset "iammustafatz/diabetes-prediction-dataset" --use-llm

text

---

## 📊 Performance Benchmarks

| Dataset | Rows | Features | Best Model | Accuracy | Time | Speedup |
|---------|------|----------|------------|----------|------|---------|
| Iris | 150 | 4 | RandomForest | 0.97 | 18s | 60x |
| Titanic | 891 | 11 | XGBoost | 0.82 | 32s | 67x |
| Diabetes | 100K | 8 | LightGBM | 0.76 | 145s | 50x |
| Heart Disease | 303 | 13 | RandomForest | 0.85 | 24s | 75x |

**Average time savings: 60x faster than manual workflow**

---

## 🔧 Configuration

### Environment Variables (.env)

Gemini API (required for LLM mode)
GEMINI_API_KEY=your-api-key-here

Kaggle API (auto-loaded from ~/.kaggle/kaggle.json)
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key

System settings (optional)
MAX_PARALLEL_TRAINING=6 # Number of parallel models
MODEL_TIMEOUT_SECONDS=30 # Timeout per model
TRAIN_TEST_SPLIT=0.2 # Validation split
RANDOM_STATE=42 # Reproducibility

text

### config.py Settings

Execution modes
MOCK_MODE = False # True = synthetic data, False = real Kaggle
USE_LLM = True # True = Gemini, False = heuristic

Paths
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"

Training
MAX_PARALLEL_TRAINING = 6
MODEL_TIMEOUT_SECONDS = 30

text

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🏆 Acknowledgments

- **Google Gemini 2.5 Flash** - LLM reasoning engine
- **Kaggle** - Dataset platform
- **Google Agents Intensive** - Course inspiration

---

## 📧 Contact

**Kaustav Dey** - [@LinkedIn](https://www.linkedin.com/in/imkd/)

**Project Link:** https://github.com/Kaustav2023/Meta-Kaggler

---

## 🎥 Video Demo

Watch the 3-minute demo: [YouTube Link]

---

**Built with Passion for Google's Agents Intensive Capstone 2025**

*Transforming hours of manual ML work into minutes of autonomous intelligence.*