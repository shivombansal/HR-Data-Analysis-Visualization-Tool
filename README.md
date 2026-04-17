# 📊 HR Data Analysis & Visualization Tool

A Streamlit-powered web application for analyzing and visualizing HR/employee data through natural language queries and interactive charts. Upload a CSV, ask questions in plain English, and get instant SQL-driven insights and visualizations.

---

## ✨ Features

- **Natural Language Queries** — Ask questions about your HR data in plain English; the app converts them to SQL automatically using GPT-4o-mini via LangChain.
- **AI-Powered Insights** — Generate detailed analytical reports from query results, including business implications and actionable recommendations.
- **Interactive Visualizations** — Create and save Bar Charts, Pie Charts, Line Charts, Scatter Plots, and Histograms using Plotly.
- **Smart Data Filters** — Sidebar filters for age ranges, departments, genders, locations, designations, and more.
- **Column Selector** — Choose exactly which columns to include in query results, organized by category (Core Info, Education, Employment, Payroll, Performance).
- **Optimized SQLite Backend** — Uploaded CSV data is automatically split into normalized relational tables (core, education, employment, payroll, performance) and loaded into an in-memory SQLite database.
- **TPM-Aware Analysis** — Handles large datasets by chunking data and managing token-per-minute limits gracefully.
- **Saved Visualizations** — Save charts to a persistent dashboard grid and remove them as needed.

---

## 🗂️ Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # API keys (not committed to repo)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key
- A Portkey API key (for LLM gateway/observability)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**

   Create a `.streamlit/secrets.toml` file in the project root:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   PORTKEY_API_KEY = "your-portkey-api-key"
   ```
   > ⚠️ Never commit `secrets.toml` to version control. Add it to `.gitignore`.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Dependencies

```txt
streamlit
pandas
plotly
langchain
langchain-openai
langchain-core
portkey-ai
httpx
```

Generate a `requirements.txt` with:
```bash
pip freeze > requirements.txt
```

---

## 📁 Expected CSV Format

The app is designed for HR/People data CSVs. It recognizes the following column categories:

| Category    | Example Columns |
|-------------|----------------|
| Core Info   | Employee Code, Gender, Age, Birth Date |
| Education   | UG Degree, UG CGPA, Highest Education Level |
| Employment  | Department, Designation, Branch / Location, Joining Date |
| Payroll     | Monthly Basic, Monthly CTC, Payroll Month |
| Performance | 2023 Performance ratings, Availed Sick Leave |

> Column names are cleaned automatically (trimmed whitespace, normalized spacing) on upload.

---

## 🧠 How It Works

### Natural Language → SQL
User queries are sent to GPT-4o-mini via LangChain with a system prompt that enforces correct table joins, column aliasing, and schema awareness across the five normalized SQLite tables.

### Data Normalization
On CSV upload, the dataframe is split into five logical tables:
- **core** — employee demographics (unique per Employee Code)
- **education** — academic background
- **employment** — job and contract details
- **payroll** — monthly compensation (unique per Employee Code + Payroll Month)
- **performance** — ratings, KPIs, leave records

### Visualization
Query results and manually selected columns are passed to Plotly Express to render charts. Users can customize axis selections and chart types, then save charts to a persistent grid.

---

## 🔒 Security Notes

- API keys are stored in Streamlit secrets and never exposed in the UI or source code.
- All data is processed in-memory using SQLite — no data is persisted to disk or sent to external servers beyond the LLM API calls.

---

## 🛠️ Customization

**Adding new filter columns:** Edit the `filter_config` dictionary inside `create_filters()`.

**Changing the LLM model:** Update the `model` parameter in `get_llm_client()`. The app currently uses `gpt-4o-mini`.

**Adding new chart types:** Extend the `create_plot()` and `create_query_visualization()` functions with new Plotly chart types.

**Adjusting TPM chunk size:** Modify the `chunk_size` parameter in `analyze_with_tpm_management()` (default: 80,000 characters).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

# Streamlit deployment [link](https://rabiztek-analytics.streamlit.app/)
