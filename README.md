# Financial Data Reconciliation System

A comprehensive Excel sheet reconciliation system that implements both brute force and machine learning approaches for matching financial transactions and identifying subset combinations that sum to target values.

## 🎯 Project Overview

This system addresses the classic **Subset Sum Problem** in financial reconciliation by providing multiple algorithmic approaches:

- **Brute Force Approach**: Direct matching and exhaustive subset search
- **Machine Learning Approach**: Feature engineering and predictive models
- **Advanced Approach**: Genetic algorithms, fuzzy matching, and clustering

## 🚀 Features

### Core Capabilities
- **Excel Data Processing**: Load, clean, and prepare financial data from Excel files
- **Multiple Reconciliation Algorithms**: Brute force, ML, and advanced techniques
- **Subset Sum Problem Solving**: Find combinations of transactions that sum to target amounts
- **Performance Benchmarking**: Compare different approaches across dataset sizes
- **Comprehensive Visualizations**: Interactive dashboards and performance charts
- **Error Handling**: Robust data validation and error recovery

### Algorithmic Approaches

#### 1. Brute Force Reconciliation
- **Direct Matching**: Exact amount matching between transactions and targets
- **Subset Sum Brute Force**: Exhaustive search for transaction combinations
- **Performance Analysis**: Execution time measurement for different dataset sizes

#### 2. Machine Learning Approach
- **Feature Engineering**: Transform reconciliation into ML problem
- **Dynamic Programming**: Optimized subset sum solution
- **Predictive Models**: Random Forest, Logistic Regression, Linear Regression
- **Model Performance**: Cross-validation and accuracy metrics

#### 3. Advanced Techniques
- **Genetic Algorithm**: Evolutionary approach for subset selection
- **Fuzzy Matching**: Similarity-based reconciliation with multiple algorithms
- **Clustering**: DBSCAN-based reconciliation for pattern discovery

## 📁 Project Structure

```
fintech/
├── config/
│   └── settings.py                 # Configuration settings
├── data/
│   ├── processed/                  # Processed data files
│   └── sample/                     # Sample Excel files
│       ├── reconciliation_data.xlsx    # Generated sample data
│       ├── KH_Bank.XLSX               # Real bank data example
│       └── Customer_Ledger_Entries_FULL.xlsx  # Customer ledger data
├── examples/
│   ├── basic_usage.py             # Basic usage demonstration
│   ├── advanced_parsing.py        # Advanced Excel parsing
│   └── performance_demo.py        # Performance benchmarking
├── reports/                       # Generated reports and visualizations
│   ├── performance_demo/          # Performance demo outputs
│   ├── performance_report.md      # Comprehensive performance report
│   ├── performance_heatmap.png    # Performance heatmap visualization
│   ├── interactive_dashboard.html # Interactive Plotly dashboard
│   ├── reconciliation_rate_comparison.png  # Reconciliation rate chart
│   └── execution_time_comparison.png       # Execution time chart
├── scripts/
│   └── run_benchmarks.py          # Benchmark execution script
├── src/
│   ├── core/
│   │   ├── excel_processor.py     # Excel data processing
│   │   ├── brute_force_reconciler.py  # Brute force algorithms
│   │   ├── ml_reconciler.py       # Machine learning approach
│   │   ├── advanced_reconciler.py # Advanced techniques
│   │   ├── performance_analyzer.py # Performance analysis
│   │   └── reconciliation_orchestrator.py # Main orchestrator
│   └── utils/
│       ├── helpers.py             # Utility functions
│       └── validators.py          # Data validation
├── tests/                         # Unit tests
│   └── test_basic_functionality.py # Core functionality tests
├── requirements.txt               # Python dependencies
└── setup.py                      # Package setup
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fintech
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python examples/basic_usage.py
   ```

## 📊 Data Format

### Expected Excel Structure

**Sheet 1 (Transactions):**
- Column A: Transaction Amount (e.g., 150.00, 75.50, 200.00)
- Column B: Description (e.g., "Invoice #001", "Payment XYZ")

**Sheet 2 (Targets):**
- Column C: Target Amount (e.g., 225.50, 300.00, 150.00)
- Column D: Reference ID (e.g., "REF001", "REF002")

### Data Cleaning Features
- Currency symbol removal
- Decimal format standardization
- Missing value handling
- Invalid data filtering

### Sample Data Files
The project includes several sample data files for testing:

- **`data/sample/reconciliation_data.xlsx`**: Generated sample data with clean format
- **`data/sample/KH_Bank.XLSX`**: Real bank transaction data (313KB)
- **`data/sample/Customer_Ledger_Entries_FULL.xlsx`**: Customer ledger entries (1.2MB)

## 🎮 Usage Examples

### Basic Usage

```python
from src.core.reconciliation_orchestrator import ReconciliationOrchestrator

# Initialize orchestrator
orchestrator = ReconciliationOrchestrator(tolerance=0.01)

# Load and prepare data
transactions_df, targets_df = orchestrator.load_and_prepare_data(
    "data/sample/financial_data.xlsx"
)

# Run comprehensive reconciliation
results = orchestrator.run_comprehensive_reconciliation(transactions_df, targets_df)

# Get summary report
print(orchestrator.get_summary_report())
```

### Advanced Usage with Custom Parameters

```python
# Custom tolerance and column mappings
orchestrator = ReconciliationOrchestrator(tolerance=0.05)

# Load with custom column mappings
transactions_df, targets_df = orchestrator.load_and_prepare_data(
    "data/sample/financial_data.xlsx",
    sheet1_name="Transactions",
    sheet2_name="Targets",
    amount_col1="Amount",
    desc_col1="Description",
    amount_col2="Target_Amount",
    ref_col2="Reference_ID"
)

# Run specific approaches
bf_results = orchestrator.brute_force_reconciler.get_reconciliation_summary(
    transactions_df, targets_df
)

ml_results = orchestrator.ml_reconciler.get_ml_reconciliation_summary(
    transactions_df, targets_df
)
```

### Performance Benchmarking

```python
# Run comprehensive benchmark analysis
benchmark_results = orchestrator.run_benchmark_analysis(
    transactions_df, targets_df,
    dataset_sizes=[10, 25, 50, 100, 200]
)

# Create visualizations
visualization_files = orchestrator.performance_analyzer.create_performance_visualizations()
```

## 📈 Performance Analysis

### Benchmarking Capabilities
- **Execution Time Comparison**: Across different approaches and dataset sizes
- **Reconciliation Rate Analysis**: Accuracy measurement
- **Scaling Analysis**: Performance vs dataset size
- **Memory Usage**: Resource consumption tracking

### Visualization Outputs
- **Execution Time Charts**: Line plots comparing approaches
- **Reconciliation Rate Charts**: Accuracy comparison
- **Interactive Dashboard**: Plotly-based 3D visualization
- **Performance Heatmaps**: Matrix visualization of metrics

### Generated Reports and Visualizations

The system automatically generates comprehensive reports and visualizations in the `reports/` directory:

#### 📊 Performance Visualizations
- **`execution_time_comparison.png`** (125KB): Line chart comparing execution times across approaches
- **`reconciliation_rate_comparison.png`** (124KB): Bar chart showing reconciliation accuracy rates
- **`performance_heatmap.png`** (139KB): Heatmap matrix of performance metrics
- **`interactive_dashboard.html`** (4.5MB): Interactive Plotly dashboard with 3D visualizations

#### 📋 Performance Reports
- **`performance_report.md`**: Comprehensive markdown report with:
  - Executive summary
  - Detailed results by approach
  - Performance recommendations
  - Visualization references

#### 📁 Performance Demo Outputs
- **`reports/performance_demo/`**: Directory containing outputs from performance demonstration runs

## 🔧 Configuration

### Tolerance Settings
```python
# High precision (0.001)
orchestrator = ReconciliationOrchestrator(tolerance=0.001)

# Standard precision (0.01)
orchestrator = ReconciliationOrchestrator(tolerance=0.01)

# Low precision (0.1)
orchestrator = ReconciliationOrchestrator(tolerance=0.1)
```

### Algorithm Parameters
```python
# Genetic Algorithm parameters
ga_results = orchestrator.advanced_reconciler.genetic_algorithm_subset_selection(
    transactions_df, targets_df,
    population_size=100,
    generations=200
)

# Fuzzy matching parameters
fuzzy_results = orchestrator.advanced_reconciler.fuzzy_matching_with_similarity_scores(
    transactions_df, targets_df,
    similarity_threshold=0.8
)
```

## 📊 Output and Visualizations

### Generated Files
- **JSON Results**: Comprehensive reconciliation results
- **Visualization Files**: PNG charts with performance metrics
- **Log Files**: Detailed execution logs

### Visualization Contents
- **Execution Time Charts**: Performance comparison across approaches
- **Reconciliation Rate Charts**: Accuracy metrics by dataset size
- **Performance Heatmaps**: Time vs rate trade-off analysis
- **All charts use proper error handling and data validation**

### Viewing Generated Visualizations

1. **Visualization Images**: Open PNG files in any image viewer
2. **Log Files**: Check console output and generated log files for detailed execution information

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Test Categories
```bash
# Test Excel processing
python -m pytest tests/test_excel_processor.py

# Test reconciliation algorithms
python -m pytest tests/test_brute_force_reconciler.py
python -m pytest tests/test_ml_reconciler.py
python -m pytest tests/test_advanced_reconciler.py

# Run basic functionality tests
python tests/test_basic_functionality.py
```

### Test Results
The system includes comprehensive unit tests covering:
- Excel data processing and cleaning
- Brute force reconciliation algorithms
- Machine learning model training and prediction
- Advanced reconciliation techniques
- Performance analysis and benchmarking

## 🚀 Running Examples

### Basic Example
```bash
python examples/basic_usage.py
```

### Advanced Parsing Example
```bash
python examples/advanced_parsing.py
```

### Performance Demonstration
```bash
python examples/performance_demo.py
```

### Command Line Benchmarking
```bash
python scripts/run_benchmarks.py --mode comprehensive --sizes 10,25,50,100
```

## 📋 Requirements

### Core Dependencies
- `pandas>=1.5.0`: Data manipulation and analysis
- `numpy>=1.21.0`: Numerical computing
- `openpyxl>=3.0.0`: Excel file handling
- `scikit-learn>=1.1.0`: Machine learning algorithms
- `matplotlib>=3.5.0`: Static plotting
- `seaborn>=0.11.0`: Statistical visualization
- `plotly>=5.0.0`: Interactive visualizations

### Advanced Dependencies
- `fuzzywuzzy>=0.18.0`: Fuzzy string matching
- `python-Levenshtein>=0.12.0`: String similarity
- `deap>=1.3.0`: Genetic algorithms
- `tqdm>=4.64.0`: Progress bars

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `jupyter>=1.0.0`: Jupyter notebooks
- `ipywidgets>=8.0.0`: Interactive widgets

## 🎯 Use Cases

### Financial Reconciliation
- Bank statement reconciliation
- Invoice matching
- Payment verification
- Expense tracking

### Data Analysis
- Pattern recognition in financial data
- Anomaly detection
- Trend analysis
- Performance optimization

### Research Applications
- Algorithm comparison studies
- Performance benchmarking
- Machine learning research
- Financial data mining

## 📈 Current Status

### ✅ Completed Features
- **Excel Data Processing**: Full implementation with data cleaning
- **Brute Force Algorithms**: Direct matching and subset sum
- **Machine Learning**: Feature engineering and predictive models
- **Advanced Techniques**: Genetic algorithms, fuzzy matching, clustering
- **Performance Analysis**: Comprehensive benchmarking and visualization
- **Testing**: Unit tests for all core functionality
- **Documentation**: Complete README and code documentation

### 🎯 Performance Highlights
- **Multi-approach reconciliation**: Brute force, ML, and advanced techniques
- **Robust visualizations**: Static PNG charts with proper error handling
- **Robust error handling**: Graceful handling of data issues and edge cases
- **Scalable architecture**: Modular design for easy extension

### 📊 Generated Outputs
- **Visualization Charts**: PNG format charts for execution time and reconciliation rates
- **Performance Heatmaps**: Time vs accuracy trade-off analysis
- **Sample Data**: Real and generated financial data for testing

---

