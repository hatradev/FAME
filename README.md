# FAME (Frequency-Aware Multi-scale Ensemble)

FAME is a framework for financial time series forecasting with frequency-aware decomposition and multi-scale ensemble learning. It provides a user-friendly interface (via Streamlit) to upload or select datasets, train hybrid models, visualize results, and evaluate predictive performance.

---

## 1. System Requirements

* **Python**: 3.11.0
* **pip**: Included with Python installation

---

## 2. Required Tools

* **pip** (Python package manager)
* **Code editor** (e.g., VS Code, PyCharm, etc.)
* **Web browser** (to open the Streamlit interface)

---

## 3. Dependencies

All required libraries are listed in:

```
src/requirements.txt
```

---

## 4. Installation

First, clone this repository:

```bash
git clone https://github.com/hatradev/FAME.git
cd FAME
````

Then install the dependencies (make sure you are connected to the internet):

```bash
cd src
pip install -r requirements.txt
```

---

## 5. Project Structure

```
FAME/
├── src/
│   ├── app.py
│   ├── requirements.txt
│   ├── utils/
│   ├── pipeline.py
│   ├── hybrid_model.py
│   ├── ...
├── README.md
├── LICENSE
├── ...
```

---

## 6. Usage

### Run the Application

1. Navigate to the source directory:

   ```bash
   cd src
   ```
2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Once running, the browser will automatically open at:
   [http://localhost:8501](http://localhost:8501)

---

### Interface Guide

* **📁 Select an existing CSV file** from the list.
* **📤 Or upload your own CSV file** with **two required columns**:

  * `time` (date string)
  * `close` (closing price)
* **🧠 Choose a forecasting model** from the sidebar.

The interface will then automatically execute the pipeline and display:

* Raw data vs predictions
* IMF (Intrinsic Mode Function) decomposition (if applicable)
* **LEG** (Layered Explainability Graph)
* Performance of each base model
* Hybrid ensemble model results
* Simulated trading performance

---

### ⚠️ Notes

* **Training can be very time-consuming.**

  * Example: A dataset with \~2,500 rows may take **\~120 minutes** to complete.
* **Recommendation**: Start with small or medium datasets for faster testing and evaluation.

---

### CSV File Requirements

* Must include at least two columns:

  * `time` (string format, e.g., date)
  * `close` (closing price)
* Dataset should have **at least 200 rows** for effective model training.

---

### Exit the Program

* Press `Ctrl + C` in the terminal to stop the server
* Or simply close the browser tab at: [http://localhost:8501](http://localhost:8501)