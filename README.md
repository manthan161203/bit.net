# BitNet Chat 🤖

A modern Streamlit-based chat interface for the **BitNet b1.58** (1-bit LLM) model. This application allows you to interact with the BitNet 2B model in a user-friendly chat environment.

## Features

- **Streamlit UI**: Clean and responsive chat interface.
- **Optimized for CPU**: Designed to run efficiently on CPU environments like Streamlit Cloud.
- **Chat History**: Maintains conversation context during your session.
- **Model Caching**: Loads the model once and reuses it for faster subsequent interactions.

## Getting Started

### Prerequisites

- Python 3.10 or 3.11 (Python 3.11 is recommended for stability)
- Virtual Environment (highly recommended)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd BitNet
    ```

2.  **Set up a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

To start the chat application, run the following command:

```bash
streamlit run app.py
```

The application will be available in your browser at `http://localhost:8501`.

## Troubleshooting

- **Slow Inference**: Running BitNet on CPU can be slow due to weight unpacking. This is expected behavior on hardware without a compatible GPU.
- **Memory Issues**: If the app crashes on deployment, try reducing `max_new_tokens` in `app.py`.
- **Import/Loading Errors**: Ensure you have cleared the cache on Streamlit Cloud before re-deploying if you face mysterious version conflicts.

## Technical Details

- **Model**: `microsoft/bitnet-b1.58-2B-4T`
- **Precision**: `float32` (Optimized for CPU stability)
- **Library**: Custom `transformers` fork with BitNet support.

