# HeAlify Doc-Agent 🧑‍⚕️❤️💉

Welcome to the **HeAlify Doc-Agent** repository! This project is an AI-powered healthcare assistant designed to help you understand your medical reports, provide personalized health advice, and assist you with any medical inquiries.

## Features

- **Medical Report Summarization:** Upload your medical reports and get concise summaries, key insights, and actionable advice.
- **Dr. HeAlify Bot:** Chat with our AI-powered doctor for real-time assistance and medical advice.



1. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add your Groq API key and Google API key to the `.env` file:
    ```env
    groq_api_key=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2. **Navigate through the application:**
    - **Home:** Welcome page with an introduction to the HeAlify Doc-Agent.
    - **Medical Report Summarization:** Upload your medical reports in PDF format and receive detailed summaries.
    - **Dr. HeAlify Bot:** Interact with our AI-powered chatbot for medical assistance and advice.

## File Structure

```
.
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not included, to be created)
```




