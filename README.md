# caohoc-nlp-lab2

1. Download checkpoint [here](https://drive.google.com/file/d/1v7qkdE9Cd1CqklP-munsjh-cd7unY88T/view?usp=sharing), upzip this file and move files to the directory:
```backend/checkpoints```
2. Create ChatCPT API và insert this key in file ```backend/app.py```
    - **Access:** [https://platform.openai.com/](https://platform.openai.com/)
    - **Create an account** or **log in**.
    - **Go to API Keys** → create a **new key**. This key will be used to call the API.
3. pip install -r requirements.txt
4. uvicorn backend.app:app --reload
