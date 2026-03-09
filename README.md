# student-suicide-risk-assessment
A machine learning–based web application that assesses suicide risk among students using well-being survey data and a Random Forest classifier.

Streamlit link: https://studentriskassess.streamlit.app/
Kaggle dataset: https://www.kaggle.com/datasets/samiulalom01/early-suicide-prediction

# IMPORTANT NOTE
‼️ This is a purely academic project and should not be used in an actual medical or other professional setting, nor should it be used to make actual predicitons. ‼️

If you are actively going to harm yourself, **you are in a medical emergency**.  
**PLEASE IMMEDITELY CALL YOUR LOCAL EMERGENCY LINE**.  
Common numbers are listed below, with a full list here: https://en.wikipedia.org/wiki/List_of_emergency_telephone_numbers

| Countries / Regions | Number |
| ----------- | ----------- |
| USA / Canada / Mexico | 911 |
| EU / India / Nigeria / Indonesia | 112 |
| China | 120 |
| Japan / South Korea | 119 |
| Australia | 999 |
| Brazil | 192 |

If you are struggling, considering harming yourself, and/or need someone to talk to, find your country's available suicide and crisis hotlines here: https://www.iasp.info/suicidalthoughts/

You are not alone, and deserve to live.

# Instructions for website use
1. **Navigate to https://studentriskassess.streamlit.app/.**
   - If the application is inactive, simply click the button to wake it up.

2. **Enter the student's survey responses into the form and click "Submit responses".**
   - The application requires that at least 5 questions be answered to calculate a prediction.
   - The accepted age range is 18 – 99.

3. **The application will then return the predicted suicide risk level, the class probabilities, and the three visualizations.**

4. **To submit another questionnaire**, the answers in the form can simply be adjusted, or the **Clear form** button can be clicked to reset the form and page.

# Installation instructions for local machine

1. **Install prerequisites.**
   - Python (version 3.12 recommended)
   - Git (optional if not downloading files via GitHub)

2. **Download the project.**
   - **Option A: Using Git**
     1. Open a terminal (or Command Prompt / PowerShell on Windows).
     2. Run the following:
```bash
        git clone https://github.com/travage/student-suicide-risk-assessment.git
        cd student-suicide-risk-assessment
```
   - **Option B: Via ZIP file**
     1. Go to https://github.com/travage/student-suicide-risk-assessment.
     2. Click **Code** → **Download ZIP**.
     3. Extract the ZIP file.
     4. Open a terminal and `cd` into the extracted `student-suicide-risk-assessment` folder.

3. **Create and activate a virtual environment.**
      - Activate it:
     - **macOS / Linux**
    ```bash
    source .venv/bin/activate
    ```
     - **Windows Command Prompt**
    ```bash
    .venv\Scripts\activate
    ```
    - **Windows PowerShell**
   ```bash
   .venv\Scripts\Activate.ps1
   ```


4. **Install dependencies.**
   - From the project root (`student-suicide-risk-assessment`), run:
```bash
     pip install -r requirements.txt
```

5. **Run the Streamlit app.**
   - From the project root, run:
```bash
     streamlit run app/app.py
```
   - After a few seconds, Streamlit will start a local server and open your browser automatically.
     - If it does not, enter this URL into your web browser: http://localhost:8501.

6. **Enter the student's survey responses into the form and click "Submit responses".**
   - The application requires that at least 5 questions be answered to calculate a prediction.
   - The accepted age range is 18 – 99.

7. **The application will then return the predicted suicide risk level, the class probabilities, and the three visualizations.**

8. **To submit another questionnaire**, the answers in the form can simply be adjusted, or the **Clear form** button can be clicked to reset the form and page.

9. **To stop the app**, go back to your terminal and press:
   - `Ctrl + C`

10. **To deactivate the virtual environment**, enter:
```bash
    deactivate
```


