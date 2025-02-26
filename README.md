
# AI Diagnosis TOOL

A **Powerful AI-Based Diagnosis Tool** developed for the *Google Girls Hackathon*. The tool allows users to upload medical scan reports, and the AI model predicts possible diseases based on the uploaded data. The system is designed to help healthcare professionals and individuals get preliminary insights into potential medical conditions.


## KEY FEATURES

- **Secure Registration**: Users can securely register via a tokenized system.

- **Report Upload**: Users can upload medical reports and symptom details.

- **AI-Powered Diagnosis**: A Python-based backend with Flask, TensorFlow, and PyTorch processes the reports to generate predictions.

- **History Management**: If the user is registered, their history of report generation is maintained.

- **Graph-Based Model Selection**: The ML system follows a graph-based structure to select appropriate trained models.


## APP WORKFLOW

![alt text](https://github.com/ayushii06/Google_GH_2025/blob/main/App_workflow.png)

## Python Model Workflow
![alt text](https://github.com/ayushii06/Google_GH_2025/blob/main/python_workflow.png)


## TECH STACK

**Client:** React, Vite, TailwindCSS

**Server:** Python (FLASK) , Tensorflow, PyTorch


## Model Selection Logic

The system follows a graph-based structure to determine which AI model to use.

Based on scan type, the model selects an appropriate trained model:

- Chest X-ray → Predicts Pneumonia, COVID-19, or Lung Infections

- Brain MRI → Predicts neurological diseases

- Eye OCT Scan → Predicts retinal disorders
and similary other models will be integrated!

Right Now, due to shortage of time, I have only brain tumor detection , Pneumonia and COVID detection model.
## Installation

1.  Clone the Repository:


```bash
    git clone https://github.com/ayushii06/Google_GH_2025

```

2. Backend Setup

```bash
   cd python
   pip install -r requirements.txt
   python app.py
```
3. Frontend Setup

```bash
    cd client
    npm install
    npm run dev
```
## Future Enhancements

Due to shortage of time, I was not able to implement my full solution.

But here are my Future Enhancements for this project -

- Secured user registration with **MongoDB** and **Firebase**

- Maintaining **History** of registerd User Reports.

- Training more diagnosis based model and improving accuracy. 

- Improving **Symptoms based** diagnosis.
