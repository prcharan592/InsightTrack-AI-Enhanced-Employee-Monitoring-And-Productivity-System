
title: "InsightTrack - AI-Enhanced Employee Monitoring And Productivity System"
author: "Your Name"
date: "`r Sys.Date()`"
output: html_document
---

# InsightTrack - AI-Enhanced Employee Monitoring And Productivity System

AI enhances workplace performance by enabling data-driven decisions for monitoring activities, detecting inefficiencies, and optimizing resources. It helps organizations boost productivity, improve space utilization, and drive operational efficiency.

## Project Structure

```{bash, eval=FALSE}
project-folder/
│
├── App.py
├── last1.pt                # Your trained YOLOv8 model weights
├── .env                    # Environment variables (API keys)
└── requirements.txt        # Python dependencies

python -m venv venv
source venv/bin/activate



GOOGLE_API_KEY=your_google_gemini_api_key_here

opencv-python
numpy
Pillow
python-dotenv
google-generativeai
ultralytics

streamlit run App.py
