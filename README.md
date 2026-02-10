# Blog Writing Agent

An AI-powered application that researches, plans, and writes technical blog posts.

## Features
- **Research**: Autonomously gathers information from the web.
- **Planning**: Creates structured outlines based on best practices.
- **Writing**: Generates content section-by-section.
- **Images**: Automatically proposes and generates relevant diagrams/images.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (create a `.env` file):
   ```bash
   GOOGLE_API_KEY=...
   TAVILY_API_KEY=...
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
