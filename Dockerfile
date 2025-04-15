FROM python:3.12

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app files
COPY . .

# Command to run the app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--reload"]
