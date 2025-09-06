# Step 1: Specify the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install the system dependency needed by LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Step 4: Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code into the container
COPY . .

# Step 6: Expose the port Streamlit runs on
EXPOSE 8501

# Step 7: Define the command to run when the container starts
CMD ["streamlit", "run", "app.py"]