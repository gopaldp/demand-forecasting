# Step 1: Specify the base image
FROM python:3.9-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Install the missing system dependency for LightGBM
# apt-get update refreshes the package list
# apt-get install -y libgomp1 installs the required library
# The -y flag automatically answers "yes" to any prompts.
RUN apt-get update && apt-get install -y libgomp1

# Step 4: Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code
COPY . .

# Step 6: Expose the port
EXPOSE 8501

# Step 7: Define the startup command
CMD ["streamlit", "run", "app.py"]