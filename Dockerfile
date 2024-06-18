# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask==2.0.2 tensorflow==2.12.0 werkzeug==2.0.2 flask-cors pillow gradio langchain_google_genai google-generativeai langchain

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]



