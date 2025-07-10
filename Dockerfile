# File: Dockerfile

# Use a slim Python image as the base image
FROM python:3.11-slim

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variable for pipenv to create virtual environments in the project directory
# This helps in keeping the dependencies isolated within the project directory as .venv
# instead of creating them in the user's home directory, which is useful for deployment scenarios.
# This also prevents pipenv from creating a virtual environment in the user's home directory,
# a virtualenv in the default system path (e.g. under ~/.local/share/virtualenvs/...)
ENV PIPENV_VENV_IN_PROJECT=1

# Set the working directory
WORKDIR /app   

# Copy the application code
# Note: Ensure that 'app.py' is the main file of your Streamlit application. 
COPY . /app  

# Copy the Pipfile and Pipfile.lock
# COPY Pipfile Pipfile.lock /app/
    
# Install pipenv
RUN pip install --upgrade pip pipenv

# Install the application dependencies using pipenv
RUN pipenv install --deploy --ignore-pipfile  

# Expose the port for the application
EXPOSE 8501 

# Set the entrypoint for the container  
ENTRYPOINT ["pipenv", "run", "streamlit", "run"]

# Run the Streamlit application
# CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["app.py"]
   