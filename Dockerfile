FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. System packages (Runs as root at the start)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Create a non-root user named "user" with UID 1000 (Required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 3. Set the working directory inside the user's home directory
WORKDIR $HOME/app

# 4. Install backend dependencies (Upgrade pip as root to avoid permission warnings, then switch user context)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip 

# Switch to the non-root user before running user-space pip installs and copying files
USER user

# Install python dependencies to user space
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy backend code and ensure the files are owned by the 'user'
COPY --chown=user:user app ./app
COPY --chown=user:user src ./src
COPY --chown=user:user models ./models

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run your backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]