# Stage 1: Download dependencies
FROM python:3.12 as downloader

WORKDIR /app
COPY requirements.txt .

# Download wheels without installing
RUN pip download --timeout 600 --retries 10 -d /wheels -r requirements.txt

# Stage 2: Install from local wheels
FROM python:3.12

WORKDIR /app

# Copy downloaded wheels
COPY --from=downloader /wheels /wheels
COPY requirements.txt .

# Install from local wheels (much faster and more reliable)
RUN pip install --no-cache-dir --find-links /wheels -r requirements.txt

# Clean up wheels to reduce image size
RUN rm -rf /wheels

# Copy the rest of the application code
COPY . .

EXPOSE 1256
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1256"]