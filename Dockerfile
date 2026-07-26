FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for C++ extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir pybind11 streamlit fastapi uvicorn pydantic

# Build C++ bindings
RUN cd cpp && mkdir -p build && cd build && cmake .. && make && cd ../..

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "odx.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
