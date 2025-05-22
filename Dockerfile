# Sử dụng image Python chính thức làm base image
FROM python:3.9-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng vào thư mục làm việc
COPY main.py .

# Expose cổng mà ứng dụng FastAPI sẽ chạy (mặc định là 8000)
EXPOSE 8000

# Lệnh để chạy ứng dụng bằng uvicorn
# --host 0.0.0.0 để ứng dụng có thể truy cập từ bên ngoài container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]