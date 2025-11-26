# Używamy lekkiej wersji Pythona
FROM python:3.9-slim

# Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy plik z wymaganiami i instalujemy biblioteki
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę kodu aplikacji do kontenera
COPY . .

CMD ["python", "src/predict.py", "--image_path", "data/sample_cloud.jpg"]