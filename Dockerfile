FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/model.pkl ./artifacts/model.pkl

ENV MODEL_PATH=/var/task/artifacts/model.pkl

CMD ["src.score_api.app"]
