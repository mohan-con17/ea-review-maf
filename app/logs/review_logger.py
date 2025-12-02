import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from app.config.settings import settings

class AzureBlobLogger:
    def __init__(self) -> None:
        self._blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_BLOB_CONNECTION_STRING
        )
        self._container_client = self._blob_service_client.get_container_client(
            settings.AZURE_LOGS_CONTAINER_NAME
        )

        try:
            self.container_client.create_container()
        except Exception:
            pass

    def log(self, review_id: str, data: dict):
        now = datetime.utcnow()
        month_folder = now.strftime("%b %Y")      # "Nov 2025"
        date_folder = now.strftime("%d-%m-%Y")    # "27-11-2025"
        timestamp = now.strftime("%H-%M-%S")        # e.g., 15-42-30

        # Full "path" inside blob container
        blob_name = f"{month_folder}/{date_folder}/{review_id}_{timestamp}.json"
        blob_client = self._container_client.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(data, default=str), overwrite=True)
