from __future__ import annotations

import base64
import logging
import json
from datetime import datetime, timedelta
from typing import Tuple, Any, Dict

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)
from openai import AzureOpenAI

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, ImageAnalysisResult

logger = logging.getLogger(__name__)


class ImageAnalyzerAgent:

    def __init__(self) -> None:
        # ---- Azure Blob client ----
        self._blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_BLOB_CONNECTION_STRING
        )
        # Container is assumed to already exist (NO create_container here)
        self._container_client = self._blob_service_client.get_container_client(
            settings.AZURE_CONTAINER_NAME
        )

        # ---- Azure OpenAI multimodal client ----
        # Using the pattern you provided with AzureOpenAI().chat.completions.create
        self._client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )

        # System prompt: how the model should analyze the architecture diagram
        self._system_prompt = """
        You are an Enterprise Architecture Diagram Analysis Agent operating in a regulated BFSI environment.

        You will be given an architecture diagram image via image_url. Analyze this image thoroughly, based only on the content visible in the diagram.

        Focus on:
        - Traffic flow, entry points, and channels.
        - Cloud and hosting details (on-prem, cloud provider, regions).
        - Users and exposure (internal/external, VPN, internet, etc.).
        - Network and security structure (firewalls, DMZ, load balancers, etc.).
        - DC/DR and resilience.
        - Data stores and integrations.
        - Risks and notable observations.

        **IMPORTANT: Your output must be strictly valid JSON, parseable by a standard JSON parser.**
        Do NOT include any text outside the JSON. Do not add explanations, comments, or Markdown.

        The JSON must have this structure:

        {
            "Image_Summary": "<A detailed, natural-language summary of the architecture diagram>",
            "image_components_json": {
                "<Component Name>": "yes",
                "<Another Component>": "yes",
                ...
            }
        }

        - If a component is not present, do not include it in the JSON.
        - If some information is not visible in the diagram, indicate it clearly in Image_Summary, but do not add anything outside the JSON.
        - Do not return Markdown, code blocks, or any additional text.

        Example output:

        {
            "Image_Summary": "The diagram shows a web application hosted on AWS, with a DMZ containing a WAF and load balancer. Traffic enters from the internet and internal users access via VPN...",
            "image_components_json": {
                "AWS": "yes",
                "WAF": "yes",
                "Load Balancer": "yes"
            }
        }
        """.strip()

    async def run(self, ctx: ReviewSessionContext) -> ImageAnalysisResult:

        logger.info(
            f"[{ctx.review_id}] ImageAnalyzerAgent started | "
            f"metadata_type={type(ctx.metadata).__name__}"
        )

        if not isinstance(ctx.metadata, dict):
            logger.error(f"[{ctx.review_id}] metadata is not a dict")
            result = ImageAnalysisResult(
                architecture_summary="No architecture diagram could be analyzed because metadata was not a valid JSON object."
            )
            ctx.image_analysis = result
            return result

        arch_img_url = ctx.metadata.get("arch_img_url")
        if not arch_img_url:
            logger.error(f"[{ctx.review_id}] arch_img_url missing in metadata")
            result = ImageAnalysisResult(
                architecture_summary="No architecture diagram was provided (arch_img_url missing in metadata)."
            )
            ctx.image_analysis = result
            return result

        # 1) Parse data URL → bytes
        try:
            image_bytes, content_type, ext = self._parse_data_url_to_bytes(arch_img_url)
            logger.info(
                f"[{ctx.review_id}] Parsed arch_img_url | content_type={content_type} | "
                f"ext={ext} | bytes_len={len(image_bytes)}"
            )
        except ValueError as e:
            logger.error(
                f"[{ctx.review_id}] Failed to parse base64 image from arch_img_url: {e}"
            )
            result = ImageAnalysisResult(
                architecture_summary="The provided architecture image could not be decoded."
            )
            ctx.image_analysis = result
            return result

        # 2) Upload image and get SAS URL
        try:
            image_url, blob_name = self._upload_image_and_get_sas_url(
                review_id=ctx.review_id,
                image_bytes=image_bytes,
                content_type=content_type,
                ext=ext,
            )
            logger.info(
                f"[{ctx.review_id}] Blob upload successful | image_url_prefix={image_url[:100]}"
            )
        except Exception as e:
            logger.error(
                f"[{ctx.review_id}] Failed to upload image to Azure Blob: {e}"
            )
            result = ImageAnalysisResult(
                architecture_summary="Architecture image upload failed; analysis could not proceed."
            )
            ctx.image_analysis = result
            return result

        # 3) Call LLM to analyze the image and get a detailed text review
        logger.info(f"[{ctx.review_id}] Calling Azure OpenAI multimodal for image analysis")
        review_text = await self._analyze_image_with_llm(
            image_url=image_url,
            review_id=ctx.review_id,
        )

        logger.info(
            f"[{ctx.review_id}] LLM analysis complete | length={len(review_text)}"
        )

        # 4) Build final ImageAnalysisResult – just the text review
        try:
            parsed = json.loads(review_text)
            
            image_summary = parsed.get("Image_Summary") or ""
            components_json = parsed.get("image_components_json") or {}

            result = ImageAnalysisResult(
                architecture_summary=image_summary,
                image_components_json=components_json
            )

        except Exception as e:
            logger.error(f"[{ctx.review_id}] Failed to parse LLM JSON output: {e}")
            
            # fallback – store raw text only
            result = ImageAnalysisResult(
                architecture_summary=review_text,
                image_components_json={}
            )
        
        ctx.image_analysis = result
        
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            logger.info(f"[{ctx.review_id}] Blob deleted immediately after analysis: {blob_name}")
        except Exception as e:
            logger.error(f"[{ctx.review_id}] Failed to delete blob: {e}")

        print("                                                                                      ")
        print("-----------------------------------Image Analysis Outupt-----------------------------------")
        print(result)
        print("                                                                                      ")

        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _parse_data_url_to_bytes(self, data_url: str) -> Tuple[bytes, str, str]:

        if not isinstance(data_url, str) or not data_url.strip():
            raise ValueError("arch_img_url is empty or not a string.")

        if data_url.startswith("data:"):
            # Example: data:image/png;base64,iVBORw0KGgo...
            try:
                header, b64_part = data_url.split(",", 1)
            except ValueError:
                raise ValueError("Invalid data URL format; missing comma separator.")

            # header example: "data:image/png;base64"
            if ";base64" not in header:
                raise ValueError("Data URL is not base64-encoded.")

            mime_part = header[len("data:"): header.index(";base64")]
            content_type = mime_part.strip() or "image/png"
            base64_str = b64_part.strip()
        else:
            # Treat as raw base64 with unknown type; default to PNG
            content_type = "image/png"
            base64_str = data_url.strip()

        try:
            image_bytes = base64.b64decode(base64_str, validate=True)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")

        if not image_bytes:
            raise ValueError("Decoded image bytes are empty.")

        # Normalize extension from MIME type
        ext = "png"
        if content_type.lower() in ("image/jpeg", "image/jpg"):
            ext = "jpg"
        elif content_type.lower() == "image/webp":
            ext = "webp"
        # otherwise keep default "png"

        return image_bytes, content_type, ext

    def _upload_image_and_get_sas_url(
        self,
        review_id: str,
        image_bytes: bytes,
        content_type: str,
        ext: str,
    ) -> str:
        """
        Upload image bytes to Azure Blob Storage and return a SAS URL.

        Path pattern inside the container:
            <YYYYMMDD>/<review_id>_<HHMMSS>.<ext>
        """
        now = datetime.utcnow()
        date_folder = now.strftime("%Y%m%d")
        timestamp = now.strftime("%H%M%S")

        blob_name = f"{date_folder}/{review_id}_{timestamp}.{ext}"

        blob_client = self._container_client.get_blob_client(blob_name)

        content_settings = ContentSettings(content_type=content_type)

        logger.debug(
            f"[{review_id}] Uploading architecture image to blob storage | blob_name={blob_name}"
        )

        blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=content_settings,
        )

        # Generate SAS URL
        sas_token = generate_blob_sas(
            account_name=settings.AZURE_ACCOUNT_NAME,
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=settings.AZURE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=now + timedelta(minutes=30),
        )

        image_url = (
            f"https://{settings.AZURE_ACCOUNT_NAME}.blob.core.windows.net/"
            f"{settings.AZURE_CONTAINER_NAME}/{blob_name}?{sas_token}"
        )

        return image_url, blob_name

    async def _analyze_image_with_llm(
        self,
        image_url: str,
        review_id: str,
    ) -> str:
        """
        Call Azure OpenAI multimodal GPT model using an image_url and return
        the natural-language review text.
        """
        combined_text = (
            "Please analyze the attached architecture diagram image from the given URL "
            "and provide a detailed review as per your system instructions."
        )

        image_inputs: list[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        ]

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text},
                    *image_inputs,
                ],
            },
        ]

        logger.debug(
            f"[{review_id}] Sending multimodal request to Azure OpenAI | "
            f"image_url_prefix={image_url[:100]}"
        )

        response = self._client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.2,
        )

        # Extract text content from response
        content = ""
        try:
            msg = response.choices[0].message
            # message.content can be a string or a list of parts
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Concatenate any text parts
                text_parts = [
                    part.get("text", "")
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)
        except Exception as e:
            logger.error(
                f"[{review_id}] Failed to extract text from Azure OpenAI response: {e}"
            )
            content = "The model did not return a readable analysis for the architecture diagram."

        logger.debug(
            f"[{review_id}] Raw LLM text snippet:\n{content[:1000]}"
        )

        return content.strip()