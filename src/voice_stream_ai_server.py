import ray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ray import serve

import websockets
import uuid
import json
import asyncio
import logging

from src.audio_utils import save_audio_to_file
from src.client import Client

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)
fastapi_app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.deployment(name="transcription_server")
@serve.ingress(fastapi_app)
class TranscriptionServer:
    """
    Represents the WebSocket server for handling real-time audio transcription.

    This class manages WebSocket connections, processes incoming audio data,
    and interacts with VAD and ASR pipelines for voice activity detection and
    speech recognition.

    Attributes:
        vad_pipeline: An instance of a voice activity detection pipeline.
        asr_pipeline: An instance of an automatic speech recognition pipeline.
        host (str): Host address of the server.
        port (int): Port on which the server listens.
        sampling_rate (int): The sampling rate of audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
        connected_clients (dict): A dictionary mapping client IDs to Client objects.
    """

    def __init__(self, sampling_rate=16000, samples_width=2):

        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.connected_clients = {}

        from src.asr.asr_factory import ASRFactory
        from src.vad.vad_factory import VADFactory

        self.vad_pipeline = VADFactory.create_vad_pipeline("pyannote")
        self.asr_pipeline = ASRFactory.create_asr_pipeline("faster_whisper")

    async def handle_audio(self, client: Client, websocket: WebSocket):
        while True:
            message = await websocket.receive()
            if "bytes" in message.keys():
                client.append_audio_data(message['bytes'])
            # TODO: need to verify this case
            elif "text" in message.keys():
                config = json.loads(message)
                if config.get('type') == 'config':
                    client.update_config(config['data'])
                    continue
            else:
                logger.error(
                    f"{type(message)} is not a valid message type")
                logger.error(
                    f"Unexpected message type from {client.client_id}")

            client.process_audio(
                websocket, self.vad_pipeline, self.asr_pipeline)

    @fastapi_app.websocket("/")
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()

        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        logger.info(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        except WebSocketDisconnect as e:
            logger.warn(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]


app = TranscriptionServer.bind()
