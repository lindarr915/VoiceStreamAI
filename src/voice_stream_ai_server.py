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
        # self.vad_pipeline = vad_pipeline
        # self.asr_pipeline = asr_pipeline
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.connected_clients = {}

        from src.asr.asr_factory import ASRFactory
        from src.vad.vad_factory import VADFactory

        self.vad_pipeline = VADFactory.create_vad_pipeline("pyannote")
        self.asr_pipeline = ASRFactory.create_asr_pipeline("faster_whisper")


    async def handle_audio(self, client : Client, websocket: WebSocket):
        message = await websocket.receive_text()
        config = json.loads(message)

        if config.get('type') == 'config':
            client.update_config(config['data'])

        while True:
            message = await websocket.receive_bytes()
            client.append_audio_data(message)            # this is synchronous, any async operation is in BufferingStrategy
            client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline)

    @fastapi_app.websocket("/")
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()

        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        print(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        except websockets.ConnectionClosed as e:
            print(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]


app = TranscriptionServer.bind()
