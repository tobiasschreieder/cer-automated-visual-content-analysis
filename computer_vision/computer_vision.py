from preprocessing.data_entry import DataEntry

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from typing import Dict, List
from pathlib import Path
import json
from config import Config


cfg = Config.get()

# Specify Clarifai model
USER_ID = 'clarifai'
APP_ID = 'main'
MODEL_ID = 'general-image-recognition'


def read_pat() -> str:
    """
    Read Clarifai Personal Access Token (PAT); Put pat.txt in working/
    :return: String with PAT
    """
    with cfg.working_dir.joinpath('pat.txt').open(encoding='utf8') as file:
        pat = file.readline()
    return pat


channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + read_pat()),)
userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)


def call_clarifai(image_file_location: Path) -> Dict[str, float]:
    """
    API call for specified Clarifai Model
    :param image_file_location: String with location of image
    :return: Dictionary with Clarifai outputs (label: score)
    """
    with open(image_file_location, "rb") as f:
        file_bytes = f.read()

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    output = post_model_outputs_response.outputs[0]

    results = dict()
    for concept in output.data.concepts:
        results.setdefault(concept.name, concept.value)

    return results


def save_clarifai_output(results: Dict[str, float], image_id: str):
    """
    Save Clarifai output
    :param results: Dictionary with results
    :param image_id: String with image-id for file-name
    """
    file_name = "clarifai_" + image_id + ".json"
    clarifai_path = cfg.working_dir.joinpath("clarifai")
    Path(clarifai_path).mkdir(parents=True, exist_ok=True)

    with open(clarifai_path.joinpath(file_name), 'w') as file:
        json.dump(results, file)


def run_clarifai(image_ids: List[str]):
    """
    Run Clarifai Computer Vision for given image-ids
    :param image_ids: List with image-ids
    """
    for image_id in image_ids:
        image_path = DataEntry.load(image_id=image_id).webp_path
        results = call_clarifai(image_file_location=image_path)
        save_clarifai_output(results=results, image_id=image_id)
