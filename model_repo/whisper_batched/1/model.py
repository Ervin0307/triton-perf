import json
import logging
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from utils import count_hpu_graphs, initialize_model_n_processor, read_audio
from optimum.habana.utils import get_hpu_memory_stats
import soundfile
from time import time
import os

gen_kwargs = {}

class habana_args:
    device = 'hpu'
    model_name_or_path = "openai/whisper-large-v2"
    audio_file = "en1.wav"
    token = None
    bf16 = True
    use_hpu_graphs = gen_kwargs.get('hpu_graphs', True)
    seed = 42
    batch_size = -1
    model_revision = "main"
    sampling_rate = 16000
    global_rank = 0
    world_size = 1
    local_rank = 0

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        print(f'Initializing')
        self.model, self.processor = initialize_model_n_processor(habana_args, logger)
        self.device = self.model.device
        self.model_dtype = self.model.dtype
        self.sampling_rate = habana_args.sampling_rate
        self.padding_size = 480000
        
        # TEST A SAMPLE DURING INITIALISATION 
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        input_speech_arr, sampling_rate = read_audio(os.path.join(cur_dir, habana_args.audio_file))
        for i in range(1):
            t1 = time()
            out_transcript = self.infer_transcript(input_speech_arr, habana_args.sampling_rate)
            t2 = time()
            print(f"Test inference time:{t2-t1}secs  {out_transcript}")

        print('Initialize finished')
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT configurations
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

    def infer_transcript(self, audio_batch, sampling_rate=16000):
        t1 = time()
        input_features = self.processor(audio_batch, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.model.generate(input_features.to(self.model_dtype), **gen_kwargs)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        t2 = time()
        inference_time = t2 - t1
        print(f"Time for {len(transcription)} samples : {inference_time}secs")
        return transcription, inference_time
    
    def batched_inference(self, requests):
        request_batch = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            request_batch.append(in_0.as_numpy())

        request_batch = np.array(request_batch).squeeze()    
        print(f"xxxxxxxxxxx AUDIO BATCHED INPUT SIZE : {request_batch.shape} INPUT TYPE : {type(request_batch)}")

        out_0, inference_time = self.infer_transcript(request_batch, habana_args.sampling_rate)

        return out_0, inference_time/len(requests)

    def execute(self, requests):
        responses = []
        print(f"NUM REQUESTS {len(requests)}")

        if len(requests) > 1:  # More than 1 requests are received, batch them and infer at once 
            out_0_batched, inference_time = self.batched_inference(requests)
            responses = []
            for i in range(len(requests)):
                # Create OUTPUT tensors
                out_tensor_0 = pb_utils.Tensor("OUTPUT0", np.array(out_0_batched[i], dtype=self.output0_dtype))
                out_tensor_1 = pb_utils.Tensor("OUTPUT1", np.array(inference_time, dtype=self.output1_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
                responses.append(inference_response)
        else:  # Single sample inference
            for request in requests:
                # Get INPUT
                in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
                input_speech_arr = in_0.as_numpy()
                print(f"xxxxxxxxxxx AUDIO INPUT SIZE : {input_speech_arr.shape} INPUT TYPE : {type(input_speech_arr)}")
                
                out_0, inference_time = self.infer_transcript(input_speech_arr, habana_args.sampling_rate)

                # Create OUTPUT tensors
                out_tensor_0 = pb_utils.Tensor("OUTPUT0", np.array(out_0, dtype=self.output0_dtype))
                out_tensor_1 = pb_utils.Tensor("OUTPUT1", np.array(inference_time, dtype=self.output1_dtype))

                # Create InferenceResponse with both outputs
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
                responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")