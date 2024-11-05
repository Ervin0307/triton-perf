import copy
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from transformers.utils import check_min_version
from optimum.habana.utils import check_habana_frameworks_min_version, check_optimum_habana_min_version, set_seed
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import soundfile

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)

def read_audio(audio_file_path):
    audio_array, sample_rate = soundfile.read(audio_file_path)
    return audio_array, sample_rate

def override_print(enable):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def override_logger(logger, enable):
    logger_info = logger.info

    def info(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            logger_info(*args, **kwargs)

    logger.info = info


def count_hpu_graphs():
    return len(glob.glob(".graph_dumps/*PreGraph*"))


def override_prints(enable, logger):
    override_print(enable)
    override_logger(logger, enable)


def setup_distributed(args):
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "0"))
    args.global_rank = int(os.getenv("RANK", "0"))


def setup_quantization(model):
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const
    from habana_frameworks.torch.hpu import hpu

    print("Initializing inference with quantization")
    _mark_params_as_const(model)
    _check_params_as_const(model)

    hpu.enable_quantization()
    htcore.hpu_initialize(model)
    return model


def setup_env(args):
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.34.0")
    check_optimum_habana_min_version("1.13.0.dev0")

    if args.global_rank == 0:
        os.environ.setdefault("GRAPH_VISUALIZATION", "true")
        shutil.rmtree(".graph_dumps", ignore_errors=True)

    if args.world_size > 0:
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    # Tweak generation so that it runs faster on Gaudi
    adapt_transformers_to_gaudi()


def setup_device(args):
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore
    return torch.device(args.device)

def setup_distributed_model(args, model_dtype, model_kwargs, logger):
    """
    TO BE IMPLEMENTED
    """
    raise Exception("Distributed model using Deepspeed yet to be implemented")
    return 

def setup_model(args, model_dtype, model_kwargs, logger):
    logger.info("Single-device run.")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs) #
    if args.use_hpu_graphs:
        model = wrap_in_hpu_graph(model.eval())
    return model

def setup_model_pipe(args, model_dtype, model_kwargs, logger):
    logger.info("Single-device run.")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name_or_path, torch_dtype=model_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
    )
    if args.use_hpu_graphs: 
        model = wrap_in_hpu_graph(model.eval())
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=model_dtype,
        device=args.device,
    )
    #generate_kwargs={"language": "hindi", "task": "translate"},
    #        **model_kwargs
    return pipe

def setup_processor(args, model_kwargs, logger):
    logger.info("Single-device run.")
    print(args.model_name_or_path)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path) #, model_kwargs, logger)
    return processor

def initialize_model(args, logger):
    init_start = time.perf_counter()
    setup_distributed(args)
    override_prints(args.global_rank == 0 or args.verbose_workers, logger)
    setup_env(args)
    setup_device(args)
    set_seed(args.seed)
    get_repo_root(args.model_name_or_path, local_rank=args.local_rank, token=args.token)
    use_deepspeed = args.world_size > 0

    if use_deepspeed or args.bf16 or args.fp8:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False

    model_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
    }

    model_pipe = setup_model_pipe(args, model_dtype, model_kwargs, logger)

    init_end = time.perf_counter()
    logger.info(f"Args: {args}")
    logger.info(f"device: {args.device}, n_hpu: {args.world_size}, bf16: {model_dtype == torch.bfloat16}")
    logger.info(f"Model initialization took {(init_end - init_start):.3f}s")
    
    return model_pipe