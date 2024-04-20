import argparse
import math
import os.path
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from omegaconf import OmegaConf
import omegaconf
from icecream import ic
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
)
from tqdm import tqdm

from dataset import *
from engine import *
from util import load_rttm, rttm_to_annotation

DEFAULT_CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


class BenchmarkTypes(Enum):
    ACCURACY = "ACCURACY"
    CPU = "CPU"
    MEMORY = "MEMORY"


def _engine_params_parser(conf: omegaconf.DictConfig) -> Dict[str, Any]:
    kwargs_engine = {}
    engine = Engines(conf.engine)
    if engine is Engines.PICOVOICE_FALCON:
        if conf.picovoice.access_key is None:
            raise ValueError(f"Engine {conf.engine} requires --picovoice-access-key")
        kwargs_engine.update(access_key=conf.picovoice.access_key)
    elif engine is Engines.PYANNOTE:
        if conf.pyannote.auth_token is None:
            raise ValueError(f"Engine {conf.engine} requires --pyannote-auth-token")
        kwargs_engine.update(auth_token=conf.pyannote.auth_token)
    elif engine is Engines.AWS_TRANSCRIBE:
        if conf.aws.profile is None:
            raise ValueError(f"Engine {conf.engine} requires --aws-profile")
        os.environ["AWS_PROFILE"] = conf.aws.profile
        if conf.aws.s3_bucket_name is None:
            raise ValueError(f"Engine {conf.engine} requires --aws-s3-bucket-name")
        kwargs_engine.update(bucket_name=conf.aws.s3_bucket_name)
    elif engine in [Engines.GOOGLE_SPEECH_TO_TEXT, Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED]:
        if conf.gcp.credentials is None:
            raise ValueError(f"Engine {conf.engine} requires --gcp-credentials")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = conf.gcp.credentials
        if conf.gcp.bucket_name is None:
            raise ValueError(f"Engine {conf.engine} requires --gcp-bucket-name")
        kwargs_engine.update(bucket_name=conf.gcp.bucket_name)
    elif engine is Engines.AZURE_SPEECH_TO_TEXT:
        if conf.azure.storage_account_name is None:
            raise ValueError(f"Engine {conf.engine} requires --azure-storage-account-name")
        if conf.azure.storage_account_key is None:
            raise ValueError(f"Engine {conf.engine} requires --azure-storage-account-key")
        if conf.azure.storage_container_name is None:
            raise ValueError(f"Engine {conf.engine} requires --azure-storage-container-name")
        if conf.azure.subscription_key is None:
            raise ValueError(f"Engine {conf.engine} requires --azure-subscription-key")
        if conf.azure.region is None:
            raise ValueError(f"Engine {conf.engine} requires --azure-region")

        kwargs_engine.update(
            storage_account_name=conf.azure.storage_account_name,
            storage_account_key=conf.azure.storage_account_key,
            storage_container_name=conf.azure.storage_container_name,
            subscription_key=conf.azure.subscription_key,
            region=conf.azure.region)

    return kwargs_engine


def _process_accuracy(engine: Engine, dataset: Dataset, verbose: bool = False) -> None:
    metric_der = DiarizationErrorRate(detailed=True, skip_overlap=True)
    metric_jer = JaccardErrorRate(detailed=True, skip_overlap=True)
    ic(metric_der)
    ic(metric_jer)
    metrics = [metric_der, metric_jer]

    cache_folder = os.path.join(DEFAULT_CACHE_FOLDER, str(dataset), str(engine))
    print(f"Cache folder: {cache_folder}")
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_FOLDER, str(dataset)), exist_ok=True)
    try:
        for index in tqdm(range(dataset.size)):
            audio_path, audio_length, ground_truth = dataset.get(index)
            if verbose:
                print(f"Processing {audio_path}...")

            cache_path = os.path.join(cache_folder, f"{os.path.basename(audio_path)}_cached.rttm")

            if os.path.exists(cache_path):
                hypothesis = rttm_to_annotation(load_rttm(cache_path))
            else:
                hypothesis = engine.diarization(audio_path)

                with open(cache_path, "w") as f:
                    f.write(hypothesis.to_rttm())

            for metric in metrics:
                res = metric(ground_truth, hypothesis, detailed=True)
                if verbose:
                    print(f"{metric.name}: {res}")
    except KeyboardInterrupt:
        print("Stopping benchmark...")

    results = dict()
    for metric in metrics:
        ic(metric)
        results[metric.name] = abs(metric)
    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    results_details_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}.log")
    with open(results_details_path, "w") as f:
        for metric in metrics:
            f.write(f"{metric.name}:\n{str(metric)}")
            f.write("\n")


WorkerResult = namedtuple(
    'WorkerResult',
    [
        'total_audio_sec',
        'process_time_sec',
    ])


def _process_worker(
        engine_type: str,
        engine_params: Dict[str, Any],
        samples: Sequence[Sample]) -> WorkerResult:
    engine = Engine.create(Engines(engine_type), **engine_params)
    total_audio_sec = 0
    process_time = 0

    for sample in samples:
        audio_path, _, audio_length = sample
        total_audio_sec += audio_length
        tic = perf_counter()
        _ = engine.diarization(audio_path)
        toc = perf_counter()
        process_time += (toc - tic)

    engine.cleanup()
    return WorkerResult(total_audio_sec, process_time)


def _process_pool(
        engine: str,
        engine_params: Dict[str, Any],
        dataset: Dataset,
        num_samples: Optional[int] = None) -> None:
    num_workers = os.cpu_count()

    samples = list(dataset.samples[:])
    if num_samples is not None:
        samples = samples[:num_samples]

    chunk_size = math.floor(len(samples) / num_workers)
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            chunk = samples[i * chunk_size: (i + 1) * chunk_size]
            future = executor.submit(
                _process_worker,
                engine_type=engine,
                engine_params=engine_params,
                samples=chunk)
            futures.append(future)

    res = [f.result() for f in futures]
    total_audio_time_sec = sum([r.total_audio_sec for r in res])
    total_process_time_sec = sum([r.process_time_sec for r in res])

    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}_cpu.json")
    results = {
        "total_audio_time_sec": total_audio_time_sec,
        "total_process_time_sec": total_process_time_sec,
        "num_workers": num_workers,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    conf = OmegaConf.load(os.path.join(os.curdir, "src/config/app.env.yaml"))
    print(conf)

    # exit()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    # parser.add_argument("--data-folder", required=True)
    # parser.add_argument("--label-folder", required=True)
    # parser.add_argument("--engine", choices=[en.value for en in Engines], required=True)
    # parser.add_argument("--verbose", action="store_true")
    # parser.add_argument("--aws-profile")
    # parser.add_argument("--aws-s3-bucket-name")
    # parser.add_argument("--azure-region")
    # parser.add_argument("--azure-storage-account-key")
    # parser.add_argument("--azure-storage-account-name")
    # parser.add_argument("--azure-storage-container-name")
    # parser.add_argument("--azure-subscription-key")
    # parser.add_argument("--gcp-bucket-name")
    # parser.add_argument("--gcp-credentials")
    # parser.add_argument("--picovoice-access-key")
    # parser.add_argument("--pyannote-auth-token")
    # parser.add_argument("--type", choices=[bt.value for bt in BenchmarkTypes], required=True)
    # parser.add_argument("--num-samples", type=int)
    # args = parser.parse_args()

    engine_args = _engine_params_parser(conf)

    dataset = Dataset.create(Datasets(conf.dataset), data_folder=conf.data_folder, label_folder=conf.label_folder)
    ic(f"Dataset: {dataset}")

    engine = Engine.create(Engines(conf.engine), **engine_args)
    ic(f"Engine: {engine}")

    if conf.type == BenchmarkTypes.ACCURACY.value:
        _process_accuracy(engine, dataset, verbose=conf.verbose)
    elif conf.type == BenchmarkTypes.CPU.value:
        if not engine.is_offline():
            raise ValueError("CPU benchmark is only supported for offline engines")
        _process_pool(
            engine=conf.engine,
            engine_params=engine_args,
            dataset=dataset,
            num_samples=conf.num_samples)
    elif conf.type == BenchmarkTypes.MEMORY.value:
        if not engine.is_offline():
            raise ValueError("Memory benchmark is only supported for offline engines")
        print("Please make sure the `mem_monitor.py` script is running and then press enter to continue...")
        input()
        _process_pool(
            engine=conf.engine,
            engine_params=engine_args,
            dataset=dataset,
            num_samples=conf.num_samples)


if __name__ == "__main__":
    main()

__all__ = [
    "RESULTS_FOLDER",
]
