import logging
import os
import random
import math
import re
import shutil
import warnings
import datetime
import time
from collections import defaultdict, deque
from typing import List, Optional, Tuple, Union

from torch.cuda.amp import autocast as autocast
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import Wav2Vec2FeatureExtractor
from omegaconf import OmegaConf

from services.configuration_musilingo import MusiLingoConfig, PATH
import timm.models.hub as timm_hub

from transformers import LlamaTokenizer, Wav2Vec2FeatureExtractor, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, \
    replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import PreTrainedModel


def download_url(
        url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    if _is_remote_location_available():
        _download_file_from_remote_location(fpath, url)
    else:
        # expand redirect chain if needed
        url = _get_redirect_url(url, max_hops=max_redirect_hops)

        # check if file is located on Google Drive
        file_id = _get_google_drive_file_id(url)
        if file_id is not None:
            return download_file_from_google_drive(file_id, root, filename, md5)

        # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            _urlretrieve(url, fpath)
        except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
                _urlretrieve(url, fpath)
            else:
                raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
            self,
            epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
            self,
            epoch,
            start_iters,
            iters_per_inner_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
            self,
            epoch,
            iters_per_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            start_iters=None,
            log_freq=50,
            cuda_enabled=False,
            accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if necessary.
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative.
        """
        anns = self.config.build_info.annotations

        splits = anns.keys()

        cache_root = registry.get_path("cache_root")

        for split in splits:
            info = anns[split]

            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(urls, str):
                urls = [urls]
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                # if storage_path is relative, make it full by prefixing with cache_root.
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                dirname = os.path.dirname(storage_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if os.path.isfile(url_or_filename):
                    src, dst = url_or_filename, storage_path
                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logging.info("Using existing file {}.".format(dst))
                else:
                    if os.path.isdir(storage_path):
                        # if only dirname is provided, suffix with basename of URL.
                        raise ValueError(
                            "Expecting storage_path to be a file path, got directory {}".format(
                                storage_path
                            )
                        )
                    else:
                        filename = os.path.basename(storage_path)

                    download_url(url=url_or_filename, root=dirname, filename=filename)

    def _download_vis(self):

        storage_path = self.config.build_info.get(self.data_type).storage
        storage_path = get_cache_path(storage_path)

        if not os.path.exists(storage_path):
            warnings.warn(
                f"""
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
            )

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = os.path.join(vis_info.storage, split)

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets


class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the builder will be registered.

        Usage:

            # from lavi.common.registry import registry
            # from lavi.datasets.base_dataset_builder import BaseDatasetBuilder
        """

        def wrap(builder_cls):
            # from musilingo.datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            # from lavi.common.registry import registry
        """

        def wrap(task_cls):
            # from musilingo.tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            # from lavi.common.registry import registry
        """

        def wrap(model_cls):
            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            # from lavi.common.registry import registry
        """

        def wrap(processor_cls):
            # from musilingo.processors import BaseProcessor

            assert issubclass(
                processor_cls, BaseProcessor
            ), "All processors must inherit BaseProcessor class"
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap

    @classmethod
    def register_lr_scheduler(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            # from minigpt4.common.registry import registry
        """

        def wrap(lr_sched_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_sched_cls
            return lr_sched_cls

        return wrap

    @classmethod
    def register_runner(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            # from minigpt4.common.registry import registry
        """

        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap

    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            # from minigpt4.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            # from minigpt4.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    # @classmethod
    # def get_trainer_class(cls, name):
    #     return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
                "writer" in cls.mapping["state"]
                and value == default
                and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            # from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
                model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                    finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            query_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if query_embeds is not None:
            inputs_embeds = torch.cat([query_embeds, inputs_embeds], dim=1)
            batch_size, seq_length, _ = inputs_embeds.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            query_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            query_embeds=query_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, query_embeds=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                query_embeds = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "query_embeds": query_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@registry.register_model("musilingo")
class MusiLingo(BaseModel):
    """
    MERT GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/musilingo.yaml",
    }

    def __init__(
            self,
            mert_model,
            llama_model,
            config,
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.low_resource = low_resource

        print('Loading Audio Encoder')
        self.audio_encoder = AutoModel.from_pretrained(mert_model, trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model, trust_remote_code=True)

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder = self.audio_encoder.eval()

        print('Loading Audio Encoder Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.audio_encoder.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<AudioHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def audioenc_to_cpu(self):
        self.audio_encoder.to("cpu")
        self.audio_encoder.float()

    def encode_audio(self, audio, attn=None):
        device = audio.device
        if self.low_resource:
            self.audioenc_to_cpu()
            audio = audio.to("cpu")

        if attn is None:

            audio_embeds = torch.stack(self.audio_encoder(input_values=audio,
                                                          output_hidden_states=True).hidden_states)  # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3)  # [B, T, 1024]

        else:

            audio_embeds = torch.stack(self.audio_encoder(input_values=audio,
                                                          output_hidden_states=True,
                                                          attention_mask=attn).hidden_states)  # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3)  # [B, T, 1024]

        # Average time steps:
        t = 325
        B, T, D = audio_embeds.shape
        avg_tmp = audio_embeds[:, :T // t * t].reshape(B, T // t, t, D).mean(2)

        # Average the remaining steps
        if T % t > 0:
            avg_last = audio_embeds[:, T // t * t:].reshape(B, 1, T % t, D).mean(2)
            audio_embeds = torch.concat([avg_tmp, avg_last], dim=1)
        else:
            audio_embeds = avg_tmp
        audio_embeds = audio_embeds.to(device)
        inputs_llama = self.llama_proj(audio_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(audio.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before, p_after = prompt.split('<AudioHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = atts_audio[:, :1].expand(-1, wrapped_audio_embeds.shape[1])
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio

    def instruction_prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before = []
            p_after = []

            for i in range(batch_size):
                p_b, p_a = prompt[i].split('<AudioHere>')
                p_before.append(p_b)
                p_after.append(p_a)

            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", padding='longest', add_special_tokens=False).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", padding='longest', add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = torch.cat([p_before_tokens.attention_mask, atts_audio, p_after_tokens.attention_mask],
                                           dim=1)
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio

    def forward(self, samples):
        audio = samples["audio"]
        attn = samples["attention_mask"] if "attention_mask" in samples else None
        audio_embeds, atts_audio = self.encode_audio(audio, attn)

        if 'instruction_input' in samples:  # instruction tuning dataset
            instruction_prompt = []
            for instruction in samples['instruction_input']:
                prompt = '<Audio><AudioHere></Audio> ' + instruction
                instruction_prompt.append(self.prompt_template.format(prompt))
            audio_embeds, atts_audio = self.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)

        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            audio_embeds, atts_audio = self.prompt_wrap(audio_embeds, atts_audio, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(audio.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_audio.shape[0], atts_audio.shape[1] + 1],
                       dtype=torch.long).to(audio.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = audio_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_audio[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, audio_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_audio, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        mert_model = cfg.get("mert_model", "")
        llama_model = cfg.get("llama_model")

        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            mert_model=mert_model,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load ckpt weights of MusiLingo
        if ckpt_path:
            print("Load MERT-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model


class MusilingoModel(PreTrainedModel):
    config_class = MusiLingoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MusiLingo(
            mert_model=config.mert_model,
            llama_model=config.llama_model,
            config=config,
            prompt_path=config.prompt_path,
            prompt_template=config.prompt_template,
            max_txt_len=config.max_txt_len,
            end_sym=config.end_sym,
            low_resource=config.low_resource,
            device_8bit=config.device_8bit
            # self.linear_ckpt_path = config.linear_ckpt_path``
        )

    def forward(self, tensor):
        return self.model.forward(tensor)