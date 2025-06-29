# SPDX-License-Identifier: Apache-2.0
import contextlib
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import ray

from vllm.distributed.kv_transfer.kv_connector.v1.cpu_connector_utils import (
    DecoderKVSpec, DestinationSpec, KVSenderInterface, SendTask, SendTaskState,
    SourceSpec)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.kv_transfer.kv_connector.v1.cpu_connector_utils import RingBufferAllocator
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

@ray.remote(num_gpus=1)
class CPUSender:

    def __init__(
        self,
    ) -> None:
        pass

    @ray.method(tensor_transport="gloo")
    def send(self, data):
        source_spec = data[0]
        cpu_tensor = data[1]
        logger.debug(f"Sending tensor for request {source_spec.request_id} with spec {source_spec} and shape {cpu_tensor.shape}")

        return data

    def close(self) -> None:
        pass


@ray.remote(num_gpus=1)
class CPUReceiver:

    def __init__(
        self,
    ) -> None:
        self._received_tensors: dict[str, tuple[SourceSpec, torch.Tensor]] = {}

    def recv(self, data):
        source_spec = data[0]
        cpu_tensor = data[1]
        logger.debug(f"Received tensor for request {source_spec.request_id} with spec {source_spec}")

        assert source_spec is not None, "Source spec is None"
        assert source_spec.request_id not in self._received_tensors, f"Request {source_spec.request_id} already received"
        self._received_tensors[source_spec.request_id] = (source_spec, cpu_tensor)
    
    def get_finished(self, clear=False) -> list[tuple[SourceSpec, torch.Tensor]]:
        ret = [(source_spec, cpu_tensor)
               for source_spec, cpu_tensor in self._received_tensors.values()]
        if clear:
            self._received_tensors.clear()
        return ret

    def close(self):
        pass

class RaySendTaskManager(KVSenderInterface):
    """RaySendTaskManager is an implementation of KVSenderInterface that provides a
    ring buffer allocator for managing pin memory allocation and deallocation,
    with Ray for sending data.
    """

    def __init__(self, buffer_size: int) -> None:
        super().__init__()
        self._buffer_size = buffer_size
        self._allocator = RingBufferAllocator(self._buffer_size)

    def create_send_task(
        self,
        source_spec: SourceSpec,
        destination_spec: DestinationSpec,
        sender_actor: CPUSender,
        receiver_actor: CPUReceiver,
    ) -> SendTask:
        """Create a non-ready send task with a CPU buffer allocated.

        Args:
            source_spec (SourceSpec): The source specification of the send 
                task.
            destination_spec (DestinationSpec): The destination 
                specification of the send task.
        """
        # Allocate a buffer for the send task
        size = source_spec.get_size()
        address, buffer = self._allocator.allocate(size)
        while address == -1:
            # If allocation fails, wait for a while to process
            # and try again
            time.sleep(0.001)
            self.progress()
            address, buffer = self._allocator.allocate(size)
        assert buffer is not None, "Buffer allocation failed"

        task = RaySendTask(cuda_event=None,
                           sender_actor=sender_actor,
                           receiver_actor=receiver_actor,
                           request_uuid=source_spec.request_id)

        self.add_send_task(task)
        return task

    def free_task(self, task: SendTask) -> None:
        """Free the send task.
        Will be called in the pre-implemented progress() method.

        Args:
            task (SendTask): The send task to be freed.
        """
        assert isinstance(task, RaySendTask), \
            "Task is not a NixlSendTask"
        # Free the buffer in the ring buffer allocator
        self._allocator.free(task.buffer_vaddr)

    def send_task(self, task: SendTask) -> None:
        """Send the send task after it is ready.
        Will be called in the pre-implemented progress() method.

        Args:
            task (SendTask): The send task to be sent.
        """
        assert isinstance(task, RaySendTask), \
            "Task is not a NixlSendTask"
        assert task.ref is not None, "Ref is not set"
        tmp_ref = task.sender_actor.send.remote([task.source_spec, task.buffer])
        task.ref = task.receiver_actor.recv.remote(tmp_ref)
        task.mark_sending()
        return

    def pre_progress_hook(self) -> None:
        for task in self.get_send_tasks():
            task.update_states()

    def post_progress_hook(self) -> None:
        pass

    def wait_for_all_tasks(self) -> None:
        """Wait for all tasks to finish. Mainly for debug, test,
        and offline inferences.
        """
        # Wait for all tasks to finish
        tasks = self.get_send_tasks()
        while tasks:
            self.progress()
            time.sleep(1)
            tasks = self.get_send_tasks()
            logger.info("Still waiting for %d tasks to finish", len(tasks))

    def close(self):
        self.wait_for_all_tasks()
        self._nixl_sender.close()


@dataclass
class RaySendTask(SendTask):
    """NixlSendTask is a send task that uses CPU memory for the buffer and
    Nixl for sending.
    """
    buffer_vaddr: int

    sender_actor: CPUSender
    receiver_actor: CPUReceiver
    ref: Optional[ray.ObjectRef] = None

    request_uuid: str

    # Optional fields that will be updated later
    # Cuda event for h2d copy
    cuda_event: Optional[torch.cuda.Event] = None

    def __post_init__(self) -> None:
        self.creation_time = time.time()

    def update_states(self) -> None:
        """Update the states of the send task.
        """
        # Check the cuda event
        if not self.state.sender_ready and self.cuda_event is not None \
                and self.cuda_event.query():
            self.state.sender_ready = True

        assert self.ref is not None, "Ref is not set"
        if not self.is_done() and ray.wait([self.ref], timeout=0.001, fetch_local=False):
            self.state.send_done = True

        if self.state.is_ready() and not self.state.is_done():
            self.send_task()



