# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
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
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

@ray.remote(num_cpus=1)
class CPUSender:

    def __init__(
        self,
    ) -> None:
        logger.info("=== CPUSender Ray Actor Initialized ===")
        logger.info("Actor PID: %d", os.getpid())
        logger.debug("CPUSender ready to send tensors")

    @ray.method(tensor_transport="gloo")
    def send(self, data):
        source_spec = data[0]
        cpu_tensor = data[1]
        logger.debug(f"Sending tensor for request {source_spec.request_id} with spec {source_spec} and shape {cpu_tensor.shape}")

        return data

    def close(self) -> None:
        logger.info("CPUSender closing...")
        logger.info("CPUSender closed successfully")


@ray.remote(num_cpus=1)
class CPUReceiver:

    def __init__(
        self,
    ) -> None:
        logger.info("=== CPUReceiver Ray Actor Initialized ===")
        logger.info("Actor PID: %d", os.getpid())
        self._tensor_lock = threading.Lock()
        self._received_tensors: dict[str, tuple[SourceSpec, torch.Tensor]] = {}
        logger.debug("CPUReceiver ready to receive tensors")

    def recv(self, data):
        source_spec = data[0]
        cpu_tensor = data[1]
        logger.debug(f"Received tensor for request {source_spec.request_id} with spec {source_spec}")

        assert source_spec is not None, "Source spec is None"
        assert source_spec.request_id not in self._received_tensors, f"Request {source_spec.request_id} already received"
        with self._tensor_lock:
            self._received_tensors[source_spec.request_id] = (source_spec, cpu_tensor)
        
        logger.debug("✓ Tensor stored for request %s", source_spec.request_id)
    
    def get_finished(self, clear=False) -> list[tuple[SourceSpec, torch.Tensor]]:
        with self._tensor_lock:
            ret = [(source_spec, cpu_tensor)
                   for source_spec, cpu_tensor in self._received_tensors.values()]
            if clear:
                self._received_tensors.clear()
            return ret

    def close(self):
        logger.info("CPUReceiver closing...")
        with self._tensor_lock:
            logger.info("CPUReceiver had %d received tensors", len(self._received_tensors))
            self._received_tensors.clear()
        logger.info("CPUReceiver closed successfully")

class RaySendTaskManager(KVSenderInterface):
    """RaySendTaskManager is an implementation of KVSenderInterface that provides a
    ring buffer allocator for managing pin memory allocation and deallocation,
    with Ray for sending data.
    """

    def __init__(self, buffer_size: int) -> None:
        logger.info("=== RaySendTaskManager Initialization ===")
        logger.info("Buffer size: %.2f GB (%d bytes)", buffer_size / (1 << 30), buffer_size)
        super().__init__()
        self._buffer_size = buffer_size
        self._allocator = RingBufferAllocator(self._buffer_size)
        logger.info("✓ RaySendTaskManager initialized successfully")

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
        logger.debug("Creating send task for request: %s", source_spec.request_id)
        
        # Allocate a buffer for the send task
        size = source_spec.get_size()
        logger.debug("Allocating buffer of size: %.2f MB (%d bytes)", size / (1 << 20), size)
        
        address, buffer = self._allocator.allocate(size)
        retry_count = 0
        while address == -1:
            # If allocation fails, wait for a while to process
            # and try again
            retry_count += 1
            if retry_count % 1000 == 0:  # Log every 1000 retries
                logger.debug("Buffer allocation retry %d for request %s", retry_count, source_spec.request_id)
            time.sleep(0.001)
            self.progress()
            address, buffer = self._allocator.allocate(size)
        assert buffer is not None, "Buffer allocation failed"
        
        if retry_count > 0:
            logger.debug("✓ Buffer allocated after %d retries for request %s", retry_count, source_spec.request_id)
        else:
            logger.debug("✓ Buffer allocated immediately for request %s", source_spec.request_id)

        task = RaySendTask(buffer_vaddr=address,
                           sender_actor=sender_actor,
                           receiver_actor=receiver_actor,
                           request_uuid=source_spec.request_id)
        
        # Set required attributes for compatibility with base class
        task.tensor = buffer
        task.buffer = buffer
        task.source_spec = source_spec

        self.add_send_task(task)
        logger.debug("✓ Send task created and added for request %s", source_spec.request_id)
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
    """RaySendTask is a send task that uses CPU memory for the buffer and
    Ray for sending.
    """
    buffer_vaddr: int
    sender_actor: CPUSender
    receiver_actor: CPUReceiver
    request_uuid: str
    
    # Optional fields with default values (must come after non-default fields)
    ref: Optional[ray.ObjectRef] = None
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


class RayDecodeManager:

    def __init__(self, buffer_size: int, host: str, port: int) -> None:
        self._buffer_size = buffer_size
        self._receiver_actor = ray.get_actor(f"receiver_actor_{host}_{port}")

        # How many tokens are received for each request, each layer
        # (p_request_id, layer_id) -> num_tokens
        self._received_tokens: dict[str, dict[int, int]] = {}

        # How many tokens are expected for each request
        # p_request_id -> num_tokens
        self._expected_tokens: dict[str, int] = {}

        # The detailed specs of the requests
        # (p_request_id, layer_id) -> (SourceSpec, vaddr)
        self._request_specs: dict[tuple[str, int], list[tuple[SourceSpec,
                                                              int]]] = {}

        # Metadata
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # Multi process receiving check
        # p_request_id -> number of ready workers
        self._done_receiving_count: defaultdict[str,
                                                int] = defaultdict(lambda: 0)

        # Already 'ready' request, we don't want to check and return it
        # again.
        self._already_ready_requests: set[str] = set()

    def _check_receive_and_update(self):
        """Checks the KV cache receiving status and update the internal
        states
        """
        finished_list = self._receiver_actor.get_finished(clear=True)
        for source_spec, vaddr in finished_list:
            # Get the request id and layer id
            p_request_id = source_spec.request_id
            layer_id = source_spec.layer_id
            num_received_tokens = source_spec.stop - source_spec.start

            if p_request_id not in self._expected_tokens:
                self._expected_tokens[
                    p_request_id] = source_spec.num_all_tokens

            # Update the received tokens
            if p_request_id not in self._received_tokens:
                self._received_tokens[p_request_id] = {}
            if layer_id not in self._received_tokens[p_request_id]:
                self._received_tokens[p_request_id][layer_id] = 0
            self._received_tokens[p_request_id][
                layer_id] += num_received_tokens

            # Update received specs
            if (p_request_id, layer_id) not in self._request_specs:
                self._request_specs[(p_request_id, layer_id)] = []
            self._request_specs[(p_request_id, layer_id)].append(
                (source_spec, vaddr))

    def progress(self) -> None:
        """Process the received requests and the data. Updates the internal
        status and respond to the allocation requests.
        """
        logger.debug("RayDecodeManager progress: do nothing")
        pass
        # self._nixl_receiver.progress()

    def get_finished(self, num_expected_layers: int) -> list[str]:
        """Get the prefill node request_ids of the requests that finishes 
        receiving (which means the KV caches of all tokens and all layers 
        are in CPU memory).

        By default, if a request's id will only be returned once. However,
        the caller can call `remove_ready_request` to force the get_finished
        to return the request id again in the next call.

        Returns:
            list[str]: A list of prefill-side request ids.
        """
        ready_requests = []
        self._check_receive_and_update()
        for p_request_id in self._expected_tokens:
            if p_request_id in self._already_ready_requests:
                # Already checked and ready, skip it
                continue

            expected_tokens = self._expected_tokens[p_request_id]
            assert p_request_id in self._received_tokens
            # check if all the layers are there
            if len(self._received_tokens[p_request_id]) != num_expected_layers:
                continue
            # check if all the tokens are there
            ready = True
            for layer_id in self._received_tokens[p_request_id]:
                received_tokens = self._received_tokens[p_request_id][layer_id]
                if received_tokens != expected_tokens:
                    ready = False
                    break
            if ready:
                ready_requests.append(p_request_id)
                self._already_ready_requests.add(p_request_id)

        if self.world_size == 1:
            return ready_requests

        # For multi-process
        if self.rank == 0:
            for p_request_id in ready_requests:
                self._done_receiving_count[p_request_id] += 1

            other_ranks_finished_ids: list[str] = []
            for i in range(1, self.world_size):
                other_ranks_finished_ids.extend(
                    self.tp_group.recv_object(src=i))
            for p_request_id in other_ranks_finished_ids:
                self._done_receiving_count[p_request_id] += 1

            all_done_recving: list[str] = []
            for p_request_id in self._done_receiving_count:
                if self._done_receiving_count[p_request_id] == \
                        self.world_size:
                    all_done_recving.append(p_request_id)

            # Clear the done receiving count for the requests that are done
            for p_request_id in all_done_recving:
                self._done_receiving_count.pop(p_request_id)
            return all_done_recving
        else:
            self.tp_group.send_object(ready_requests, dst=0)
            return ready_requests

    def remove_ready_request(self, p_request_id: str) -> None:
        """Remove the request from the 'ready' request list so that
        it will be checked again in the next of get_finished.

        Args:
            p_request_id (str): The prefill-side request id.
        """
        self._already_ready_requests.discard(p_request_id)

    def _create_decoder_kv_spec(self, source_spec: SourceSpec,
                                vaddr: int) -> DecoderKVSpec:
        """Create a DecoderKVSpec from the source spec and the virtual address.
        """
        # Get the correct buffer
        return DecoderKVSpec(start=source_spec.start,
                             stop=source_spec.stop,
                             buffer=self._allocator.view_as_tensor(
                                 vaddr, source_spec.dtype,
                                 source_spec.tensor_shape))

    def get_kv_specs(self, p_request_id: str,
                     layer_id: int) -> list[DecoderKVSpec]:
        """Get the KV specs for the given request id and layer id, which 
        will be used for connector to load the KV back to CPU

        Args:
            p_request_id (str): The original request id from prefiller.
            layer_id (int): The layer id of the request.
        """
        ret: list[DecoderKVSpec] = []
        if (p_request_id, layer_id) not in self._request_specs:
            logger.warning("Request %s not found in request specs",
                           (p_request_id, layer_id))
            return ret

        for source_spec, vaddr in self._request_specs[(p_request_id,
                                                       layer_id)]:
            # Create the decoder kv spec
            decoder_kv_spec = self._create_decoder_kv_spec(source_spec, vaddr)
            ret.append(decoder_kv_spec)

        return ret

    def free_request(self, p_request_id):
        """Free the request's memory with the given request id.

        Args:
            p_request_id (str): The original request id from prefiller.
        """
        # Free the memory and clear the internal states
        self._expected_tokens.pop(p_request_id, None)
        rcv_tokens = self._received_tokens.pop(p_request_id, None)
        if rcv_tokens is not None:
            for layer_id in rcv_tokens:
                assert (p_request_id, layer_id) in self._request_specs, \
                    "Found received tokens but no request specs"

                # Free the memory
                for src_spec, vaddr in self._request_specs[(p_request_id,
                                                            layer_id)]:
                    self._allocator.free(vaddr)

                # Clear the request specs
                self._request_specs.pop((p_request_id, layer_id), None)

        else:
            logger.warning("Request %s not found in received tokens",
                           p_request_id)

        self.remove_ready_request(p_request_id)

    def close(self):
        self._nixl_receiver.close()
