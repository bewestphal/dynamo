# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def get_current_k8s_namespace() -> str:
    """Get the current namespace if running inside a k8s cluster"""
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to 'default' if not running in k8s
        return "default"


class KubernetesAPI:
    def __init__(self, k8s_namespace: Optional[str] = None):
        # Load kubernetes configuration
        try:
            config.load_incluster_config()  # for in-cluster deployment
        except ConfigException:
            config.load_kube_config()  # for out-of-cluster deployment

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()
        self.current_namespace = k8s_namespace or get_current_k8s_namespace()

    def get_worker_namespace_suffix(
        self, graph_deployment_name: str, dynamo_component: str
    ) -> Optional[str]:
        """Return the `nvidia.com/dynamo-worker-hash` label value for the
        worker pods of a given DGD service.

        The Dynamo operator stamps a per-worker-set hash onto worker pods
        (label ``nvidia.com/dynamo-worker-hash``) and exports the same
        value into the worker container as ``DYN_NAMESPACE_WORKER_SUFFIX``.
        The worker uses that suffix when registering event-plane channels
        (e.g. forward-pass-metrics), so a planner that wants to subscribe
        to those channels must rebuild the same suffixed namespace.

        Returns ``None`` when no matching worker pod is found or none
        carries the hash label. If multiple distinct hashes exist (e.g.
        during a rolling worker update) the most-recent pod's hash wins
        and a warning is logged — the typical caller subscribes to one
        suffix at a time today.
        """
        label_selector = (
            f"nvidia.com/dynamo-graph-deployment-name={graph_deployment_name},"
            f"nvidia.com/dynamo-component={dynamo_component}"
        )
        try:
            pods = self.core_api.list_namespaced_pod(
                namespace=self.current_namespace,
                label_selector=label_selector,
            )
        except client.exceptions.ApiException as e:
            logger.warning(
                f"Failed to list pods for worker-hash lookup "
                f"(selector='{label_selector}'): {e}"
            )
            return None

        hashes = set()
        chosen: Optional[str] = None
        latest_ts = None
        for pod in pods.items:
            labels = (pod.metadata.labels or {}) if pod.metadata else {}
            wh = labels.get("nvidia.com/dynamo-worker-hash")
            if not wh:
                continue
            hashes.add(wh)
            ts = pod.metadata.creation_timestamp if pod.metadata else None
            if chosen is None or (ts is not None and ts > latest_ts):
                chosen = wh
                latest_ts = ts
        if len(hashes) > 1:
            logger.warning(
                f"Multiple dynamo-worker-hash values found for "
                f"{graph_deployment_name}/{dynamo_component}: {sorted(hashes)}; "
                f"using most-recent={chosen}. FPM subscribers attached to one "
                f"hash will miss events from the others — typical during "
                f"a rolling worker update."
            )
        return chosen

    def _get_graph_deployment_from_name(self, graph_deployment_name: str) -> dict:
        """Get the graph deployment from the dynamo graph deployment name"""
        return self.custom_api.get_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
            name=graph_deployment_name,
        )

    def list_graph_deployments(self) -> list[dict]:
        """List all DynamoGraphDeployments in the current namespace."""
        result = self.custom_api.list_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
        )
        return result.get("items", [])

    def get_graph_deployment(self, graph_deployment_name: str) -> dict:
        """
        Get the parent DynamoGraphDeployment

        Returns:
            The DynamoGraphDeployment object

        Raises:
            DynamoGraphDeploymentNotFoundError: If the parent graph deployment is not found
        """
        try:
            return self._get_graph_deployment_from_name(graph_deployment_name)
        except client.ApiException as e:
            if e.status == 404:
                raise DynamoGraphDeploymentNotFoundError(
                    deployment_name=graph_deployment_name,
                    namespace=self.current_namespace,
                )
            raise

    def update_service_replicas(
        self, graph_deployment_name: str, service_name: str, replicas: int
    ) -> None:
        """
        Update replicas for a service using Scale subresource when DGDSA exists.
        Falls back to DGD patch for backward compatibility with older operators.

        Args:
            graph_deployment_name: Name of the DynamoGraphDeployment
            service_name: Name of the service in DGD.spec.services
            replicas: Desired number of replicas
        """
        # DGDSA naming convention: <dgd-name>-<lowercase-service-name>
        adapter_name = f"{graph_deployment_name}-{service_name.lower()}"

        try:
            # Try to scale via DGDSA Scale subresource
            self.custom_api.patch_namespaced_custom_object_scale(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.current_namespace,
                plural="dynamographdeploymentscalingadapters",
                name=adapter_name,
                body={"spec": {"replicas": replicas}},
            )
            logger.info(f"Scaled DGDSA {adapter_name} to {replicas} replicas")

        except client.ApiException as e:
            if e.status == 404:
                # DGDSA doesn't exist - fall back to DGD patch (old operator)
                logger.info(
                    f"DGDSA {adapter_name} not found, falling back to DGD update"
                )
                self._update_dgd_replicas(graph_deployment_name, service_name, replicas)
            else:
                raise

    def _update_dgd_replicas(
        self, graph_deployment_name: str, service_name: str, replicas: int
    ) -> None:
        """Update replicas directly in DGD (fallback for old operators)"""
        patch = {"spec": {"services": {service_name: {"replicas": replicas}}}}
        self.custom_api.patch_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
            name=graph_deployment_name,
            body=patch,
        )
        logger.info(
            f"Updated DGD {graph_deployment_name} service {service_name} to {replicas} replicas"
        )

    def update_graph_replicas(
        self, graph_deployment_name: str, component_name: str, replicas: int
    ) -> None:
        """
        Update replicas for a service. Now uses DGDSA when available.

        Deprecated: Use update_service_replicas() instead for clarity.
        This method is kept for backward compatibility.
        """
        self.update_service_replicas(graph_deployment_name, component_name, replicas)

    def is_deployment_ready(self, deployment: dict) -> bool:
        """Check if a graph deployment is ready"""

        conditions = deployment.get("status", {}).get("conditions", [])
        ready_condition = next(
            (c for c in conditions if c.get("type") == "Ready"), None
        )

        return ready_condition is not None and ready_condition.get("status") == "True"

    def get_service_replica_status(
        self, deployment: dict, service_name: str
    ) -> tuple[int, bool]:
        """
        Get the actual ready replica count for a service from DGD status.

        Returns:
            tuple[int, bool]: (replica_count, is_stable)
            - replica_count: number of replicas serving traffic (availableReplicas if present, else readyReplicas)
            - is_stable: no rollout is in progress (desired == updated == ready/available)
        """
        # Get desired replicas from spec
        service_spec = (
            deployment.get("spec", {}).get("services", {}).get(service_name, {})
        )
        desired_replicas = service_spec.get("replicas", 0)

        # Get status fields
        service_status = (
            deployment.get("status", {}).get("services", {}).get(service_name, {})
        )
        available = service_status.get("availableReplicas")
        ready = service_status.get("readyReplicas", 0)
        updated = service_status.get("updatedReplicas", 0)

        # availableReplicas takes precedence over readyReplicas for the count
        # refer to ServiceReplicaStatus type (https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/api/v1alpha1/dynamographdeployment_types.go#L157)
        if available is not None:
            traffic_serving_replicas = available
        else:
            traffic_serving_replicas = ready

        # Stable means: desired == updated == ready/available
        # This ensures we're not in a scale-up, scale-down, or rollout
        is_stable = desired_replicas == updated == traffic_serving_replicas

        return traffic_serving_replicas, is_stable

    async def wait_for_graph_deployment_ready(
        self,
        graph_deployment_name: str,
        include_planner: bool = True,
        max_attempts: int = 180,  # default: 30 minutes total
        delay_seconds: int = 10,  # default: check every 10 seconds
    ) -> None:
        """Wait for a graph deployment to be ready.

        Args:
            graph_deployment_name: Name of the DGD to wait for.
            include_planner: If False, skip services with componentType "planner"
                and check per-service readiness instead of the global DGD Ready
                condition. This avoids a circular wait when the planner itself
                is one of the services in the DGD.
            max_attempts: Maximum polling iterations.
            delay_seconds: Seconds between polls.
        """
        for attempt in range(max_attempts):
            await asyncio.sleep(delay_seconds)

            graph_deployment = self.get_graph_deployment(graph_deployment_name)

            if include_planner:
                conditions = graph_deployment.get("status", {}).get("conditions", [])
                ready_condition = next(
                    (c for c in conditions if c.get("type") == "Ready"), None
                )
                if ready_condition and ready_condition.get("status") == "True":
                    return

                logger.info(
                    f"[Attempt {attempt + 1}/{max_attempts}] "
                    f"(status: {ready_condition.get('status') if ready_condition else 'N/A'}, "
                    f"message: {ready_condition.get('message') if ready_condition else 'no condition found'})"
                )
            else:
                services = graph_deployment.get("spec", {}).get("services", {})
                not_ready: list[str] = []
                for svc_name, svc_spec in services.items():
                    if svc_spec.get("componentType", "") == "planner":
                        continue
                    _, is_stable = self.get_service_replica_status(
                        graph_deployment, svc_name
                    )
                    if not is_stable:
                        not_ready.append(svc_name)

                if not not_ready:
                    return

                logger.info(
                    f"[Attempt {attempt + 1}/{max_attempts}] "
                    f"Waiting for services (excluding planner): "
                    f"not ready: {not_ready}"
                )

        raise TimeoutError(
            f"Graph deployment '{graph_deployment_name}' "
            f"is not ready after {max_attempts * delay_seconds} seconds"
        )
