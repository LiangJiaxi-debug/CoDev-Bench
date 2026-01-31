import docker
import resource

from argparse import ArgumentParser

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.docker_build import build_instance_images
from swebench.harness.docker_utils import list_images
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool, optional_str


def filter_dataset_to_build(
    dataset: list,
    instance_ids: list | None,
    client: docker.DockerClient,
    force_rebuild: bool,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
):
    """
    Filter the dataset to only include instances that need to be built.

    Args:
        dataset (list): List of instances (usually all of SWE-bench dev/test split)
        instance_ids (list): List of instance IDs to build.
        client (docker.DockerClient): Docker client.
        force_rebuild (bool): Whether to force rebuild all images.
    """
    # Get existing images
    existing_images = list_images(client)
    data_to_build = []

    if instance_ids is None:
        instance_ids = [instance[KEY_INSTANCE_ID] for instance in dataset]

    # Check if all instance IDs are in the dataset
    not_in_dataset = set(instance_ids).difference(
        set([instance[KEY_INSTANCE_ID] for instance in dataset])
    )
    if not_in_dataset:
        raise ValueError(f"Instance IDs not found in dataset: {not_in_dataset}")

    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in instance_ids:
            # Skip instances not in the list
            continue

        # Check if the instance needs to be built (based on force_rebuild flag and existing images)
        spec = make_test_spec(
            instance,
            namespace=namespace,
            instance_image_tag=tag,
            env_image_tag=env_image_tag,
        )
        if force_rebuild:
            data_to_build.append(instance)
        elif spec.instance_image_key not in existing_images:
            data_to_build.append(instance)

    return data_to_build

def retag_and_cleanup(client: docker.DockerClient, old_tag: str):
    """
    将所有 :old_tag 标签镜像重新打上 :latest，然后删除旧标签。
    """
    if not old_tag or old_tag == "latest":
        return
    images = client.images.list()
    retagged = 0
    for img in images:
        for t in img.tags:
            if t.endswith(f":{old_tag}"):
                repo = t.rsplit(":", 1)[0]
                latest_ref = f"{repo}:latest"
                try:
                    client.api.tag(image=img.id, repository=repo, tag="latest")
                    # 删除旧标签
                    client.images.remove(t, noprune=True, force=False)
                    retagged += 1
                    print(f"Retagged {t} -> {latest_ref} (旧标签已删除)")
                except Exception as e:
                    print(f"Retag {t} -> {latest_ref} 失败: {e}")
    if retagged:
        print(f"完成: {retagged} 个镜像从 :{old_tag} 改为 :latest")
    else:
        print(f"未发现需要改标签的镜像 (: {old_tag})")

def main(
    dataset_name,
    split,
    instance_ids,
    max_workers,
    force_rebuild,
    open_file_limit,
    namespace,
    tag,
    env_image_tag,
):
    """
    Build Docker images for the specified instances.

    Args:
        instance_ids (list): List of instance IDs to build.
        max_workers (int): Number of workers for parallel processing.
        force_rebuild (bool): Whether to force rebuild all images.
        open_file_limit (int): Open file limit.
    """
    # Set open file limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # Filter out instances that were not specified
    dataset = load_swebench_dataset(dataset_name, split)
    dataset = filter_dataset_to_build(
        dataset, instance_ids, client, force_rebuild, namespace, tag, env_image_tag
    )

    if len(dataset) == 0:
        print("All images exist. Nothing left to build.")
        if tag and tag != "latest":
            retag_and_cleanup(client, tag)
        if env_image_tag and env_image_tag != "latest":
            retag_and_cleanup(client, env_image_tag)
        return 0

    # Build images for remaining instances
    successful, failed = build_instance_images(
        client=client,
        dataset=dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        namespace=namespace,
        tag=tag,
        env_image_tag=env_image_tag,
    )
    print(f"Successfully built {len(successful)} images")
    print(f"Failed to build {len(failed)} images")

    # 重打标签: 实例镜像
    if tag and tag != "latest":
        retag_and_cleanup(client, tag)
    # 重打标签: 环境镜像
    if env_image_tag and env_image_tag != "latest":
        retag_and_cleanup(client, env_image_tag)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWE-bench/SWE-bench_Lite",
        help="Name of the dataset to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers for parallel processing"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild images"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=8192, help="Open file limit"
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default=None,
        help="Namespace to use for the images (default: None)",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Tag to use for the images"
    )
    parser.add_argument(
        "--env_image_tag", type=str, default=None, help="Environment image tag to use"
    )
    args = parser.parse_args()
    main(**vars(args))
