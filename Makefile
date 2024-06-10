docker-bash:
	docker run --rm -ti -v ${PWD}:/workspace -w /workspace -v /mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/:/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/ --entrypoint bash  mam10eks/reneuir-tinybert:0.0.1
