docker-bash:
	docker run --rm -ti \
		-v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets/ \
		-v ${PWD}:/workspace -w /workspace \
		-v /mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/:/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/ \
		--entrypoint bash  mam10eks/reneuir-tinybert:0.0.1
