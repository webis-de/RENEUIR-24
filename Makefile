docker-bash:
	docker run --rm -ti \
		-v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets/ \
		-v ${PWD}:/workspace -w /workspace \
		-v /mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/:/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/ \
		--entrypoint bash  mam10eks/reneuir-tinybert:0.0.1

jupyter:
	docker run --rm -ti -p 8888:8888 \
		-v ${PWD}:/workspace/ \
		-v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets:ro \
		-w /workspace \
		webis/ir-lab-wise-2023:0.0.4 jupyter notebook --allow-root --ip 0.0.0.0
