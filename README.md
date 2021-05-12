# DataEngineering

#Starting Ray on each machine

#onHead
ray start --head --port=6379

#on workers
ray start --address=<address> --redis-password='<password>'

#cluster configuration file
cluster_name: local-default
provider:
    type: local
    head_ip: YOUR_HEAD_NODE_HOSTNAME
    worker_ips: [WORKER_NODE_1_HOSTNAME, WORKER_NODE_2_HOSTNAME, ... ]
auth: {ssh_user: YOUR_USERNAME, ssh_private_key: ~/.ssh/id_rsa}
## Typically for local clusters, min_workers == max_workers.
min_workers: 3
max_workers: 3
setup_commands:  # Set up each node.
    - pip install ray torch torchvision tabulate tensorboard
    
    
    
#run config file
ray up tune-default.yaml

#run python code
ray submit tune-default.yaml tune_script.py -- --ray-address=localhost:6379
