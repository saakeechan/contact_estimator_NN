docker start contact_estimator_train && docker exec -it contact_estimator_train /bin/bash 

python3 utils/csv2numpy.py --config_name config/csv2numpy_config.yaml

python3 src/train.py --config_name config/network_params.yaml

python3 src/test.py --config_name config/test_params.yaml
