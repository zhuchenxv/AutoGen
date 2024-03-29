import socket

config = {}

host = socket.gethostname()
config['host'] = host.lower()

config['data_path'] = '/NAS2020/Workspaces/DMGroup/code/handler_dataset/'
config['env'] = 'gpu'

config['dtype'] = 'float32'
config['scale'] = 0.001
config['minval'] = - config['scale']
config['maxval'] = config['scale']
config['mean'] = 0
config['stddev'] = 0.001
config['sigma'] = config['stddev']
config['const_value'] = 0
config['rnd_type'] = 'uniform'
config['factor_type'] = 'avg'
config['magnitude'] = 3

