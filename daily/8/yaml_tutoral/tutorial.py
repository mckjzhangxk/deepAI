from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.HYPERPARAMETER_1 = 0.1
# The all important scales for the stuff
_C.TRAIN.SCALES = (2, 4, 8, 16)


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



# C=CN()
#
# C.SOLVER=CN()
# C.SOLVER.LR=1e-3
# C.SOLVER.STEPS=(3000,5000)
#
#
# def get_default_config():
#     return C.clone()
#
# config=get_default_config()
# config.merge_from_file('my.yaml')
# # config.merge_from_list(['SOLVER.MaxIters',3000])
# print(config.dump())