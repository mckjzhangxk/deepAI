from tutorial import get_cfg_defaults

if __name__ == '__main__':
    config=get_cfg_defaults()
    config.merge_from_file('my.yaml')
    config.merge_from_list(['SYSTEM.NUM_WORKERS',3000])

    config.freeze()
    print(config)