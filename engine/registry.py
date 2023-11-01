class Registry:
    mapping = {
        'dataset_name_mapping': {},
    }
    @classmethod
    def register_dataset(cls, name):
        def wrap(dataset_cls):
            cls.mapping["dataset_name_mapping"][name] = dataset_cls
            return dataset_cls

        return wrap

    @classmethod
    def get_dataset_class(cls, name):
        return cls.mapping["dataset_name_mapping"].get(name, None)

registry = Registry()
