import argparse

def layer_list_type(allow_empty):
    def validator(value):
        value = value.strip()
        if len(value) == 0:
            if allow_empty:
                return []
            else:
                raise argparse.ArgumentError("must specify at least one layer size")

        sizes = []
        for subvalue in value.split(","):
            subvalue = int(subvalue)
            if subvalue <= 0:
                raise argparse.ArgumentError("all layer counts must be positive, non-zero integers")

            sizes.append(subvalue)

        return sizes

    return validator
