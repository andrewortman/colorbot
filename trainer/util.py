import argparse

def layer_list_type(allow_empty):
    def validator(value):
        sizes = []
        for subvalue in value.split(","):
            subvalue = int(subvalue)
            if subvalue <= 0:
                raise argparse.ArgumentError("all layer counts must be positive, non-zero integers")

            sizes.append(subvalue)

        if not allow_empty and len(sizes) == 0:
            raise argparse.ArgumentError("must specify at least one layer size")

        return sizes

    return validator
