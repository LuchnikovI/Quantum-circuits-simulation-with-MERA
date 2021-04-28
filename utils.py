rotate = lambda x: x[1:] + [x[0]]


def ternary_converter(value, layer):
    result = value
    converted_value = ''
    while result > 0:
        new_result = result // 3
        remainder = result - 3 * new_result
        result = new_result
        converted_value = str(remainder) + converted_value
    return '_' + (layer - len(converted_value)) * '0' + converted_value


def disentangled_nodes(layer):
    return list(zip(map(lambda x: ternary_converter(x, layer), [i for i in range(2, 3 ** layer, 3)]),
                    map(lambda x: ternary_converter(x, layer), rotate([i for i in range(0, 3 ** layer, 3)]))))