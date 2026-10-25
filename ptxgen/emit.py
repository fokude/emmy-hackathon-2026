from ptxgen.consts import (
    VISIBLE,
    ENTRY,
    PARAM_DECLARATION,
    PARAM_PREFIX,
    U64_TYPE,
    TAB_SPACE,
    SPACE_CHAR,
)

def parse_graph(graph, metadata):
    for kernel in metadata["kernels"]:
        parse_kernel_signature(kernel)
        parse_kernel_body(graph)

def parse_kernel_signature(kernel):
    entry_name = kernel["entry"]
    arg_names = kernel["arg_order"]

    print(f"""
{VISIBLE} {ENTRY} {entry_name}(
{parse_params(arg_names, kernel)}
)
    """)

def parse_kernel_body(graph):
    print(f"""
{{
}}
    """)

def parse_params(arg_names, kernel):
    # ask if metadata is guaranteed to always have same num args as arg order, and if args in args will be in same order as arg order..
    arg_dict = {}
    for arg in kernel["args"]:
        arg_dict[arg["name"]] = arg

    param_strings = []
    for idx, arg_name in enumerate(arg_names):
        arg = arg_dict[arg_name]
        param_string = f"{SPACE_CHAR * TAB_SPACE} {PARAM_DECLARATION} {U64_TYPE} {PARAM_PREFIX + arg["name"]}"
        param_strings.append(param_string)
    return ', \n'.join(param_strings)
