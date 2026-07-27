#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

def generate_register_verilog(module_name="myreg", param_name="DataWidth", input_name="data_in", output_name="data_out"):
    register_template = """\
module {module_name} #(parameter {param_name}=16) (
    input [{param_name}-1:0] {input_name},
    input wire clk,
    input wire rst,
    output reg [{param_name}-1:0] {output_name}
    );
    always@(posedge clk) begin
    if(!rst)
        {output_name}<={input_name};
    else
        {output_name}<=0;
    end
endmodule\n
"""
    return register_template.format(    module_name=module_name,
                                        param_name=param_name,
                                        input_name=input_name,
                                        output_name=output_name)

def generate_logicnets_verilog(module_name: str, input_name: str, input_bits: int, output_name: str, output_bits: int, module_contents: str):
    logicnets_template = """\
module {module_name} (input [{input_bits_1:d}:0] {input_name}, input clk, input rst, output[{output_bits_1:d}:0] {output_name});
{module_contents}
endmodule\n"""
    return logicnets_template.format( module_name=module_name,
                                input_name=input_name,
                                input_bits_1=input_bits-1,
                                output_name=output_name,
                                output_bits_1=output_bits-1,
                                module_contents=module_contents)

def layer_connection_verilog(layer_string: str, input_string: str, input_bits: int, output_string: str, output_bits: int, output_wire=True, register=False):
    if register:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
myreg #(.DataWidth({input_bits})) {layer_string}_reg (.data_in({input_string}), .clk(clk), .rst(rst), .data_out({input_string}w));\n"""
    else:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
assign {input_string}w = {input_string};\n"""
    layer_connection_template += "wire [{output_bits_1:d}:0] {output_string};\n" if output_wire else ""
    layer_connection_template += "{layer_string} {layer_string}_inst (.M0({input_string}w), .M1({output_string}));\n"
    return layer_connection_template.format(    layer_string=layer_string,
                                                input_string=input_string,
                                                input_bits=input_bits,
                                                input_bits_1=input_bits-1,
                                                output_string=output_string,
                                                output_bits_1=output_bits-1)

def _format_lut_hex(val, num_hex_digits):
    s = f"{val:0{num_hex_digits}X}"
    return "_".join(s[i:i+4] for i in range(0, len(s), 4))

def generate_lut_verilog(module_name, input_fanin_bits, output_bits, lut_values):
    """Generate a compact LUT module using hex literals.

    lut_values: list of integers, one per output bit (LSB first).
    lut_values[j] has bit i set iff the output bit j is 1 when M0 == i.
    """
    num_entries = 1 << input_fanin_bits
    hex_digits = num_entries // 4
    lines = []
    if output_bits == 1:
        formatted = _format_lut_hex(lut_values[0], hex_digits)
        lines.append(f"    wire [{num_entries-1}:0] lut = {num_entries}'h{formatted};")
        lines.append(f"    assign M1 = lut[M0];")
    else:
        for j in range(output_bits):
            formatted = _format_lut_hex(lut_values[j], hex_digits)
            lines.append(f"    wire [{num_entries-1}:0] lut_{j} = {num_entries}'h{formatted};")
        for j in range(output_bits):
            lines.append(f"    assign M1[{j}] = lut_{j}[M0];")
    body = "\n".join(lines)
    return (
        f"module {module_name} ( input [{input_fanin_bits-1}:0] M0, output [{output_bits-1}:0] M1 );\n\n"
        f"{body}\n\nendmodule\n"
    )

def generate_neuron_connection_verilog(input_indices, input_bitwidth):
    connection_string = ""
    for i in range(len(input_indices)):
        index = input_indices[i]
        offset = index*input_bitwidth
        for b in reversed(range(input_bitwidth)):
            connection_string += f"M0[{offset+b}]"
            if not (i == len(input_indices)-1 and b == 0):
                connection_string += ", "
    return connection_string

