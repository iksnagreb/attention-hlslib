# Open a project for this testbench
open_project tb_attention
# Add the source file containing the attention operator top
add_files attention_top.cpp -cflags "--std=c++14 -I$::env(FINN_HLSLIB)
 -I$::env(ATTENTION_HLSLIB) -I$::env(ATTENTION_HLSLIB)/tb"
# Add the testbench main source
add_files -tb attention_tb.cpp -cflags "--std=c++14 -I$::env(FINN_HLSLIB)
 -I$::env(ATTENTION_HLSLIB) -I$::env(ATTENTION_HLSLIB)/tb"
# Configure the top entity of the design, i.e. the attention operator to test
set_top attention_top

# Start a new solution of the design
open_solution sol1
# Configure the target device
# TODO: What is this?
set_part {xczu3eg-sbva484-1-i}
# Create a virtual clock for this solution
create_clock -period 5 -name default

# Run C simulation of the design
csim_design
# Synthesize the HLS design to RTL
csynth_design
# Run cosimulation of the C++ testbench and RTL design
# TODO: Currently this fails, but why?
cosim_design

# Done
exit
