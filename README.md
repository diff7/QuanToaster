# QuanToaster - Quatization Optimal Architechture searcher

### Full precision NAS VS Joint quantization:
To search with different bit-widths set desired bit-widths in sr_config.arch.bits. <br>
<br>
Examples:  <br>
1. sr_config.arch.bits = [8,4,2] will perform mixed precision search for all specified bit-widths. <br>
2. sr_config.arch.bits = [32] will perform full precision search. <br>
3. sr_config.arch.bits = [32,4] will perform mixed precision search for all specified bit-widths, proper relu activations for 32 bit-widths will be used. <br>


<br>
<br>

### Batch experiments: <br>
python batch_exp.py -v 0 0.001 0.005 -d gumbel -r 3 -g 3 <br>
-d experiment out directory <br>
-r number of repears <br>
-g gpu number <br>
-v values for flops penalty or regularizations <br>


### To modify search space:
1. Edit genotypes.py to set specific functions for each block. <br>
2. To edit number of operations in each block and the number of repeated cells change sr_config.arch.arch_pattern. <br>